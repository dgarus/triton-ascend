#include "TritonToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"

#include <string>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_NORMALIZEDEBUGLINELOCATIONS
#include "ascend/include/TritonToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kOriginAttr = "triton.debug_line.origin";
constexpr llvm::StringLiteral kClassAttr = "triton.debug_line.class";

enum class DebugLineLocClass { Semantic, Control, Synthetic };

struct SourceLine {
  StringAttr file;
  unsigned line = 0;

  explicit operator bool() const { return file && line != 0; }

  bool operator==(const SourceLine &rhs) const {
    return file == rhs.file && line == rhs.line;
  }
};

bool isContainerOp(Operation *op) {
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Case("builtin.module", true)
      .Case("func.func", true)
      .Case("llvm.func", true)
      .Case("tt.func", true)
      .Default(false);
}

bool hasRealFileLineColLoc(Location loc) {
  return static_cast<bool>(loc->findInstanceOf<FileLineColLoc>());
}

SourceLine getSourceLine(Location loc) {
  if (auto fileLoc = loc->findInstanceOf<FileLineColLoc>())
    return {fileLoc.getFilename(), fileLoc.getLine()};
  return {};
}

std::string getSourceLineKey(SourceLine line) {
  if (!line)
    return {};
  std::string key = line.file.getValue().str();
  key += ":";
  key += std::to_string(line.line);
  return key;
}

bool isInternalName(StringRef name) {
  return name.contains("synthetic") || name.contains("lowering") ||
         name.contains("helper") || name.contains("tmp") ||
         name.contains("__") || name.starts_with("llvm.") ||
         name.starts_with("triton.");
}

Location canonicalizeSourceLoc(Location loc) {
  if (isa<UnknownLoc>(loc))
    return loc;

  // If debug scopes were already materialized, keep the fused metadata intact.
  if (loc->findInstanceOf<FusedLocWith<LLVM::DIScopeAttr>>())
    return loc;

  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    Location child = nameLoc.getChildLoc();
    if (hasRealFileLineColLoc(child) &&
        isInternalName(nameLoc.getName().getValue()))
      return child;
  }

  return loc;
}

bool hasSourceLikeLocation(Operation *op) {
  Location loc = canonicalizeSourceLoc(op->getLoc());
  return !isa<UnknownLoc>(loc) && hasRealFileLineColLoc(loc);
}

bool isControlOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
      .Case("scf.for", true)
      .Case("scf.while", true)
      .Case("scf.if", true)
      .Case("scf.condition", true)
      .Case("cf.br", true)
      .Case("cf.cond_br", true)
      .Case("cf.switch", true)
      .Case("llvm.br", true)
      .Case("llvm.cond_br", true)
      .Case("llvm.switch", true)
      .Case("llvm.return", true)
      .Case("func.return", true)
      .Default(false);
}

bool isAlwaysSyntheticOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
      .Case("scf.yield", true)
      .Case("builtin.unrealized_conversion_cast", true)
      .Case("memref.copy", true)
      .Case("llvm.mlir.undef", true)
      .Case("llvm.mlir.zero", true)
      .Case("llvm.mlir.constant", true)
      .Case("llvm.mlir.poison", true)
      .Case("llvm.mlir.none", true)
      .Default(false);
}

bool isMaybeHelperConstant(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "arith.constant";
}

bool isTensorOnlyLinalgFillOp(Operation *op) {
  if (op->getName().getStringRef() != "linalg.fill")
    return false;

  return llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<TensorType>(type); });
}

bool isExplicitlyMarkedSynthetic(Operation *op) {
  if (auto classAttr = op->getAttrOfType<StringAttr>(kClassAttr))
    return classAttr.getValue() == "synthetic";

  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if ((name.contains("synthetic") || name.contains("lowering")) &&
        (isa<UnitAttr>(attr.getValue()) ||
         (isa<BoolAttr>(attr.getValue()) &&
          cast<BoolAttr>(attr.getValue()).getValue())))
      return true;
  }

  return false;
}

bool isRealMemoryOrCallOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.ends_with(".load") || name.ends_with(".store") ||
         name.contains("atomic") || name.ends_with(".call") ||
         name == "func.call" || name == "llvm.call";
}

bool isArithmeticOrCastOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.starts_with("arith.") || name.ends_with(".add") ||
         name.ends_with(".sub") || name.ends_with(".mul") ||
         name.ends_with(".div") || name.ends_with(".rem") ||
         name.ends_with(".and") || name.ends_with(".or") ||
         name.ends_with(".xor") || name.ends_with(".shl") ||
         name.ends_with(".shr") || name.contains(".cmp") ||
         name.contains("cast") || name.contains("ext") || name.contains("trunc");
}

bool isUserVisibleStoreAnchorOp(Operation *op) {
  return op->getName().getStringRef() ==
         "bufferization.materialize_in_destination";
}

bool isValuePreparationOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
             .Case("arith.constant", true)
             .Case("tensor.empty", true)
             .Case("tensor.from_elements", true)
             .Case("bufferization.alloc_tensor", true)
             .Case("bufferization.to_tensor", true)
             .Case("bufferization.to_buffer", true)
             .Default(false) ||
         isTensorOnlyLinalgFillOp(op);
}

bool isAddressComputationOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
             .Case("memref.reinterpret_cast", true)
             .Case("memref.subview", true)
             .Case("memref.memory_space_cast", true)
             .Default(false) ||
         name.contains("cast") || name.contains("ext") ||
         name.contains("trunc");
}

DebugLineLocClass classifyOperation(
    Operation *op, const llvm::StringMap<unsigned> &semanticAnchors) {
  if (isContainerOp(op))
    return DebugLineLocClass::Semantic;

  if (isControlOp(op))
    return DebugLineLocClass::Control;

  if (isExplicitlyMarkedSynthetic(op))
    return DebugLineLocClass::Synthetic;

  if (!hasSourceLikeLocation(op))
    return DebugLineLocClass::Synthetic;

  if (isUserVisibleStoreAnchorOp(op))
    return DebugLineLocClass::Semantic;

  if (isRealMemoryOrCallOp(op))
    return DebugLineLocClass::Semantic;

  SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
  bool hasSemanticAnchorOnSameLine =
      line && semanticAnchors.lookup(getSourceLineKey(line)) > 0;

  if (isValuePreparationOp(op))
    return DebugLineLocClass::Synthetic;

  if (isAddressComputationOp(op))
    return DebugLineLocClass::Synthetic;

  if (isArithmeticOrCastOp(op)) {
    if (hasSemanticAnchorOnSameLine)
      return DebugLineLocClass::Synthetic;
    return DebugLineLocClass::Semantic;
  }

  if (isAlwaysSyntheticOp(op))
    return DebugLineLocClass::Synthetic;

  return DebugLineLocClass::Semantic;
}

StringRef stringifyClass(DebugLineLocClass locClass) {
  switch (locClass) {
  case DebugLineLocClass::Semantic:
    return "semantic";
  case DebugLineLocClass::Control:
    return "control";
  case DebugLineLocClass::Synthetic:
    return "synthetic";
  }
  llvm_unreachable("unknown debug line location class");
}

bool isReorderedBackwardStep(Operation *op, DebugLineLocClass locClass,
                             SourceLine previousLine) {
  if (locClass == DebugLineLocClass::Control || !previousLine)
    return false;

  SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
  if (!line || line.file != previousLine.file || line.line >= previousLine.line)
    return false;

  return isMaybeHelperConstant(op) || isArithmeticOrCastOp(op) ||
         isValuePreparationOp(op) || isAddressComputationOp(op) ||
         isAlwaysSyntheticOp(op);
}

Location makeSyntheticDebugLoc(Operation *op, Location originalLoc) {
  MLIRContext *context = op->getContext();

  Location sourceLoc = canonicalizeSourceLoc(originalLoc);

  if (auto fileLoc = sourceLoc->findInstanceOf<FileLineColLoc>()) {
    // Do not use UnknownLoc here: downstream LLVM/Ascend debug-info lowering
    // expects a file-like location for some operations and may fail on UnknownLoc.
    //
    // DWARF line 0 is used as a non-user source anchor: it preserves a valid file
    // identity while preventing synthetic/helper ops from being associated with
    // real Python source lines.
    return FileLineColLoc::get(
        context,
        fileLoc.getFilename().getValue(),
        /*line=*/0,
        /*column=*/0);
  }

  return originalLoc;
}

void markOperation(Operation *op, DebugLineLocClass locClass) {
  MLIRContext *context = op->getContext();

  Location originalLoc = op->getLoc();
  Location loc = canonicalizeSourceLoc(originalLoc);

  op->setAttr(kClassAttr, StringAttr::get(context, stringifyClass(locClass)));

  if (locClass == DebugLineLocClass::Synthetic) {
    if (!isa<UnknownLoc>(originalLoc) && !op->hasAttr(kOriginAttr))
      op->setAttr(kOriginAttr, originalLoc);

    op->setLoc(makeSyntheticDebugLoc(op, originalLoc));
    return;
  }

  if (loc != originalLoc)
    op->setLoc(loc);
}

void collectSemanticAnchors(Block &block,
                            llvm::StringMap<unsigned> &semanticAnchors) {
  for (Operation &op : block) {
    if (isContainerOp(&op) || isControlOp(&op) ||
        isExplicitlyMarkedSynthetic(&op) || !hasSourceLikeLocation(&op))
      continue;

    if (isUserVisibleStoreAnchorOp(&op) || isRealMemoryOrCallOp(&op))
      if (SourceLine line = getSourceLine(canonicalizeSourceLoc(op.getLoc())))
        ++semanticAnchors[getSourceLineKey(line)];
  }
}

struct NormalizeDebugLineLocationsPass
    : public triton::impl::NormalizeDebugLineLocationsBase<
          NormalizeDebugLineLocationsPass> {
  void processBlock(Block &block) {
    llvm::StringMap<unsigned> semanticAnchors;
    collectSemanticAnchors(block, semanticAnchors);

    SourceLine previousLine;
    for (Operation &op : block) {
      if (isContainerOp(&op))
        continue;

      DebugLineLocClass locClass = classifyOperation(&op, semanticAnchors);
      if (isReorderedBackwardStep(&op, locClass, previousLine))
        locClass = DebugLineLocClass::Synthetic;

      markOperation(&op, locClass);

      if (locClass == DebugLineLocClass::Semantic ||
          locClass == DebugLineLocClass::Control)
        if (SourceLine line = getSourceLine(op.getLoc()))
          previousLine = line;
    }
  }

  void processRegions(Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        processBlock(block);
        for (Operation &nestedOp : block)
          processRegions(&nestedOp);
      }
    }
  }

  void runOnOperation() override { processRegions(getOperation()); }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createNormalizeDebugLineLocationsPass() {
  return std::make_unique<NormalizeDebugLineLocationsPass>();
}
