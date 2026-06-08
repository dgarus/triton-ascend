#include "TritonToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
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
         name.contains("__") || name.startswith("llvm.") ||
         name.startswith("triton.");
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
      .Case("bufferization.alloc_tensor", true)
      .Case("bufferization.to_tensor", true)
      .Case("bufferization.to_buffer", true)
      .Case("bufferization.materialize_in_destination", true)
      .Case("memref.subview", true)
      .Case("memref.reinterpret_cast", true)
      .Case("memref.memory_space_cast", true)
      .Case("memref.copy", true)
      .Case("llvm.mlir.undef", true)
      .Case("llvm.mlir.zero", true)
      .Case("llvm.mlir.constant", true)
      .Case("llvm.mlir.poison", true)
      .Case("llvm.mlir.none", true)
      .Case("tensor.empty", true)
      .Case("tensor.from_elements", true)
      .Default(false);
}

bool isMaybeHelperConstant(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "arith.constant";
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
  return name.endswith(".load") || name.endswith(".store") ||
         name.contains("atomic") || name.endswith(".call") ||
         name == "func.call" || name == "llvm.call";
}

bool isArithmeticOrCastOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.startswith("arith.") || name.endswith(".add") ||
         name.endswith(".sub") || name.endswith(".mul") ||
         name.endswith(".div") || name.endswith(".rem") ||
         name.endswith(".and") || name.endswith(".or") ||
         name.endswith(".xor") || name.endswith(".shl") ||
         name.endswith(".shr") || name.contains(".cmp") ||
         name.contains("cast") || name.contains("ext") || name.contains("trunc");
}

DebugLineLocClass classifyOperation(Operation *op,
                                    const llvm::StringMap<unsigned>
                                        &semanticAnchors) {
  if (isContainerOp(op))
    return DebugLineLocClass::Semantic;
  if (isControlOp(op))
    return DebugLineLocClass::Control;
  if (isExplicitlyMarkedSynthetic(op) || isAlwaysSyntheticOp(op))
    return DebugLineLocClass::Synthetic;

  if (!hasSourceLikeLocation(op))
    return DebugLineLocClass::Synthetic;

  if (isMaybeHelperConstant(op)) {
    SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
    if (!line || semanticAnchors.lookup(getSourceLineKey(line)) > 0)
      return DebugLineLocClass::Synthetic;
  }

  if (isRealMemoryOrCallOp(op) || isArithmeticOrCastOp(op))
    return DebugLineLocClass::Semantic;

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
         isAlwaysSyntheticOp(op);
}

void markOperation(Operation *op, DebugLineLocClass locClass) {
  MLIRContext *context = op->getContext();
  op->setAttr(kClassAttr, StringAttr::get(context, stringifyClass(locClass)));

  Location loc = canonicalizeSourceLoc(op->getLoc());
  if (locClass == DebugLineLocClass::Synthetic) {
    if (!isa<UnknownLoc>(op->getLoc()) && !op->hasAttr(kOriginAttr))
      op->setAttr(kOriginAttr, op->getLoc());
    op->setLoc(UnknownLoc::get(context));
    return;
  }

  if (loc != op->getLoc())
    op->setLoc(loc);
}

void collectSemanticAnchors(Block &block,
                            llvm::StringMap<unsigned> &semanticAnchors) {
  for (Operation &op : block) {
    if (isContainerOp(&op) || isControlOp(&op) || isAlwaysSyntheticOp(&op) ||
        isMaybeHelperConstant(&op))
      continue;
    if (isRealMemoryOrCallOp(&op) || isArithmeticOrCastOp(&op))
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
