/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_LLVM_CONVERSION_PASSES
#define TRITON_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLLVM : Pass<"triton-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Triton to LLVM dialect";
  let constructor = "triton::createTritonToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "tensor::TensorDialect", "arith::ArithDialect"];
}

def NormalizeDebugLineLocations : Pass<"normalize-debug-line-locations", "mlir::ModuleOp"> {
  let summary = "Normalize locations for source-level debug line tables.";
  let description = [{
    This pass classifies operations as semantic, control, or synthetic from the
    perspective of source-level debug line table generation. It preserves source
    locations for semantic/control operations and replaces locations of synthetic
    lowering-only operations with UnknownLoc while preserving original
    provenance in attributes. The pass must not change IR semantics or codegen.
  }];
  let constructor = "triton::createNormalizeDebugLineLocationsPass()";
}

#endif // TRITON_TO_LLVM_CONVERSION_PASSES
