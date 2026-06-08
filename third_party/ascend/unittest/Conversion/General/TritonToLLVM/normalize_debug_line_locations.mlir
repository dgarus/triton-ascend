// RUN: triton-opt %s --normalize-debug-line-locations --mlir-print-debuginfo | FileCheck %s

#di_file = #llvm.di_file<"scope.py" in "/tmp">
#di_cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "triton", isOptimized = true, emissionKind = LineTablesOnly>
#di_sp = #llvm.di_subprogram<compileUnit = #di_cu, scope = #di_file, name = "scoped", file = #di_file, subprogramFlags = "Definition|Optimized">

module {
  func.func @control_ops(%cond: i1) {
    cf.br ^bb1 loc("control.py":2:3)
  ^bb1:
    scf.if %cond {
    } else {
    } loc("control.py":3:5)
    %c0 = arith.constant 0 : index loc("control.py":4:5)
    %c1 = arith.constant 1 : index loc("control.py":4:9)
    scf.for %i = %c0 to %c1 step %c1 {
      scf.yield loc("control.py":5:9)
    } loc("control.py":5:5)
    func.return loc("control.py":6:3)
  }

  func.func @synthetic_glue(%arg0: memref<8xf32>, %idx: index, %cond: i1) {
    %cast = builtin.unrealized_conversion_cast %idx : index to i64 loc("glue.py":2:5)
    %sub = memref.subview %arg0[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("glue.py":3:5)
    scf.if %cond {
    } loc("glue.py":4:5)
    func.return loc("glue.py":5:3)
  }

  func.func @future_line_constant(%arg0: memref<4xf32>) {
    %c0 = arith.constant 0 : index loc("future.py":8:5)
    %c1 = arith.constant 1 : index loc("future.py":24:5)
    %v = memref.load %arg0[%c0] : memref<4xf32> loc("future.py":24:7)
    %sum = arith.addi %c0, %c1 : index loc("future.py":13:5)
    func.return loc("future.py":14:3)
  }

  llvm.func @llvm_constant() {
    %0 = llvm.mlir.constant(7 : i32) : i32 loc("llvm.py":20:5)
    llvm.return loc("llvm.py":21:3)
  }

  func.func @nameloc_is_source(%arg0: memref<4xf32>) {
    %c0 = arith.constant 0 : index loc("name.py":2:5)
    %v = memref.load %arg0[%c0] : memref<4xf32> loc("trace_ptr"("name.py":7:9))
    func.return loc("name.py":8:3)
  }

  func.func @fused_scope_is_preserved(%arg0: memref<4xf32>) {
    %c0 = arith.constant 0 : index loc("scope.py":2:5)
    %v = memref.load %arg0[%c0] : memref<4xf32> loc(fused<#di_sp>["scope.py":7:9])
    func.return loc("scope.py":8:3)
  }

  func.func @shape_is_unchanged(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %idx: index) -> i32 {
    %sub = memref.subview %arg0[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("shape.py":2:5)
    %v = memref.load %arg1[%idx] : memref<8xf32> loc("shape.py":3:5)
    %c1 = arith.constant 1 : i32 loc("shape.py":4:5)
    %r = arith.addi %c1, %c1 : i32 loc("shape.py":5:5)
    return %r : i32 loc("shape.py":6:3)
  }
}

// CHECK-LABEL: func.func @control_ops
// CHECK: cf.br
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[$CONTROL_BR_LOC:[A-Za-z0-9_]+]])
// CHECK: scf.if
// CHECK: } {triton.debug_line.class = "control"} loc(#[[$CONTROL_IF_LOC:[A-Za-z0-9_]+]])
// CHECK: scf.for
// CHECK: } {triton.debug_line.class = "control"} loc(#[[$CONTROL_FOR_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[$CONTROL_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @synthetic_glue
// CHECK: builtin.unrealized_conversion_cast
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: loc(#[[$UNKNOWN_LOC:[A-Za-z0-9_]+]])
// CHECK: memref.subview
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[$UNKNOWN_LOC]])

// CHECK-LABEL: func.func @future_line_constant
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: 0 : index
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: 1 : index
// CHECK-SAME: loc(#[[$UNKNOWN_LOC]])
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[$FUTURE_LOAD_LOC:[A-Za-z0-9_]+]])
// CHECK: arith.addi
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: : index
// CHECK-SAME: loc(#[[$UNKNOWN_LOC]])

// CHECK-LABEL: llvm.func @llvm_constant
// CHECK: llvm.mlir.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: loc(#[[$UNKNOWN_LOC]])

// CHECK-LABEL: func.func @nameloc_is_source
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[$TRACE_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @fused_scope_is_preserved
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[$FUSED_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @shape_is_unchanged(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[ARG1:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[IDX:[A-Za-z0-9_]+]]: index
// CHECK-SAME: -> i32
// CHECK: %[[SUB:.*]] = memref.subview %[[ARG0]][%[[IDX]]] [4] [1]
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[$UNKNOWN_LOC]])
// CHECK: %[[V:.*]] = memref.load %[[ARG1]][%[[IDX]]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : memref<8xf32>
// CHECK-SAME: loc(#[[$SHAPE_LOAD_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[C1:.*]] = arith.constant
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: 1 : i32
// CHECK-SAME: loc(#[[$SHAPE_CONST_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[R:.*]] = arith.addi %[[C1]], %[[C1]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : i32
// CHECK-SAME: loc(#[[$SHAPE_ADDI_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: %[[R]] : i32
// CHECK-SAME: loc(#[[$SHAPE_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[$UNKNOWN_LOC]] = loc(unknown)
// CHECK-DAG: #[[$CONTROL_BR_LOC]] = loc("control.py":2:3)
// CHECK-DAG: #[[$CONTROL_IF_LOC]] = loc("control.py":3:5)
// CHECK-DAG: #[[$CONTROL_FOR_LOC]] = loc("control.py":5:5)
// CHECK-DAG: #[[$CONTROL_RETURN_LOC]] = loc("control.py":6:3)
// CHECK-DAG: #[[$FUTURE_LOAD_LOC]] = loc("future.py":24:7)
// CHECK: #[[$NAME_FILE_LOC:[A-Za-z0-9_]+]] = loc("name.py":7:9)
// CHECK: #[[$SCOPE_FILE_LOC:[A-Za-z0-9_]+]] = loc("scope.py":7:9)
// CHECK: #[[$TRACE_LOC]] = loc("trace_ptr"(#[[$NAME_FILE_LOC]]))
// CHECK: #[[$FUSED_LOC]] = loc(fused<{{.*}}>[#[[$SCOPE_FILE_LOC]]])
// CHECK-DAG: #[[$SHAPE_LOAD_LOC]] = loc("shape.py":3:5)
// CHECK-DAG: #[[$SHAPE_CONST_LOC]] = loc("shape.py":4:5)
// CHECK-DAG: #[[$SHAPE_ADDI_LOC]] = loc("shape.py":5:5)
// CHECK-DAG: #[[$SHAPE_RETURN_LOC]] = loc("shape.py":6:3)
