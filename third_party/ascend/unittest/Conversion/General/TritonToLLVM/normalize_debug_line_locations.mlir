// RUN: triton-opt %s --normalize-debug-line-locations --mlir-print-debuginfo | FileCheck %s

#di_file = #llvm.di_file<"scope.py" in "/tmp">
#di_cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "triton", isOptimized = true, emissionKind = LineTablesOnly>
#di_sp = #llvm.di_subprogram<compileUnit = #di_cu, scope = #di_file, name = "scoped", file = #di_file, subprogramFlags = "Definition|Optimized">

module {
  func.func @control_ops(%cond: i1) loc("control.py":1:1) {
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

  func.func @synthetic_glue(%arg0: memref<8xf32>, %idx: index, %cond: i1) loc("glue.py":1:1) {
    %cast = builtin.unrealized_conversion_cast %idx : index to i64 loc("glue.py":2:5)
    %sub = memref.subview %arg0[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("glue.py":3:5)
    scf.if %cond {
    } loc("glue.py":4:5)
    func.return loc("glue.py":5:3)
  }

  func.func @future_line_constant(%arg0: memref<4xf32>) loc("future.py":1:1) {
    %c0 = arith.constant 0 : index loc("future.py":8:5)
    %c1 = arith.constant 1 : index loc("future.py":24:5)
    %v = memref.load %arg0[%c0] : memref<4xf32> loc("future.py":24:7)
    %sum = arith.addi %c0, %c1 : index loc("future.py":13:5)
    func.return loc("future.py":14:3)
  }

  llvm.func @llvm_constant() {
    %0 = llvm.mlir.constant(7 : i32) : i32 loc("llvm.py":20:5)
    llvm.return loc("llvm.py":21:3)
  } loc("llvm.py":1:1)

  func.func @nameloc_is_source(%arg0: memref<4xf32>) loc("name.py":1:1) {
    %c0 = arith.constant 0 : index loc("name.py":2:5)
    %v = memref.load %arg0[%c0] : memref<4xf32> loc("trace_ptr"("name.py":7:9))
    func.return loc("name.py":8:3)
  }

  func.func @fused_scope_is_preserved(%arg0: memref<4xf32>) loc("scope.py":1:1) {
    %c0 = arith.constant 0 : index loc("scope.py":2:5)
    %v = memref.load %arg0[%c0] : memref<4xf32> loc(fused<#di_sp>["scope.py":7:9])
    func.return loc("scope.py":8:3)
  }

  func.func @shape_is_unchanged(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %idx: index) -> i32 loc("shape.py":1:1) {
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
// CHECK-SAME: loc("control.py":2:3)
// CHECK: scf.if
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc("control.py":3:5)
// CHECK: scf.for
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc("control.py":5:5)
// CHECK: scf.yield
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = loc("control.py":5:9)
// CHECK-SAME: loc(unknown)

// CHECK-LABEL: func.func @synthetic_glue
// CHECK: builtin.unrealized_conversion_cast
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = loc("glue.py":2:5)
// CHECK-SAME: loc(unknown)
// CHECK: memref.subview
// CHECK-SAME: memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = loc("glue.py":3:5)
// CHECK-SAME: loc(unknown)

// CHECK-LABEL: func.func @future_line_constant
// CHECK: arith.constant 1
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = loc("future.py":24:5)
// CHECK-SAME: loc(unknown)
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc("future.py":24:7)
// CHECK: arith.addi
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = loc("future.py":13:5)
// CHECK-SAME: loc(unknown)

// CHECK-LABEL: llvm.func @llvm_constant
// CHECK: llvm.mlir.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = loc("llvm.py":20:5)
// CHECK-SAME: loc(unknown)

// CHECK-LABEL: func.func @nameloc_is_source
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc("trace_ptr"("name.py":7:9))

// CHECK-LABEL: func.func @fused_scope_is_preserved
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(fused<#di_sp>["scope.py":7:9])

// CHECK-LABEL: func.func @shape_is_unchanged(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[ARG1:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[IDX:[A-Za-z0-9_]+]]: index
// CHECK: %[[SUB:.*]] = memref.subview %[[ARG0]][%[[IDX]]] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: loc(unknown)
// CHECK: %[[V:.*]] = memref.load %[[ARG1]][%[[IDX]]] : memref<8xf32>
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK: %[[R:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK: return %[[R]] : i32
// CHECK-SAME: triton.debug_line.class = "control"
