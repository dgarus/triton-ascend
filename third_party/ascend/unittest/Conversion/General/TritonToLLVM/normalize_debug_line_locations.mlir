// RUN: triton-opt %s -split-input-file --normalize-debug-line-locations --allow-unregistered-dialect --mlir-print-debuginfo | FileCheck %s

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

  func.func @source_line_write_anchor(%trace: memref<8xi32>, %counter: memref<1xi32>) {
    %c0 = arith.constant 0 : index loc("loop_reordering.py":23:21)
    %ticket = memref.load %counter[%c0] : memref<1xi32> loc("ticket"("loop_reordering.py":23:21))
    %ticket_idx = arith.index_cast %ticket : i32 to index loc("ticket"("loop_reordering.py":23:21))
    %c200_i32 = arith.constant 200 : i32 loc("loop_reordering.py":24:33)
    %empty = tensor.empty() : tensor<1xi32> loc("loop_reordering.py":24:33)
    %filled = linalg.fill ins(%c200_i32 : i32) outs(%empty : tensor<1xi32>) -> tensor<1xi32> loc("loop_reordering.py":24:33)
    %view = memref.reinterpret_cast %trace to offset: [%ticket_idx], sizes: [1], strides: [1] : memref<8xi32> to memref<1xi32, strided<[1], offset: ?>> loc("loop_reordering.py":24:33)
    bufferization.materialize_in_destination %filled in writable %view : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> () loc("loop_reordering.py":24:33)
    %c1_i32 = arith.constant 1 : i32 loc("loop_reordering.py":25:35)
    %next_ticket = arith.addi %ticket, %c1_i32 : i32 loc("loop_reordering.py":25:35)
    %inserted = tensor.insert %next_ticket into %empty[%c0] : tensor<1xi32> loc("loop_reordering.py":25:26)
    bufferization.materialize_in_destination %inserted in writable %counter : (tensor<1xi32>, memref<1xi32>) -> () loc("loop_reordering.py":25:26)
    func.return loc("loop_reordering.py":26:3)
  }

  func.func @memref_fill_anchor(%dst: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32 loc("memfill.py":3:5)
    linalg.fill ins(%cst : f32) outs(%dst : memref<4xf32>) loc("memfill.py":3:7)
    func.return loc("memfill.py":4:3)
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
// CHECK-SAME: loc(#[[CONTROL_BR_LOC:[A-Za-z0-9_]+]])
// CHECK: scf.if
// CHECK: } {triton.debug_line.class = "control"} loc(#[[CONTROL_IF_LOC:[A-Za-z0-9_]+]])
// CHECK: scf.for
// CHECK: } {triton.debug_line.class = "control"} loc(#[[CONTROL_FOR_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[CONTROL_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @synthetic_glue
// CHECK: builtin.unrealized_conversion_cast
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: loc(#[[GLUE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: memref.subview
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[GLUE_SYNTH_LOC]])

// CHECK-LABEL: func.func @source_line_write_anchor
// CHECK: %[[TICKET:[A-Za-z0-9_]+]] = memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : memref<1xi32>
// CHECK-SAME: loc(#[[LOOP_TICKET_LOC:[A-Za-z0-9_]+]])

// CHECK: %[[TICKET_IDX:[A-Za-z0-9_]+]] = arith.index_cast %[[TICKET]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: triton.debug_line.origin = #[[LOOP_TICKET_LOC]]
// CHECK-SAME: : i32 to index
// CHECK-SAME: loc(#[[LOOP_WRITE_LOC:[A-Za-z0-9_]+]])

// CHECK: %[[C200:[A-Za-z0-9_]+]] = arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[LOOP_WRITE_LOC]]
// CHECK-SAME: 200 : i32
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC:[A-Za-z0-9_]+]])

// CHECK: %[[EMPTY:[A-Za-z0-9_]+]] = tensor.empty
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[LOOP_WRITE_LOC]]
// CHECK-SAME: tensor<1xi32>
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC]])

// CHECK: %[[FILLED:[A-Za-z0-9_]+]] = linalg.fill
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[LOOP_WRITE_LOC]]
// CHECK-SAME: ins(%[[C200]] : i32) outs(%[[EMPTY]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC]])

// CHECK: %[[VIEW:[A-Za-z0-9_]+]] = memref.reinterpret_cast
// CHECK-SAME: %[[TICKET_IDX]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: memref<8xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[LOOP_WRITE_LOC]])

// CHECK: bufferization.materialize_in_destination %[[FILLED]] in writable %[[VIEW]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[LOOP_WRITE_LOC]])

// CHECK: %[[C1:[A-Za-z0-9_]+]] = arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[LOOP_NEXT_VALUE_LOC:[A-Za-z0-9_]+]]
// CHECK-SAME: 1 : i32
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC]])

// CHECK: %[[NEXT:[A-Za-z0-9_]+]] = arith.addi %[[TICKET]], %[[C1]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: triton.debug_line.origin = #[[LOOP_NEXT_VALUE_LOC]]
// CHECK-SAME: : i32
// CHECK-SAME: loc(#[[LOOP_NEXT_STORE_LOC:[A-Za-z0-9_]+]])

// CHECK: %[[INSERTED:[A-Za-z0-9_]+]] = tensor.insert %[[NEXT]] into %[[EMPTY]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: tensor<1xi32>
// CHECK-SAME: loc(#[[LOOP_NEXT_STORE_LOC]])

// CHECK: bufferization.materialize_in_destination %[[INSERTED]] in writable
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[LOOP_NEXT_STORE_LOC]])

// CHECK-LABEL: func.func @memref_fill_anchor
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: loc(#[[MEMFILL_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: linalg.fill
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: outs
// CHECK-SAME: loc(#[[MEMFILL_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @future_line_constant
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: 0 : index
// CHECK-SAME: loc(#[[FUTURE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: 1 : index
// CHECK-SAME: loc(#[[FUTURE_SYNTH_LOC]])
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FUTURE_LOAD_LOC:[A-Za-z0-9_]+]])
// CHECK: arith.addi
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: : index
// CHECK-SAME: loc(#[[FUTURE_SYNTH_LOC]])

// CHECK-LABEL: llvm.func @llvm_constant
// CHECK: llvm.mlir.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: loc(#[[LLVM_SYNTH_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @nameloc_is_source
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[TRACE_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @fused_scope_is_preserved
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FUSED_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @shape_is_unchanged(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[ARG1:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[IDX:[A-Za-z0-9_]+]]: index
// CHECK-SAME: -> i32
// CHECK: %[[SUB:.*]] = memref.subview %[[ARG0]][%[[IDX]]] [4] [1]
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[SHAPE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[V:.*]] = memref.load %[[ARG1]][%[[IDX]]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : memref<8xf32>
// CHECK-SAME: loc(#[[SHAPE_LOAD_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[C1:.*]] = arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin =
// CHECK-SAME: 1 : i32
// CHECK-SAME: loc(#[[SHAPE_SYNTH_LOC]])
// CHECK: %[[R:.*]] = arith.addi %[[C1]], %[[C1]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : i32
// CHECK-SAME: loc(#[[SHAPE_ADDI_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: %[[R]] : i32
// CHECK-SAME: loc(#[[SHAPE_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[CONTROL_BR_LOC]] = loc("control.py":2:3)
// CHECK-DAG: #[[CONTROL_IF_LOC]] = loc("control.py":3:5)
// CHECK-DAG: #[[CONTROL_FOR_LOC]] = loc("control.py":5:5)
// CHECK-DAG: #[[CONTROL_RETURN_LOC]] = loc("control.py":6:3)
// CHECK-DAG: #[[GLUE_SYNTH_LOC]] = loc("glue.py":0:0)
// CHECK-DAG: #[[LOOP_SYNTH_LOC]] = loc("loop_reordering.py":0:0)
// CHECK-DAG: #[[MEMFILL_SYNTH_LOC]] = loc("memfill.py":0:0)
// CHECK-DAG: #[[MEMFILL_LOC]] = loc("memfill.py":3:7)
// CHECK-DAG: #[[FUTURE_SYNTH_LOC]] = loc("future.py":0:0)
// CHECK-DAG: #[[FUTURE_LOAD_LOC]] = loc("future.py":24:7)
// CHECK-DAG: #[[LLVM_SYNTH_LOC]] = loc("llvm.py":0:0)
// CHECK-DAG: #[[NAME_FILE_LOC:[A-Za-z0-9_]+]] = loc("name.py":7:9)
// CHECK-DAG: #[[SCOPE_FILE_LOC:[A-Za-z0-9_]+]] = loc("scope.py":7:9)
// CHECK-DAG: #[[SHAPE_SYNTH_LOC]] = loc("shape.py":0:0)
// CHECK-DAG: #[[SHAPE_LOAD_LOC]] = loc("shape.py":3:5)
// CHECK-DAG: #[[SHAPE_ADDI_LOC]] = loc("shape.py":5:5)
// CHECK-DAG: #[[SHAPE_RETURN_LOC]] = loc("shape.py":6:3)
// CHECK: #[[TRACE_LOC]] = loc("trace_ptr"(#[[NAME_FILE_LOC]]))
// CHECK: #[[FUSED_LOC]] = loc(fused<{{.*}}>[#[[SCOPE_FILE_LOC]]])
