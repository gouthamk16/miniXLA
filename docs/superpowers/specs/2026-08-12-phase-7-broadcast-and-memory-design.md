# Phase 7: Broadcast Bias-Add & GPU Buffer Liveness

Two independent, narrowly-scoped Phase 7 deliverables from `idea.md`
("Memory Optimization: buffer reuse, tensor lifetime analysis"): broadcast
support for the single most common real shape mismatch in an MLP, and
freeing GPU intermediates as soon as their last consumer runs instead of
holding the whole graph's device memory alive until the end.

## 1. Broadcast bias-add

Phase 3's design doc explicitly deferred this: "Broadcasting in fused
ADD" was listed under Out of scope. `y = x @ W + b` with `b` shaped `[N]`
against a `[M, N]` matmul output is the standard MLP bias pattern; without
it, every constructed graph needs a pre-broadcast bias tensor, which is
both unrealistic and blocks the autodiff MLP gradient-check test from
looking like real usage.

**Scope**: NumPy's trailing-suffix broadcast rule only: `b`'s shape must
equal a trailing suffix of `a`'s shape (`b->ndim <= a->ndim` and
`b->shape[i] == a->shape[a->ndim - b->ndim + i]` for all `i`). Not full
N-dimensional broadcasting (arbitrary size-1 dims anywhere). That's a
different, considerably more complex feature nothing in this codebase
currently needs.

**Where it's implemented:**
- `tensor_add` (`tensor.c`): exact-shape path unchanged; falls through to
  the broadcast check only when shapes differ. Because both tensors are
  row-major and `b` matches `a`'s trailing dims exactly, `a`'s data is just
  `b`'s block repeated `a->size / b->size` times: `result[i] = a[i] +
  b[i % b->size]` is correct and needs no per-dimension stride walk.
- `eval_fused` (`graph.c`, the CPU fused executor): the epilogue `OP_ADD`
  step currently indexes its operand at the same `[i, j]` cell as the
  matmul output. Detects a broadcast operand by `operand->output->size !=
  M*N` and indexes `[j]` (mod `N`) instead: same trick as `tensor_add`,
  applied per-cell instead of as a whole-tensor pass.
- **GPU (`ptx.c`, `runtime.c`, `gpu_exec.c`): explicitly not implemented.**
  The PTX epilogue emitter and both GPU runtimes assume every `OP_ADD`
  operand is full `[M, N]`: extending that correctly means the emitter
  needs to know at codegen time whether each operand is broadcast (to emit
  a `j`-only index instead of `t`) and both runtimes need to upload the
  right byte count. That's real, GPU-side codegen surface I can't verify
  as thoroughly as the CPU path in this session. Rather than leave that as
  a silent correctness gap (the exact "silently reads/writes past a
  smaller buffer" bug class already fixed once for batched matmul), both
  `gpu_run_fused` (`runtime.c`) and `gpu_execute` (`gpu_exec.c`) now check
  each `OP_ADD` operand's size against `M*N` before uploading and fail
  loudly (`stderr` + `NULL`/`0`, same convention as every other failure
  path in those files) instead of computing silently-wrong output. Fusing
  a broadcast-bias graph is therefore CPU-only until GPU broadcast codegen
  is built as its own follow-up.

## 2. GPU buffer liveness in `gpu_execute`

`gpu_execute` (`gpu_exec.c`) currently uploads/allocates every node's
device buffer once and frees the whole set at the end: an N-kernel chain
holds N buffers' worth of device memory simultaneously even though most
intermediates are dead the moment their one consumer has run.

**Change**: before the execution loop, compute each node's last-use index
in the topological execution order (the index of the last node in that
order whose `inputs[]` contains it; the root's own index if it has no
consumers within the graph, since the root's result is what gets copied
out). After launching a node's kernel, free any input's device buffer
immediately if the node just processed was that input's last consumer.
`OP_INPUT` uploads are only freed this way too: a graph reusing the same
input node across two matmuls correctly keeps it alive until both have
run, then frees it before the tail of a longer graph completes.

This is a pure resource-management change (same launches, same math,
same results), verified by the existing `gpu_test.c` suite (particularly
`test_two_layer`, the only current multi-kernel graph) continuing to pass
with identical output, plus a peak-device-memory check via
`cuMemGetInfo` before/after to confirm intermediates are actually being
freed mid-run rather than at the end.

## Out of scope

- General N-D broadcasting for `tensor_add`.
- GPU codegen for broadcast epilogue operands (see above, guarded, not
  implemented).
- A general buffer allocator/arena; last-use freeing via `cuMemFree` is
  sufficient at this project's graph sizes (single-digit to low-double-
  digit kernel counts): a pooling allocator would be optimizing for a
  problem this project doesn't have yet.
