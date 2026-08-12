# Phase 8 — Multi-GPU: Design and Honest Scope

## Hardware reality on this machine

This project is developed on a single RTX 4060 laptop GPU (`nvidia-smi -L`
shows exactly one device) with no NCCL installed. Real multi-GPU execution
— tensor parallelism, pipeline parallelism, collective communication —
cannot be *verified* here the way every other phase in this codebase has
been: by building it and running it against a CPU or single-GPU reference.
Writing untested distributed-systems code and calling it "done" would be
exactly the kind of unverified, likely-buggy addition the rest of this
session has been working to avoid (see the two real bugs `graph_topo_order`
and the diamond-graph test caught in already-shipped code — code that
*was* tested, just not thoroughly enough). So: this phase is a design doc
covering the real architecture, plus one piece that genuinely can be built
and measured on a single device today. Full implementation is future work,
gated on multi-GPU hardware.

## How real multi-GPU training actually works

### Collectives: ring all-reduce

The primitive every data- and tensor-parallel scheme above is built on is
all-reduce: N devices each hold a tensor; every device ends up with the
elementwise sum (or other reduction) across all N. The standard
implementation is a **ring all-reduce**: each device's data is split into N
chunks, and the reduction happens in two phases — reduce-scatter (each
device ends up owning the fully-reduced sum for one chunk) then all-gather
(that chunk is broadcast around the ring so every device has the full
result) — 2(N-1) pipelined steps total. This is proven bandwidth-optimal
(Patarasuk & Yuan, 2009) and is what NCCL implements under the hood
[Unpacking NCCL](https://medium.com/@nitin966/unpacking-nccl-a-deep-dive-into-multi-gpu-communication-2b667e77d96d),
[nccl-tests performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
This project has no equivalent — `gpu_exec.c`'s `GpuCtx` is single-device by
construction (one `CUcontext`, one `CUdevice`).

### Tensor parallelism (Megatron-LM style)

Split individual weight matrices across devices rather than replicating the
whole model. For an MLP block (`matmul → activation → matmul`, exactly this
project's `OP_FUSED` shape), the standard split is: column-parallel the
first matmul's weight (each device gets a slice of the output features, no
communication needed before the multiply since inputs are already
replicated), then row-parallel the second matmul's weight (each device
holds a slice of the *input* features and produces a partial sum). Because
the column-parallel layer's sharded output feeds directly into the
row-parallel layer's sharded input, no communication is needed *between*
the two matmuls — only a single all-reduce after the second, summing the
partial results across devices
[Megatron-LM tensor parallelism overview](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/),
[Tensor Parallelism: One Matrix Across a Box of GPUs](https://perform.digital/blogs/tensor-parallelism-explained/).
This maps cleanly onto this project's `OP_FUSED` model: a real
implementation would extend the optimizer to *shard* a fused region's
weight before emitting PTX, tag the resulting node with a device id, and
insert an all-reduce node after the row-parallel half of an MLP block.

### Pipeline parallelism

Split the model by *layer* rather than by matrix — GPU 0 holds layers 1-4,
GPU 1 holds layers 5-8, etc., and activations are sent GPU-to-GPU between
stages. To avoid every GPU but one sitting idle most of the time, the batch
is split into microbatches so stage *k+1* starts on microbatch 1 while
stage *k* is still working on microbatch 2 (GPipe/PipeDream-style
microbatch scheduling). In this codebase's terms: partition the topological
order `graph_topo_order` already produces into contiguous device-id ranges,
and replace the intra-device `map_get`/`map_put` device-pointer lookups in
`gpu_execute` with a cross-device transfer (`cuMemcpyPeer` or a
host-staged copy) whenever a node's consumer is assigned to a different
device than the node itself.

## What's honestly buildable and tested on one GPU today

The one genuinely single-GPU-testable piece of this story is **concurrent
kernel scheduling**: `gpu_execute` currently launches every kernel on the
default stream, so independent branches run serially even when nothing
data-dependent forces that. The diamond graph in `test_diamond_shared_input`
(`gpu_test.c`) is the concrete example — `fused1` and `fused2` both read
`nx` but not each other, and could run concurrently on two streams before
`fused3` (which needs both) waits on them. This is the same "which nodes
have no dependency between them" analysis pipeline parallelism's stage
scheduler needs, just applied within one device instead of across several
— a real, useful building block, not a toy.

## Out of scope (this phase, for real this time)

- Actual multi-device execution, `cuMemcpyPeer`/NVLink transfers, and any
  form of collective communication (all-reduce, broadcast, gather) — all
  require hardware this machine doesn't have to build *and verify*
  correctly. A stub that compiles but was never run against a second
  device is worse than no code: it would look tested and wouldn't be.
- NCCL integration.
- Sharded-weight tensor parallelism in the optimizer (`OP_FUSED` region
  sharding) — designed above, not implemented; needs the multi-device
  transfer path first regardless.

## Sources

- [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations (Patarasuk & Yuan)](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
- [Unpacking NCCL: A Deep Dive into Multi-GPU Communication](https://medium.com/@nitin966/unpacking-nccl-a-deep-dive-into-multi-gpu-communication-2b667e77d96d)
- [nccl-tests performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [Tensor Parallelism and Sequence Parallelism: Detailed Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/)
- [Tensor Parallelism: One Matrix Across a Box of GPUs](https://perform.digital/blogs/tensor-parallelism-explained/)
