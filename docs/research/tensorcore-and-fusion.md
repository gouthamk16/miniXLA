# Beating PyTorch: tensor cores, and where fusion actually wins

`gemm-optimization.md` closed with 53.0% of cuBLAS on CUDA cores and named
the next lever without pulling it: cuBLAS routes FP32 `sgemm` through TF32
tensor cores on this Ada (sm_89) GPU by default, and no amount of CUDA-core
tuning closes a different-hardware-path gap. This doc is the research pass
before attempting that, plus a second, more honest question: raw GEMM
GFLOPS is the single hardest place to beat a vendor BLAS, so where does a
compiler like this one actually have a structural edge?

## 1. TF32 tensor cores: the fragment layout, verified from source

The risk this repo already flagged (`autoresearch-gemm.md`, `README.md`):
a fragment-layout mistake in a `mma.sync` kernel produces **plausible-
looking wrong numbers**, not a build error, because each thread silently
computes a real dot product over the wrong operand elements. Two blog-post
summaries fetched during this research disagreed with each other (one
described the `m16n8k16` fp16 fragment shape, mislabeled as `m16n8k8` tf32)
before the actual PTX ISA table was found. That near-miss is the whole
justification for the rule below: **do not implement from a blog's
paraphrase of the fragment layout.**

The authoritative source is the PTX ISA doc itself, section 9.7.15.5.7
("Matrix Fragments for `mma.m16n8k8`"), `.tf32` subsection — fetched and
read directly (not summarized) at
`https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688`.
Transcribed verbatim, not adapted:

```
groupID           = %laneid >> 2
threadID_in_group = %laneid % 4

Multiplicand A (four .b32 registers a0..a3, each one .tf32 element):
  row = groupID      for a0, a2
        groupID + 8   for a1, a3
  col = threadID_in_group      for a0, a1
        threadID_in_group + 4  for a2, a3

Multiplicand B (two .b32 registers b0, b1):
  row = threadID_in_group      for b0
        threadID_in_group + 4  for b1
  col = groupID  (both)

Accumulator C/D (four .f32 registers c0..c3):
  row = groupID      for c0, c1
        groupID + 8   for c2, c3
  col = (threadID_in_group * 2) + (i & 0x1)   for ci, i = {0,..,3}
```

Cross-checked against a second, independent source: CUTLASS's
`include/cutlass/arch/mma_sm80.h` (`NVIDIA/cutlass` on GitHub), whose
`Mma<gemm::GemmShape<16,8,8>, tfloat32_t, tfloat32_t, float, OpMultiplyAdd>`
specialization confirms the register *counts* this table implies:
`FragmentA = Array<tfloat32_t,4>`, `FragmentB = Array<tfloat32_t,2>`,
`FragmentC = Array<float,4>`, and the inline PTX register list order
(`{%0..%3}, {%4..%7}, {%8,%9}, {%10..%13}`) matches the instruction string
below. Two independent sources agreeing on shape is the bar this repo set
for itself before touching this instruction; it's now met.

Instruction: `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. Inputs
must be pre-converted from `.f32` to `.tf32` (itself stored in a `.b32`
register — "a register variable containing tf32 data must be declared with
`.b32` type", same PTX ISA doc) via `cvt.rna.tf32.f32 d, a;`. `.rna`
(round-to-nearest, ties away from zero) is the mode CUTLASS uses and the
PTX ISA docs list as the non-saturating default for this conversion;
`.rz` is the other documented option and keeps *less* mantissa, so `.rna`
is the correct choice here, not just the first one found.

**Precision implication for the correctness gate.** TF32 keeps the same 8
exponent bits as FP32 but only 10 explicit mantissa bits (11 with the
implicit leading bit) versus FP32's 23. Per-multiply relative error is
therefore of order 2^-11 (~5e-4), and accumulating over K in FP32 (the
accumulator stays FP32; the ISA table above confirms C/D is `.f32`, not
`.tf32`) doesn't fix the per-element rounding, it just avoids compounding a
*second* rounding on top of it. This means the existing GPU-vs-CPU
tolerance used for the CUDA-core kernel (`1e-4`–`1e-2` depending on size,
per `autoresearch-gemm.md`) is very likely too tight for a TF32 path at
larger K, and reusing it without checking would either produce spurious
"crash" verdicts (masking a working kernel) or, worse, someone widening it
ad hoc later without understanding why. The tensor-core experiments below
carry their own, explicitly wider and explicitly justified tolerance.

## 2. Warp-tiling as the CUDA-core-only alternative (siboehm kernel 10)

siboehm's kernel 9 → 10 step (84.8% → 93.7% of cuBLAS, [siboehm's
worklog](https://siboehm.com/articles/22/CUDA-MMM)) inserts an explicit
warp-tile level between the existing block-tile and thread-tile: each
warp owns a `(WSUBN×WNITER)×(WSUBM×WMITER)` chunk of the block's output,
subdivided further into `WNITER×WMITER` sub-tiles of the current `TM×TN`
thread tile. The stated payoff is occupancy (warps free to run on
different warp schedulers within an SM) and shared-memory bank-conflict
locality (conflicts only matter within a warp, so warp-local data layout
can be tuned independently of the rest of the block) — not a new hardware
path, just better use of the same CUDA cores this kernel already uses.

This is real and lower-risk than tensor cores (worst case: a wrong index
computation is a crash or an obviously-wrong number from a badly-strided
load, not a plausible one — the failure mode that made tensor cores scary
doesn't really apply here). But it tops out at 93.7% of *CUDA-core*
cuBLAS, and cuBLAS on this GPU isn't running CUDA-core sgemm — it's running
TF32 tensor cores at effectively its own peak (99.6% of its own number, per
the README table, since PyTorch and cuBLAS take the same path). Warp-
tiling alone cannot close that gap; only a tensor-core path can. Given
that, and given §1 above resolved the layout risk with a primary source,
this session prioritizes the tensor-core experiment first and treats warp-
tiling as the fallback if tensor cores don't pan out, rather than the
other way around — a change from `gemm-optimization.md`'s original
ordering, made because the blocking risk that ordering was hedging against
turned out to be resolvable.

## 3. Where a small compiler can actually beat an eager framework: fusion, not FLOPs

Raw-GEMM GFLOPS is a contest against cuBLAS's most-tuned primitive, running
on the exact same hardware path once tensor cores are in play — a photo
finish at best, for a kernel with one afternoon of tuning against a decade
of one. The more honest structural advantage this codebase's whole design
already has is elsewhere: **`OP_FUSED` is one kernel launch and zero
materialized intermediates for `matmul → add → relu`; PyTorch eager is
three kernel launches and two intermediate tensors.** At sizes small enough
that launch overhead and intermediate-allocation cost dominate (the
128–512 range, exactly where the existing README table shows even PyTorch
itself losing the most ground to cuBLAS's own peak — a tell that these
sizes are overhead-bound, not compute-bound), a single fused launch has a
real shot at winning on wall-clock latency for this exact op pattern, even
running a slower-per-FLOP kernel underneath.

This mirrors why `llm.c` (Karpathy) and `ggml`/`llama.cpp` — both small,
close-to-the-metal C codebases, much closer to this project's own style
than CUTLASS's C++ template stack — spend real engineering effort on
kernel fusion and avoiding intermediate materialization even where their
individual GEMM kernels don't out-FLOP cuBLAS/cuDNN either: for
memory-bound or launch-bound workloads (small batch, short sequence,
elementwise-heavy graphs), fusion wins on latency independent of raw
compute throughput. `torch.compile`'s own existence is evidence for the
same point from the other direction — PyTorch's own team added a fusing
compiler on top of eager mode specifically because eager's per-op launch
and materialization cost is a real, measurable tax; the fair comparison
for MiniXLA's structural claim is therefore MiniXLA vs. **both** PyTorch
eager and `torch.compile`, not eager alone.

## 4. Memory: what "beating" PyTorch would even mean here

`torch.cuda.max_memory_allocated()` measures PyTorch's *caching allocator's*
high-water mark, not raw kernel-necessary bytes — the caching allocator
intentionally holds freed blocks for reuse rather than returning them to
the driver, so a naive comparison against MiniXLA's malloc-per-buffer
`gpu_exec.c` (which does free intermediates at last-use, per the Phase 7
design doc, but calls `cuMemFree` immediately rather than pooling) isn't
automatically apples-to-apples in either direction: PyTorch's number can
look inflated (it's holding reusable blocks) or MiniXLA's can look
inflated (repeated `cuMemAlloc`/`cuMemFree` churn versus one steady-state
pool) depending on the access pattern being measured. tinygrad's own memory
planner is instructive here: its linear-scan buffer allocator (greedy
left-to-right scan over sorted live intervals, O(N log N), a free-pool of
released slots reused for later intervals — see
[tinygrad docs, Developer overview](https://docs.tinygrad.org/developer/developer/))
is exactly the kind of graph-level liveness reasoning `gpu_exec.c`'s
`last_use` array already does for *freeing*; the piece MiniXLA doesn't yet
have is reusing a same-size freed device buffer for the *next*
allocation instead of round-tripping through the CUDA driver's own
allocator for every fused node. Whether that's worth adding depends on
whether repeated-execution profiling actually shows `cuMemAlloc`/`cuMemFree`
as a measurable fraction of wall time for this graph size — the plan below
measures before building it, per this repo's own minimalism rule.

## Plan, in priority order

1. **TF32 tensor-core kernel**, additive (`emit_ptx_tensorcore` in `ptx.c`,
   gated behind its own entry points in `gpu_exec.c`, existing
   `emit_ptx_blocked` untouched). Simplest-correct first: one warp per
   block, one `mma.sync.aligned.m16n8k8` per 8-deep K-step, fragment
   elements read straight from global memory (no shared-memory blocking
   yet) — minimizes the surface area for a fragment-layout bug before
   anything else is added on top of it. Correctness-gated with an
   explicitly TF32-appropriate tolerance (§1), not the CUDA-core kernel's
   tolerance reused blind. If this holds, a shared-memory-blocked,
   multi-warp-per-block follow-up is the natural next experiment, same
   autoresearch loop.
2. **Re-measure the full size table** (128 through 2048+, plus larger if
   memory allows) once (1) is correctness-verified: tensor cores may help
   more at some sizes than others, and the honest thing is to report the
   real shape of that curve rather than a single cherry-picked number.
3. **Fused-graph latency vs. PyTorch eager and `torch.compile`** for
   `relu(matmul(a,b)+c)` across the same size sweep, wall-clock per call
   (not just GFLOPS) — the metric where fusion's structural advantage
   should actually show up, per §3.
4. **Peak memory comparison** for the same op, understanding what's
   actually being compared per §4, with a buffer-pool addition to
   `gpu_exec.c` only if profiling shows allocator churn is load-bearing at
   the sizes tested — not built speculatively.
5. **Warp-tiling** (§2) as the fallback if (1) proves infeasible to verify
   with confidence in the time available, or as a later CUDA-core-only
   follow-up regardless, since it's real and independent of whether tensor
   cores land.

## Sources

- [PTX ISA 9.3, §9.7.15.5.7 "Matrix Fragments for mma.m16n8k8"](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688) (fetched and read directly, not summarized — the primary source for the fragment layout used in this session's kernel)
- [PTX ISA 9.3, `cvt` instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) (`cvt.rna.tf32.f32` syntax and rounding-mode semantics)
- [NVIDIA/cutlass, `include/cutlass/arch/mma_sm80.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm80.h) (independent cross-check: fragment register counts and PTX register-list order for the same instruction)
- [siboehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"](https://siboehm.com/articles/22/CUDA-MMM) (kernel 10, warp-tiling)
- [tinygrad docs, Developer overview](https://docs.tinygrad.org/developer/developer/) (linear-scan memory planning as a model for graph-level buffer reuse)
- `docs/research/gemm-optimization.md`, `docs/research/autoresearch-gemm.md` (this repo's own prior research and the autoresearch mechanism this session continues)
