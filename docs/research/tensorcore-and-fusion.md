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

## Results: tensor-core stream (autoresearch/tensorcore, full log in results.tsv)

| Experiment | GFLOPS @ 2048^3 | % of cuBLAS | Status |
|---|---:|---:|---|
| cuBLAS (this session's fresh measurement) | 7622.9 | 100% | reference |
| CUDA-core kernel (existing, unchanged) | 6177.4 | 81.1% | reference |
| 1. TF32 tensor core, naive (1 warp/block, no smem) | 2934.7 | 38.5% | kept |
| 2. + shared-memory blocking (8 warps/block, 32x32x8 tile) | 3914.8 | 51.4% | kept |
| K-tile 8->32 (4x mma per smem reload) | 2572.7 | 33.8% | discarded (slower, 3 isolated repeats) |
| Block tile 32x32->64x64 (32 warps/block) | -- | -- | discarded (correctness bug, not a perf regression) |

Correctness held at every step (boundary-ragged shapes, ADD+RELU epilogue,
sizes up to 2048x2048x2049) with the sqrt(K)-scaled tolerance derived in
§1. The fragment-layout risk this doc opened with is fully resolved: not
one of the four experiments above ever produced a *wrong but plausible*
number, which was the specific failure mode this whole effort was
hedged against.

**Performance did not resolve as cleanly.** The tensor-core kernel never
caught the existing CUDA-core kernel, let alone cuBLAS, at the reference
size. Two follow-up experiments aimed at closing that gap failed for two
different reasons, both instructive:

- Deepening the K-tile (amortizing the load/barrier pair over more mma
  calls, the exact lever that helped the CUDA-core kernel go from BK=8 to
  its current shape) made this kernel *slower*, consistently, across three
  isolated-process repeats — not noise. The likely cause is register
  pressure/occupancy from unrolling 4x more loads+converts+mma per k-tile,
  but this wasn't root-caused with a profiler (Nsight Compute wasn't run
  this session); it's reported as an observation, not a diagnosed
  mechanism.
- Doubling the block tile to 64x64 (32 warps/block, the natural next step
  toward matching the CUDA-core kernel's own 128x128 block) hit a real
  bug, not a perf ceiling: the cooperative-load loop assumed
  `TC_BM*TC_BK >= TC_NTHREADS` (true at 32x32, where they're exactly
  equal) so every thread loads at least one element; at 64x64 the block
  covers 1024 threads but the A/B tiles only have 512 elements each,
  making the iteration count `A_ELEMS/TC_NTHREADS` truncate to 0 via
  integer division and leaving shared memory uninitialized. Fixable (loop
  needs to become a predicated `tid < A_ELEMS` guard rather than an
  iteration count for tile shapes where threads outnumber elements) but a
  real code change, not a one-line tune — deferred rather than rushed
  given the exact risk this whole stream has been careful about.

The honest conclusion: **this session's tensor-core kernel is correct but
not fast enough to beat cuBLAS, PyTorch, or even this repo's own
CUDA-core kernel at raw GEMM.** 51.4% of cuBLAS on a first-afternoon
tensor-core kernel against a target that's itself running near-peak
tensor cores is a real result, not a failure, but it doesn't move the
"beat PyTorch" needle on this axis. That's consistent with this doc's own
opening argument in §3: raw GEMM GFLOPS was always the hardest place to
win. See the fused-latency and memory results below for where the honest
opportunity actually was.

Merged to `main` locally (correctness-verified, additive, doesn't touch
`emit_ptx_blocked` or any existing caller) despite not beating the
CUDA-core kernel on speed, on the same basis the CPU reference path is
kept: it's a real, tested, independently useful artifact (the first
tensor-core code in this repo, with a verified fragment layout future
sessions can build on) even though it isn't today's fastest path.

## Results: fusion-latency and memory stream

Measured `relu(matmul(a,b)+c)`, square sizes 128-2048, process-isolated
(`bench --pair <kind> <S>`, `bench_pytorch.py --size <S>` -- the latter
gained a `--size` flag this session specifically to match the C harness's
isolation, after an early reading at S=128 with the *original* 10-warmup
PyTorch protocol ranged 0.014-0.077ms across three otherwise-identical
isolated runs: this GPU's clock-ramp behavior, the same phenomenon
`bench.c`'s own `WARMUP=150` comment documents, was under-warmed by
PyTorch's original `WARMUP=10`. Raising it to 150 brought the reading to a
stable 0.047ms across three repeats -- reported below, not the noisy one).
Every number is `min` of several isolated repeats per the project's
established protocol.

| Size | MiniXLA fused (1 launch) | PyTorch eager unfused (3 launches) | Latency ratio | MiniXLA peak mem | PyTorch peak mem | Mem ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.0389 ms | 0.0471 ms | **MiniXLA 1.21x faster** | 2.0 MiB | 8.4 MiB | MiniXLA 4.22x less |
| 256 | 0.0625 ms | 0.0471 ms | PyTorch 1.33x faster | 2.0 MiB | 9.4 MiB | MiniXLA 4.69x less |
| 512 | 0.1219 ms | 0.0758 ms | PyTorch 1.61x faster | 4.0 MiB | 13.1 MiB | MiniXLA 3.28x less |
| 1024 | 0.4588 ms | 0.3625 ms | PyTorch 1.27x faster | 16.0 MiB | 28.1 MiB | MiniXLA 1.76x less |
| 2048 | 3.0188 ms | 2.7351 ms | PyTorch 1.10x faster | 64.0 MiB | 88.1 MiB | MiniXLA 1.38x less |

**Latency: the fusion hypothesis in §3 was only half right.** MiniXLA's
single fused launch beats PyTorch eager's three separate launches only at
the smallest size tested (128x128x128, a real but modest 21% win) — not
across the whole small-size range as the "launch-overhead-bound" argument
predicted. At every other size, PyTorch eager wins, by a widening-then-
narrowing margin (61% slower for MiniXLA at 512, closing back to 10% at
2048). The likely explanation: MiniXLA's per-op raw throughput
disadvantage (its GEMM kernel is well behind cuBLAS at every size per the
main README table) starts outweighing the saved launches almost
immediately once problem size exceeds "trivially small" — three fast
cuBLAS-backed launches beat one slower fused launch as soon as the
per-launch fixed cost stops dominating. This is a real, if narrow,
structural win at one specific size, not the broad latency advantage
hoped for; reported as found, not rounded up. **`torch.compile` was not
included in this comparison**: this machine has no working Triton install
on Windows (`bench_pytorch.py`'s own long-standing note, re-verified this
session — `torch.compile` still raises `TritonMissing`), so the
three-way comparison the original plan called for isn't possible here;
eager is the only PyTorch baseline actually measurable on this hardware.

**Memory: this is the real, unambiguous, whole-range win.** MiniXLA uses
less peak GPU memory than PyTorch eager at every size tested, by a
narrowing but always-real margin (4.7x at the small end, 1.38x at
2048x2048x2048). This tracks the structural argument in §4 directly:
MiniXLA allocates exactly 4 buffers (`a`, `b`, `c`, `out`) with no
intermediate materialization, while PyTorch eager's three separate ops
each produce a full M×N tensor the caching allocator has to hold; PyTorch's
own allocator overhead (bucket-granularity allocation, holding freed blocks
for reuse) compounds that gap further at small sizes rather than closing
it. Note the two numbers aren't measuring identically-defined things
(MiniXLA's is a raw `cuMemGetInfo` delta around real `cuMemAlloc` calls,
no pooling; PyTorch's is `max_memory_allocated()`, the *caching allocator's*
high-water mark) — but that's the honest, real-world comparison a user
actually experiences on each system, not an apples-to-oranges artifact:
MiniXLA's own numbers show the CUDA driver's allocation granularity
flooring the two smallest sizes at 2 MiB regardless of the ~0.4 MiB the
raw tensors need, so even MiniXLA's number isn't "true" kernel-necessary
bytes either — both numbers include their platform's real allocation
overhead, which is exactly what a user's memory budget has to account for.

**Buffer pooling was not added to `gpu_exec.c`.** The latency benchmark
above (`gpu_time_kernel_ms`) uploads its operands once and times the
kernel launch alone in a loop — it does not exercise repeated
`cuMemAlloc`/`cuMemFree`, so it has no evidence to offer either way on
whether allocator churn matters for a real multi-call workload. Building a
pool without that evidence would be exactly the speculative addition this
project's own minimalism rule warns against. A future session profiling
`gpu_execute` under repeated calls on the same graph shape (not measured
this session) is the right prerequisite before adding one.

## Conclusion: does anything here beat PyTorch?

Being precise about scope: **memory, yes, clearly, across the whole size
range tested. Speed, only at one specific small size (128x128x128, ~21%),
and not at all elsewhere — raw GEMM (tensor-core stream above) didn't
close the gap either.** Both are honest, narrower results than "beats
PyTorch," and both are reported as found rather than as a broader claim
they don't support.

## Sources

- [PTX ISA 9.3, §9.7.15.5.7 "Matrix Fragments for mma.m16n8k8"](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688) (fetched and read directly, not summarized — the primary source for the fragment layout used in this session's kernel)
- [PTX ISA 9.3, `cvt` instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) (`cvt.rna.tf32.f32` syntax and rounding-mode semantics)
- [NVIDIA/cutlass, `include/cutlass/arch/mma_sm80.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm80.h) (independent cross-check: fragment register counts and PTX register-list order for the same instruction)
- [siboehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"](https://siboehm.com/articles/22/CUDA-MMM) (kernel 10, warp-tiling)
- [tinygrad docs, Developer overview](https://docs.tinygrad.org/developer/developer/) (linear-scan memory planning as a model for graph-level buffer reuse)
- `docs/research/gemm-optimization.md`, `docs/research/autoresearch-gemm.md` (this repo's own prior research and the autoresearch mechanism this session continues)
