# Closing the gap with cuBLAS: research and implementation plan

## Where we actually are, and why

The benchmark report found MiniXLA's GPU kernel at ~6-8% of cuBLAS
(782 GFLOPS vs 9500 GFLOPS at 2048×2048, RTX 4060 Laptop). Two separate
things explain that gap, and they call for different fixes:

**1. cuBLAS almost certainly isn't even doing FP32 math.** Since CUDA 11,
cuBLAS's default math mode on Ampere and later (our Ada/sm_89 GPU included)
opportunistically downcasts `cublasSgemm`'s FP32 inputs to **TF32** and runs
them through the tensor cores, accumulating in FP32
([NVIDIA: Accelerating AI Training with TF32 Tensor Cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)).
TF32 tensor cores on Ada do roughly an order of magnitude more FLOPs/cycle
than the CUDA cores our kernel uses. This is a hardware-path difference, not
just a "better-tuned kernel" difference; no amount of CUDA-core tuning
alone closes it entirely.

**2. Separately, our CUDA-core kernel itself is unoptimized relative to what
CUDA cores can actually do.** This part *is* closeable, and there's a
well-documented, independently reproduced roadmap for exactly how far:
[siboehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"](https://siboehm.com/articles/22/CUDA-MMM).
Starting from a naive one-thread-per-output kernel (1.3% of cuBLAS) and
applying six well-defined, independently-verifiable optimizations, that
worklog reaches **93.7% of cuBLAS using only CUDA cores, no tensor cores at
all**. Our current kernel (16×16/32×32 shared-memory tiling, one output per
thread, predicated boundary loads) sits at roughly that worklog's *Kernel 3*
stage (shared-memory cache-blocking, ~12.8% of cuBLAS in his numbers). The
next several steps below are proven, ordered, and each independently
verifiable against our own CPU reference before being kept.

## The proven progression (siboehm, reproduced numbers on an A6000)

| # | Technique | GFLOPS | % of cuBLAS |
|---|---|---|---|
| 1 | Naive (1 thread = 1 output, no coalescing) | 309 | 1.3% |
| 2 | Global memory coalescing (thread→output mapping so a warp's loads are contiguous) | 1,987 | 8.5% |
| 3 | Shared-memory cache-blocking (**≈ where we are now**) | 2,980 | 12.8% |
| 4 | 1D register blocking (each thread computes 8 outputs, not 1) | 8,475 | 36.5% |
| 5 | 2D register blocking (each thread computes an 8×8 grid) | 15,972 | 68.7% |
| 6 | Vectorized 128-bit loads (`float4`/`ld.v4`), transposed A in SMEM | 18,237 | 78.4% |
| 9 | Autotuned tile/thread-tile dimensions | 19,721 | 84.8% |
| 10 | Warp-level tiling (explicit warp hierarchy above the thread tile) | 21,779 | 93.7% |

The single biggest lever, by far, is **#5 (2D register blocking)**: one
change, 12.8% → 68.7%. The mechanism: today, each thread does one
`K`-length dot product and touches shared memory `2K` times for `1` FMA's
worth of reuse per load. If each thread instead computes an `TM×TN` tile of
outputs from the same `K`-slice, shared-memory traffic amortizes over
`TM×TN` FMAs per pair of loads instead of `1`: arithmetic intensity goes up
roughly `TM×TN`-fold for the same memory traffic. This is exactly the
classical register-blocking argument from
[Goto & van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (2008)](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf),
the paper that the reference BLAS implementations (GotoBLAS, and its
descendants in OpenBLAS/BLIS) are built around: a multi-level blocking
hierarchy (L2/L1 cache blocking, then register blocking at the innermost
loop) where the *register* block is sized so the inner kernel is compute-
bound, not load-bound.

## Plan for MiniXLA (`ptx.c`, the hand-written PTX emitter)

In priority order, each is its own experiment in the autoresearch loop
below, correctness-gated before it's judged on speed:

1. **2D register blocking.** Each thread computes a `TM×TN` tile (start
   with 4×4 or 8×8, autotune later) instead of one output cell. Requires
   restructuring the emitted kernel's inner loop: load a `TM`-slice of the A
   tile and a `TN`-slice of the B tile from shared memory into registers
   once per `k`, then do `TM*TN` FMAs against those registers instead of one
   load-multiply-accumulate per output. Highest expected impact by a wide
   margin (siboehm: +436% relative, 12.8%→68.7%).
2. **Vectorized shared-memory and global loads.** `ld.global.v4.f32` /
   `ld.shared.v4.f32` / `st.shared.v4.f32` (PTX syntax confirmed: `ld.global.v4.f32
   {%f0,%f1,%f2,%f3}, [addr];`, 16-byte aligned, so 4 contiguous `f32`s per
   instruction) instead of one `f32` per instruction. Requires the A tile to
   be stored transposed in shared memory so the read pattern for register
   blocking is itself contiguous (same trick siboehm uses; without it,
   vectorizing the *write* into SMEM doesn't help the *read* pattern that
   register blocking needs).
3. **Autotune the new shape.** We already autotune tile size (8/16/32) via
   `gpu_autotune`; extend the search to `(BM, BN, BK, TM, TN)` combinations
   now that they're real parameters, same infrastructure, bigger grid.
4. **Warp-level tiling**, if the above lands cleanly and there's room left:
   subdivide each block's tile into per-warp tiles so warps (not just
   threads) pipeline independently across the SM's warp schedulers. Highest
   remaining lever per siboehm's numbers (84.8%→93.7%) but the most
   invasive change to the emitter's control flow; attempt last, once 1-3
   are proven correct and measured.
5. **TF32 tensor cores (`mma.sync`), if time remains.** This is the piece
   that actually closes the *hardware-path* gap in §1, not just the kernel-
   quality gap in §2, and it's a materially different, higher-risk
   undertaking: PTX `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`
   operates on warp-distributed fragments with a specific, non-obvious
   per-thread data layout, not a simple per-thread loop. A layout mistake
   produces *plausible-looking wrong numbers*, not a crash: the single
   highest-risk failure mode for this whole effort. Only attempt after 1-4
   are solid and there's a real correctness-verification budget left for
   it; document as explicitly unattempted otherwise rather than ship an
   unverified fragment layout.

## The autoresearch loop

Following the same mechanism used for `text_diffusion`'s autoresearch runs
(`../text_diffusion/program.md`, itself following
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch)): a
dedicated branch, one hypothesis per commit, a fixed measurement protocol,
and a TSV log: kept if it wins, reverted if it doesn't, looped without
stopping to ask. The adaptation for a GEMM kernel instead of a training run
is in `docs/research/autoresearch-gemm.md`: the fixed-time-budget training
run becomes a fixed-size correctness gate (must still match the CPU
reference) before a change is even eligible to be judged on speed, since a
fast-but-wrong kernel is worse than no change at all: that gate has no
analogue in the training-loop version and is the one thing added, not
removed, from the original mechanism.

## Results (autoresearch/aug13, full log in results.tsv)

| Experiment | GFLOPS @ 2048³ | % of cuBLAS | Status |
|---|---|---|---|
| Baseline (corrected kernel-only timing) | 1225.3 | 12.9% | baseline |
| 1. 2D register blocking (BM=BN=64, TM=TN=4) | 4269.0 | 44.9% | kept |
| 2. Vectorized SMEM reads (transposed A) | 4308.5 | 45.4% | kept |
| 3. Larger tile (BM=BN=128, TM=TN=8) | 5030.7 | 53.0% | kept |
| BK 8→16 | 5009.6 | 52.7% | discarded (no change) |

4.1× over baseline, entirely on CUDA cores, correctness-gated at every
step against the CPU reference across boundary-ragged and large shapes.
**Not** PyTorch/cuBLAS parity: see below for why, honestly.

## Why this doesn't reach PyTorch's numbers, and what would

Stopping here rather than continuing to grid-search tile sizes or attempt
warp-tiling/tensor cores is a deliberate risk call, not running out of
ideas:

- **Global-memory load vectorization** (next on the original priority
  list) needs more than a tweak: the current cooperative-load thread-to-
  element mapping has each thread's 4 loads landing on the *same* K-column
  at different, non-adjacent M-rows (a consequence of `NTHREADS` being a
  clean multiple of `BK`), not 4 contiguous floats: vectorizing it means
  redesigning which elements each thread owns, plus correctly handling a
  vectorized load's boundary case (a `v4` load is all-or-nothing; the
  current scalar predicated-zero-pad trick doesn't extend to it directly).
  Real, but a redesign, not a follow-up line.
- **Warp-tiling** (siboehm: 84.8%→93.7% of cuBLAS, the single largest
  remaining CUDA-core lever) restructures the thread hierarchy itself
  (block → warp tile → thread tile) rather than tuning existing
  parameters. Bigger surface area for a subtle indexing bug than anything
  attempted so far.
- **TF32 tensor cores** are almost certainly the actual majority of the
  remaining gap (see this doc's opening section): cuBLAS's default math
  mode on Ampere+ uses them for FP32 GEMM; nothing here does. This is a
  different instruction class (`mma.sync`, warp-distributed fragment
  layouts) with a failure mode worse than a crash: a fragment-layout
  mistake produces *plausible-looking wrong numbers*, not a build error.

All three are real next experiments, not abandoned; they're the reason
this stopped at a solid, fully-verified 53% instead of pushing further
under time pressure into the highest-risk-of-silent-corruption part of the
plan. A session with a larger correctness-verification budget (more time
per experiment to stress-test boundary cases, ideally a second GPU to
cross-check against) is the right context to attempt them.

## Sources

- [siboehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA, "Accelerating AI Training with NVIDIA TF32 Tensor Cores"](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
- [Goto & van de Geijn, "Anatomy of High-Performance Matrix Multiplication", ACM TOMS 2008](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- PTX ISA vectorized load/store syntax (`ld.global.v4.f32`, `ld.shared.v4.f32`): [NVIDIA PTX ISA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Karpathy, autoresearch](https://github.com/karpathy/autoresearch): the loop mechanism this project's harness is adapted from
