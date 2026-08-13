# autoresearch: GEMM kernel optimization

Adapted from `../../../text_diffusion/program.md` (itself following
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) for
optimizing a GPU kernel instead of training a model. The mechanism is the
same — a dedicated branch, one hypothesis per commit, a fixed measurement
protocol, a TSV log, keep-if-better/revert-if-not, loop without stopping to
ask — with one addition the training-loop version doesn't need: **a
correctness gate before a change is even eligible to be judged on speed.**
A fast GEMM kernel that computes the wrong answer is worse than no change at
all, and unlike a val-loss regression, a wrong kernel can look completely
fine until you check the numbers.

## Setup

- Branch: `autoresearch/<tag>` off `main`.
- The file you modify: `ptx.c` (the PTX emitter) and, when a change needs
  it, `gpu_exec.c`/`runtime.c` (launch config, autotuning search space).
- Everything else (`graph.c`, `optimizer.c`, `tensor.c`, the CPU reference
  path) is the ground truth and does not change.

## The correctness gate

Before any change is judged on speed, it must pass, unmodified:

```sh
gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c ptx.c autodiff.c -lm && ./tests
nvcc -o gpu_test gpu_test.c graph.c tensor.c optimizer.c ptx.c runtime.c gpu_exec.c -lcuda && ./gpu_test
```

`gpu_test` compares every GPU result against the CPU reference (`execute()`)
to `1e-4`–`1e-2` depending on size — that comparison is the ground truth, the
same role `evaluate_bpb` plays in the training-loop version. If either
suite fails or a result silently drifts outside tolerance, the change is a
`crash` regardless of how fast it benchmarked, full stop.

## Measurement protocol

One fixed reference size for fast iteration: **2048×2048×2048**, the size
where cuBLAS is closest to its own peak and the CUDA-core gap is cleanest to
see. `min` of 10 timed launches after 5 warmup launches (lighter than the
published report's 20-after-10 — this runs many times per session, the
report's own numbers get a full re-measurement at the end, not every
intermediate step). Report GFLOPS and % of cuBLAS at that size (cuBLAS's own
number doesn't change between experiments — measure it once, reuse it).

## Logging results

`docs/research/results.tsv` (tab-separated; not committed — same as the
training-loop version, it's a local log, not source). Columns:

```
commit	gflops_2048	pct_cublas	status	description
```

- `commit`: short hash
- `gflops_2048`: measured GFLOPS at 2048×2048×2048, or `0` for a crash
- `pct_cublas`: `gflops_2048 / cublas_gflops_2048 * 100`, one decimal
- `status`: `keep`, `discard`, or `crash`
- `description`: one line, what the experiment tried

## The loop

LOOP:

1. Check git state (branch/commit).
2. Implement one idea from `gemm-optimization.md`'s priority list (or a
   follow-up it suggests) in `ptx.c` / `gpu_exec.c` / `runtime.c`.
3. `git commit`.
4. Run the correctness gate. Fails → fix if it's a quick bug, else `crash`,
   log it, `git reset --hard` to the last kept commit, next idea.
5. Passes → benchmark at 2048×2048×2048, log the result.
6. Better than the last kept commit → keep (branch advances).
   Equal or worse → `discard`, `git reset --hard` to the last kept commit.
7. Next idea. Do not stop to ask whether to continue.

When every item in the priority list is exhausted: re-read
`gemm-optimization.md` for the next tier (autotuning the new parameters,
warptiling, tensor cores) rather than stopping. Merge to `main` and update
the published benchmark report once the loop has produced a meaningfully
better number and there's nothing left in the plan worth the remaining risk
(tensor cores especially — see that section's risk note before attempting
it under time pressure).
