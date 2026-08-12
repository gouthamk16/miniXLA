# MiniXLA

A lightweight ML compiler and runtime exploring graph compilation, optimization,
and PTX code generation on NVIDIA GPUs — a hands-on look at how systems like
XLA, TensorRT, and TVM turn tensor ops into optimized GPU kernels.

See `idea.md` for the full phase roadmap and long-term vision, and
**[the benchmark report](docs/bench_report.html)** for honest, measured
performance against cuBLAS and PyTorch on real hardware — including where
MiniXLA is nowhere close (raw kernel throughput) and where the graph
optimizer earns its keep.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Tensor library (matmul, add, mul, relu, softmax, transpose) | Done — `tensor.c`/`tensor.h` |
| 2 | Computational graph (DAG construction, execution) | Done — `graph.c`/`graph.h` |
| 3 | Graph optimizer (constant folding, dead-node elimination, redundant-op removal, operator fusion) | Done — `optimizer.c`/`optimizer.h` |
| 4 | PTX backend (tiled shared-memory matmul + fused epilogue codegen) | Done — `ptx.c`/`ptx.h` |
| 5 | GPU runtime (CUDA Driver API execution, multi-kernel graphs, module caching, tile-size autotuning, last-use buffer freeing) | Done — `runtime.c`/`runtime.h`, `gpu_exec.c`/`gpu_exec.h` |
| 6 | Autodiff (reverse-mode, gradient-checked) | Done — `autodiff.c`/`autodiff.h` |
| 7 | Advanced optimizations (broadcast bias-add, GPU buffer liveness) | Done |
| 8 | Multi-GPU (tensor/pipeline parallelism, collectives) | **Design doc only** — see below |

Design docs for every phase live in `docs/superpowers/specs/`.

Every phase above is verified end-to-end against real hardware, not just
compiled: 18 CPU tests (`gcc`, includes a `ptxas`-validated codegen check and
finite-difference gradient checks for autodiff) and 8 GPU tests (`nvcc`,
actually execute generated PTX on a CUDA device and compare against the CPU
reference to `1e-4`–`1e-2`, looser at larger sizes where fp32 accumulation
order legitimately diverges). Two of those GPU-path tests exist specifically
because they're DAG-shaped (a shared node feeding two independent branches) —
that shape caught two real bugs in already-shipped code that every earlier,
chain-shaped test had missed. See the benchmark report's Correctness section
for the full story.

Phase 8 is a design doc, not code: this machine has one physical GPU and no
NCCL, so real multi-device execution can't be built *and verified* here.
Shipping that untested would be exactly the class of bug the rest of this
project spent its time removing from already-shipped code instead.

## Build

Requires `gcc` for the CPU-only pieces, and `nvcc` + a CUDA-capable GPU for
anything touching the GPU runtime (`nvcc` finds MSVC as its host compiler on
Windows and compiles this project's C99 directly — no separate C++ shim
needed). The benchmark harness additionally needs cuBLAS (ships with the CUDA
toolkit) and, for the PyTorch comparison points, a Python environment with
`torch` installed.

```sh
# Demo: builds the graph, optimizes it, runs it on CPU.
gcc -O2 -Wall -o demo main.c graph.c tensor.c optimizer.c -lm
./demo

# CPU test suite (graph, optimizer, autodiff, and PTX codegen validation via ptxas).
gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c ptx.c autodiff.c -lm
./tests

# GPU end-to-end tests: assembles and runs generated PTX on the GPU, checks
# it against the CPU reference. Requires nvcc and a CUDA device.
nvcc -o gpu_test gpu_test.c graph.c tensor.c optimizer.c ptx.c runtime.c gpu_exec.c -lcuda
./gpu_test

# Benchmarks: MiniXLA (GPU + CPU) vs cuBLAS, and separately vs PyTorch.
nvcc -O2 -o bench bench.c graph.c tensor.c optimizer.c ptx.c gpu_exec.c -lcuda -lcublas
./bench
python bench_pytorch.py
```

## Layout

- `tensor.c` / `tensor.h` — CPU tensor ops (matmul, add, mul, relu, softmax,
  transpose, permute). `tensor_add` supports the standard bias-broadcast
  case (a trailing-suffix shape, e.g. `[N]` against `[M,N]`).
- `graph.c` / `graph.h` — the computational graph: node construction,
  execution, `graph_topo_order` (the traversal both the optimizer and
  autodiff build on), and the `OP_FUSED` executor for optimizer-fused
  regions.
- `optimizer.c` / `optimizer.h` — the four optimization passes and the
  fixpoint pass manager.
- `autodiff.c` / `autodiff.h` — reverse-mode autodiff over an already-
  executed graph (eager gradient tensors, not a second differentiable
  graph — see the Phase 6 design doc for why).
- `ptx.c` / `ptx.h` — PTX emitter for fused matmul+epilogue regions (tiled
  shared-memory matmul, arbitrary M/N/K via predicated boundary loads).
- `runtime.c` / `runtime.h` — minimal CUDA Driver API runner for a single
  fused kernel.
- `gpu_exec.c` / `gpu_exec.h` — full-graph GPU execution: runs a chain of
  `OP_FUSED` kernels keeping intermediates on-device (freeing each one right
  after its last consumer runs), caches compiled modules, and autotunes tile
  size per epilogue shape.
- `main.c` — the CPU demo.
- `tests.c` — CPU-only test suite (`gcc`).
- `gpu_test.c` — GPU end-to-end test suite (`nvcc`, needs a device).
- `bench.c` / `bench_pytorch.py` — the benchmark harness behind
  `docs/bench_report.html`.
- `docs/superpowers/` — design docs for every phase.
- `docs/bench_report.html` — the published benchmark report.
