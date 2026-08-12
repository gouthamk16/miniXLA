# MiniXLA

A lightweight ML compiler and runtime exploring graph compilation, optimization,
and PTX code generation on NVIDIA GPUs — a hands-on look at how systems like
XLA, TensorRT, and TVM turn tensor ops into optimized GPU kernels.

See `idea.md` for the full phase roadmap and long-term vision.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Tensor library (matmul, add, relu, softmax, transpose) | Done — `tensor.c`/`tensor.h` |
| 2 | Computational graph (DAG construction, execution) | Done — `graph.c`/`graph.h` |
| 3 | Graph optimizer (constant folding, dead-node elimination, redundant-op removal, operator fusion) | Done — `optimizer.c`/`optimizer.h`, design in `docs/superpowers/specs/2026-06-16-phase-3-graph-optimizer-design.md` |
| 4 | PTX backend (tiled shared-memory matmul + fused epilogue codegen) | Done — `ptx.c`/`ptx.h`, design in `docs/superpowers/specs/2026-06-17-phase-4-ptx-backend-design.md` |
| 5 | GPU runtime (CUDA Driver API execution, multi-kernel graphs, module caching, tile-size autotuning) | Done — `runtime.c`/`runtime.h` (single fused kernel), `gpu_exec.c`/`gpu_exec.h` (full-graph execution, caching, autotuning) |
| 6–8 | Autodiff, advanced fusion/scheduling, multi-GPU | Not started |

Every phase above is verified end-to-end against real hardware, not just
compiled: `tests.exe` includes a `ptxas`-validated codegen test, and
`gpu_test.exe` runs the generated PTX on an actual CUDA device and checks the
result against the CPU reference within `1e-4`–`1e-2` tolerance (looser for
larger matmuls, where fp32 accumulation order legitimately diverges).

## Build

Requires `gcc` for the CPU-only pieces, and `nvcc` + a CUDA-capable GPU for
anything touching the GPU runtime (`nvcc` finds MSVC as its host compiler on
Windows and compiles this project's C99 directly — no separate C++ shim
needed).

```sh
# Demo: builds the graph, optimizes it, runs it on CPU.
gcc -O2 -Wall -o demo main.c graph.c tensor.c optimizer.c -lm
./demo

# CPU test suite (graph, optimizer, and PTX codegen validation via ptxas).
gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c ptx.c -lm
./tests

# GPU end-to-end tests: assembles and runs generated PTX on the GPU, checks
# it against the CPU reference. Requires nvcc and a CUDA device.
nvcc -o gpu_test gpu_test.c graph.c tensor.c optimizer.c ptx.c runtime.c gpu_exec.c -lcuda
./gpu_test
```

## Layout

- `tensor.c` / `tensor.h` — CPU tensor ops (matmul, add, relu, softmax,
  transpose, permute).
- `graph.c` / `graph.h` — the computational graph: node construction,
  execution, and the `OP_FUSED` executor for optimizer-fused regions.
- `optimizer.c` / `optimizer.h` — the four optimization passes and the
  fixpoint pass manager.
- `ptx.c` / `ptx.h` — PTX emitter for fused matmul+epilogue regions (tiled
  shared-memory matmul, arbitrary M/N/K via predicated boundary loads).
- `runtime.c` / `runtime.h` — minimal CUDA Driver API runner for a single
  fused kernel.
- `gpu_exec.c` / `gpu_exec.h` — full-graph GPU execution: runs a chain of
  `OP_FUSED` kernels keeping intermediates on-device, caches compiled
  modules, and autotunes tile size per epilogue shape.
- `main.c` — the CPU demo.
- `tests.c` — CPU-only test suite (`gcc`).
- `gpu_test.c` — GPU end-to-end test suite (`nvcc`, needs a device).
- `docs/superpowers/` — design docs and implementation plans for completed
  phases.
