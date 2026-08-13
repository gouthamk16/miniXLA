# Phase 4: PTX Backend Design

## Goal

Generate PTX assembly for the `OP_FUSED` regions produced by the Phase 3
optimizer, then prove the generated kernel correct end-to-end: assemble it,
run it on the GPU via the CUDA Driver API, and compare the result to the CPU
`execute()` path.

Scope: one PTX kernel per fused region, 2-D matmul (batch = 1). CPU-side
emitter plus a minimal Driver-API runtime. This deliberately pulls a slice of
the Phase 5 runtime forward so the kernel is numerically verified now.

## Verified toolchain

Confirmed on this machine (CUDA 12.6, RTX 4060, compute capability 8.9):

- The **PTX emitter** has no CUDA dependency: it is pure C and builds with
  gcc alongside the existing sources.
- The **GPU runtime** uses the CUDA Driver API (`cuda.h`) and is built with
  **nvcc**, which locates MSVC, compiles the project's C99 (compound literals
  included), links the driver, and runs on the GPU.
- PTX is JIT-assembled to `sm_89` by the driver at load time
  (`cuModuleLoadDataEx`). A standalone `ptxas -arch=sm_89` run is the
  pure-codegen validation gate.

No register allocator is needed: PTX uses virtual registers and `ptxas`
performs hardware allocation. The emitter declares an over-sized register pool
and emits fresh virtual register names.

## Target and threading model

One PTX kernel per `OP_FUSED` node. **One GPU thread per output cell**
`C[i, j]`:

1. Compute global thread id `t = ctaid.x * ntid.x + tid.x`.
2. Guard `t >= M*N` → return (handles the grid tail).
3. `i = t / N`, `j = t % N`.
4. `k`-loop: `acc += A[i*K+k] * B[k*N+j]` via `fma.rn.f32`.
5. Epilogue, generated from the node's `EpStep` chain in order:
   - `OP_ADD`  → `acc += operand[t]`  (operand index `= i*N+j = t`)
   - `OP_RELU` → `acc = max(acc, 0)`
6. Store `out[t] = acc`.

Launch configuration: block = 256 threads, grid = `ceil(M*N / 256)`.

Out of scope: batched matmul (`batch > 1`), non-2-D tensors. The runtime
errors out if the fused node's operands are not 2-D.

## Components

### `ptx.c` / `ptx.h`
```c
// Emit PTX for a single OP_FUSED node. Caller frees the returned string.
char* emit_ptx(const Node* fused);
```
The kernel is a fixed skeleton (`.version` / `.target sm_89` /
`.address_size 64` header, entry signature, thread-id guard, `i`/`j`
computation, `k`-loop) plus a **generated tail** driven by the epilogue:

- One `.param .u64` pointer parameter per `OP_ADD` operand, in chain order,
  named `p_op0`, `p_op1`, …
- One emitted block per epilogue step: `add.f32` (loading from the matching
  operand pointer) for `OP_ADD`, `max.f32 %f, %f, 0f00000000` for `OP_RELU`.

Registers use a counter; the `.reg` directives over-declare the pools
(`.f32`, `.b32`, `.b64`, `.pred`) so emission never runs out.

### `runtime.c` / `runtime.h`
```c
// Assemble `ptx`, run the fused node on the GPU, return the result as a fresh
// host Tensor (caller frees via free_tensor). Returns NULL on any failure.
Tensor* gpu_run_fused(const Node* fused, const char* ptx);
```
Driver-API sequence, every `CUresult` checked (on failure: log to stderr,
clean up what was allocated, return NULL):

1. `cuInit(0)`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`.
2. `cuModuleLoadDataEx(ptx)` → `cuModuleGetFunction(&fn, mod, "fused")`.
3. For `a = inputs[0]->output`, `b = inputs[1]->output`, and each ADD operand
   tensor: `cuMemAlloc` + `cuMemcpyHtoD`.
4. `cuMemAlloc` the output (`M*N` floats).
5. Build the kernel parameter array: device pointers (a, b, op0…, out) then
   `M`, `N`, `K` as `unsigned`.
6. `cuLaunchKernel(fn, gridX,1,1, 256,1,1, 0, 0, params, 0)`.
7. `cuCtxSynchronize`.
8. Allocate a host `Tensor` of shape `[M, N]`, `cuMemcpyDtoH` into it.
9. Free all device buffers and the module; release the primary context.

`M`, `N`, `K` are read from the operand shapes: `a` is `[M, K]`, `b` is
`[K, N]`.

### `gpu_test.c`
nvcc-built end-to-end test. Builds `relu(matmul(a, b) + c)`, runs `optimize`,
asserts the root is `OP_FUSED`, calls `emit_ptx` + `gpu_run_fused`, and
asserts the GPU result matches CPU `execute()` of the same graph within
`1e-4`.

## Data flow

```
graph (Phase 3 optimize) ──▶ OP_FUSED node
        │                        │
        │ emit_ptx               │ inputs[] tensors
        ▼                        ▼
   PTX string ───────────▶ gpu_run_fused ──▶ GPU ──▶ result Tensor
                                                         │
                          CPU execute(same node) ───────┤ compare (1e-4)
```

## Verification

1. **Codegen (gcc, no GPU)**: in `tests.c`: `emit_ptx` on the optimized
   fused graph, write to a temp `.ptx`, run `ptxas -arch=sm_89 <file> -o <nul>`
   via `system`, assert exit 0. Also assert the text contains `.entry fused`,
   an `fma`, and a `max.f32`.
2. **End-to-end (nvcc, GPU)**: `gpu_test.c` as described: GPU result equals
   CPU result within `1e-4`.

## Build commands

- CPU tests + codegen: `gcc -O2 -Wall -o tests tests.c graph.c tensor.c optimizer.c ptx.c -lm`
- GPU end-to-end: `nvcc -o gpu_test gpu_test.c graph.c tensor.c optimizer.c ptx.c runtime.c -lcuda`

## File layout

- `ptx.c` / `ptx.h`: PTX emitter (pure C).
- `runtime.c` / `runtime.h`: Driver-API runner (CUDA).
- `gpu_test.c`: nvcc end-to-end test.
- `tests.c`: gains the codegen / `ptxas` validation test.

## Out of scope (later phases)

- Batched and >2-D matmul kernels.
- Tiled / shared-memory matmul optimization (correctness first).
- Caching compiled modules, multi-kernel graphs, kernel scheduling (Phase 5+).
- Generating PTX for non-fused nodes (the optimizer always fuses the demo
  path; standalone-op PTX can come later if needed).
