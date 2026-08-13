#ifndef GPU_EXEC_H
#define GPU_EXEC_H

#include "graph.h"
#include "tensor.h"

typedef struct GpuCtx GpuCtx;

// Create a CUDA context (cuInit + primary context). Returns NULL on failure.
// The context caches compiled modules keyed by PTX string, so repeated
// gpu_execute calls on structurally identical graphs skip JIT compilation.
GpuCtx* gpu_ctx_create(void);
void    gpu_ctx_destroy(GpuCtx* ctx);

// Execute the full optimized DAG on GPU. Uploads OP_INPUT tensors, runs each
// OP_FUSED kernel, keeps intermediates on device, downloads the root result.
// Requires every non-leaf node in the graph to be OP_FUSED.
// Returns a fresh host Tensor (caller frees). Returns NULL on any failure.
Tensor* gpu_execute(Node* root, GpuCtx* ctx);

// Kernel-only timing for a single OP_FUSED node: uploads inputs once, then
// times `warmup` throwaway launches followed by `reps` timed launches
// (returns the min). Deliberately excludes host<->device transfer and
// allocation overhead -- gpu_execute times a full DAG's worth of that too,
// which is the right thing to measure for end-to-end multi-kernel execution
// but the wrong thing to measure when comparing this kernel's own
// throughput against a library call benchmarked the same way (e.g.
// cublasSgemm timed around the call alone, operands already resident on
// device). Requires root->inputs[0]->output and root->inputs[1]->output to
// be set. Returns -1.0 on failure.
double gpu_time_kernel_ms(Node* root, GpuCtx* ctx, int warmup, int reps);

// Same as gpu_execute but for a single OP_FUSED node, run through the TF32
// tensor-core kernel (docs/research/tensorcore-and-fusion.md) instead of
// the default register-blocked one. Additive/experimental: everything else
// in the codebase keeps using gpu_execute/emit_ptx_blocked. Returns NULL on
// failure.
Tensor* gpu_run_tensorcore(const Node* fused, GpuCtx* ctx);

// Kernel-only timing for the tensor-core path, same contract as
// gpu_time_kernel_ms.
double gpu_time_kernel_tc_ms(Node* root, GpuCtx* ctx, int warmup, int reps);

#endif
