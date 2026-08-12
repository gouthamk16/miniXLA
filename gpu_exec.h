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
// OP_FUSED kernel (using the tuned tile size if gpu_autotune was called),
// keeps intermediates on device, downloads the root result.
// Requires every non-leaf node in the graph to be OP_FUSED.
// Returns a fresh host Tensor (caller frees). Returns NULL on any failure.
Tensor* gpu_execute(Node* root, GpuCtx* ctx);

// Benchmark tile sizes {8, 16, 32} for the given OP_FUSED node using CUDA
// events (5 warmup + 5 timed runs each). Stores the optimal tile in ctx and
// returns it. Subsequent gpu_execute calls use that tile for matching nodes.
// Requires root->inputs[0]->output and root->inputs[1]->output to be set.
int gpu_autotune(Node* root, GpuCtx* ctx);

#endif
