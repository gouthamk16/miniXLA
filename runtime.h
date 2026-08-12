#ifndef RUNTIME_H
#define RUNTIME_H

#include "graph.h"
#include "tensor.h"

// Assemble `ptx`, run the fused node on the GPU, return the result as a fresh
// host Tensor (caller frees via free_tensor). Returns NULL on any failure.
Tensor* gpu_run_fused(const Node* fused, const char* ptx);

#endif
