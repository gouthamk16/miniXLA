#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "runtime.h"
#include "graph.h"
#include "ptx.h"

static int cu_check(CUresult r, const char* ctx) {
    if (r == CUDA_SUCCESS) return 1;
    const char* s = NULL;
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuda: %s: %s\n", ctx, s ? s : "unknown");
    return 0;
}

Tensor* gpu_run_fused(const Node* fused, const char* ptx) {
    Tensor* a = fused->inputs[0]->output;
    Tensor* b = fused->inputs[1]->output;
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "gpu_run_fused: only 2D tensors supported\n");
        return NULL;
    }

    int M = a->shape[0], K = a->shape[1], N = b->shape[1];
    unsigned uM = (unsigned)M, uN = (unsigned)N, uK = (unsigned)K;
    size_t out_bytes = (size_t)M * N * sizeof(float);

    int n_add = 0;
    for (int i = 0; i < fused->n_epilogue; i++)
        if (fused->epilogue[i].op == OP_ADD) n_add++;

    // The PTX epilogue always indexes an OP_ADD operand at the same flat
    // cell as the output; broadcast (e.g. bias-vector) operands aren't
    // supported by the emitter yet. Fail loudly rather than silently
    // reading past a smaller buffer -- see the Phase 7 design doc.
    for (int i = 0; i < fused->n_epilogue; i++) {
        if (fused->epilogue[i].op != OP_ADD) continue;
        Tensor* op = fused->epilogue[i].operand->output;
        if ((int)op->size != M * N) {
            fprintf(stderr, "gpu_run_fused: broadcast epilogue operands are not supported on GPU\n");
            return NULL;
        }
    }

    CUdevice    dev     = 0;
    CUcontext   ctx     = NULL;
    CUmodule    mod     = NULL;
    CUfunction  fn      = NULL;
    CUdeviceptr d_a     = 0, d_b = 0, d_out = 0;
    CUdeviceptr* d_ops  = NULL;
    Tensor*     result  = NULL;
    int ctx_retained    = 0;

    if (n_add > 0) {
        d_ops = (CUdeviceptr*)calloc(n_add, sizeof(CUdeviceptr));
        if (!d_ops) goto cleanup;
    }

    if (!cu_check(cuInit(0),                              "cuInit"))                     goto cleanup;
    if (!cu_check(cuDeviceGet(&dev, 0),                   "cuDeviceGet"))                goto cleanup;
    if (!cu_check(cuDevicePrimaryCtxRetain(&ctx, dev),    "cuDevicePrimaryCtxRetain"))   goto cleanup;
    ctx_retained = 1;
    if (!cu_check(cuCtxSetCurrent(ctx),                   "cuCtxSetCurrent"))            goto cleanup;

    if (!cu_check(cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL), "cuModuleLoadDataEx"))  goto cleanup;
    if (!cu_check(cuModuleGetFunction(&fn, mod, "fused"),        "cuModuleGetFunction")) goto cleanup;

    // Upload A and B
    if (!cu_check(cuMemAlloc(&d_a, a->size * sizeof(float)),             "cuMemAlloc a"))    goto cleanup;
    if (!cu_check(cuMemcpyHtoD(d_a, a->data, a->size * sizeof(float)),  "cuMemcpyHtoD a"))  goto cleanup;
    if (!cu_check(cuMemAlloc(&d_b, b->size * sizeof(float)),             "cuMemAlloc b"))    goto cleanup;
    if (!cu_check(cuMemcpyHtoD(d_b, b->data, b->size * sizeof(float)),  "cuMemcpyHtoD b"))  goto cleanup;

    // Upload OP_ADD operands
    int add_idx = 0;
    for (int i = 0; i < fused->n_epilogue; i++) {
        if (fused->epilogue[i].op != OP_ADD) continue;
        Tensor* op = fused->epilogue[i].operand->output;
        if (!cu_check(cuMemAlloc(&d_ops[add_idx], op->size * sizeof(float)),            "cuMemAlloc op"))   goto cleanup;
        if (!cu_check(cuMemcpyHtoD(d_ops[add_idx], op->data, op->size * sizeof(float)),"cuMemcpyHtoD op")) goto cleanup;
        add_idx++;
    }

    if (!cu_check(cuMemAlloc(&d_out, out_bytes), "cuMemAlloc out")) goto cleanup;

    // Build kernel params: [a, b, op0…, out, M, N, K]
    void** params = (void**)malloc((n_add + 6) * sizeof(void*));
    if (!params) goto cleanup;
    int pi = 0;
    params[pi++] = &d_a;
    params[pi++] = &d_b;
    for (int i = 0; i < n_add; i++) params[pi++] = &d_ops[i];
    params[pi++] = &d_out;
    params[pi++] = &uM;
    params[pi++] = &uN;
    params[pi++] = &uK;

    // Launch config matches the register-blocked kernel's own tiling
    // (GEMM_BM/BN, 1D block of GEMM_NTHREADS) -- see ptx.h.
    unsigned grid_x = ((unsigned)N + GEMM_BN - 1) / GEMM_BN;
    unsigned grid_y = ((unsigned)M + GEMM_BM - 1) / GEMM_BM;
    CUresult launch_r = cuLaunchKernel(fn, grid_x, grid_y, 1,
                                       GEMM_NTHREADS, 1, 1,
                                       0, 0, params, NULL);
    free(params);
    if (!cu_check(launch_r, "cuLaunchKernel")) goto cleanup;

    if (!cu_check(cuCtxSynchronize(), "cuCtxSynchronize")) goto cleanup;

    result = create_tensor(NULL, (int[]){M, N}, 2);
    if (!result) goto cleanup;
    if (!cu_check(cuMemcpyDtoH(result->data, d_out, out_bytes), "cuMemcpyDtoH")) {
        free_tensor(result);
        result = NULL;
    }

cleanup:
    if (d_a)   cuMemFree(d_a);
    if (d_b)   cuMemFree(d_b);
    if (d_ops) {
        for (int i = 0; i < n_add; i++) if (d_ops[i]) cuMemFree(d_ops[i]);
        free(d_ops);
    }
    if (d_out) cuMemFree(d_out);
    if (mod)   cuModuleUnload(mod);
    if (ctx_retained) cuDevicePrimaryCtxRelease(dev);
    return result;
}
