// Standalone correctness + timing check for emit_ptx_blocked (the 2D
// register-blocked kernel, see docs/research/gemm-optimization.md), kept
// separate from gpu_exec.c's tested launch/cache logic until this new
// kernel is proven correct -- launches it directly via the Driver API with
// its own grid/block dims (1D block of GEMM_NTHREADS, not tile x tile).
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "graph.h"
#include "tensor.h"
#include "ptx.h"

static int cu_check(CUresult r, const char* ctx) {
    if (r == CUDA_SUCCESS) return 1;
    const char* s = NULL;
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuda: %s: %s\n", ctx, s ? s : "unknown");
    return 0;
}

// Launches the blocked kernel for a plain matmul (a[M,K] @ b[K,N] = out[M,N],
// no epilogue) and returns the result as a host Tensor, or NULL on failure.
static Tensor* run_blocked(const Tensor* a, const Tensor* b) {
    int M = a->shape[0], K = a->shape[1], N = b->shape[1];
    Node* fused = fused_node(input_node((Tensor*)a), input_node((Tensor*)b), NULL, 0);
    char* ptx = emit_ptx_blocked(fused);

    CUdevice dev; CUcontext ctx; CUmodule mod = NULL; CUfunction fn = NULL;
    CUdeviceptr d_a = 0, d_b = 0, d_out = 0;
    Tensor* result = NULL;

    if (!cu_check(cuInit(0), "cuInit")) goto cleanup;
    if (!cu_check(cuDeviceGet(&dev, 0), "cuDeviceGet")) goto cleanup;
    if (!cu_check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain")) goto cleanup;
    if (!cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) goto cleanup;
    if (!cu_check(cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL), "cuModuleLoadDataEx")) {
        fprintf(stderr, "---- PTX ----\n%s\n-------------\n", ptx);
        goto cleanup;
    }
    if (!cu_check(cuModuleGetFunction(&fn, mod, "fused"), "cuModuleGetFunction")) goto cleanup;

    if (!cu_check(cuMemAlloc(&d_a, a->size * sizeof(float)), "alloc a")) goto cleanup;
    if (!cu_check(cuMemcpyHtoD(d_a, a->data, a->size * sizeof(float)), "H2D a")) goto cleanup;
    if (!cu_check(cuMemAlloc(&d_b, b->size * sizeof(float)), "alloc b")) goto cleanup;
    if (!cu_check(cuMemcpyHtoD(d_b, b->data, b->size * sizeof(float)), "H2D b")) goto cleanup;
    if (!cu_check(cuMemAlloc(&d_out, (size_t)M * N * sizeof(float)), "alloc out")) goto cleanup;

    {
        unsigned uM = M, uN = N, uK = K;
        void* params[] = {&d_a, &d_b, &d_out, &uM, &uN, &uK};
        unsigned gx = ((unsigned)N + GEMM_BN - 1) / GEMM_BN;
        unsigned gy = ((unsigned)M + GEMM_BM - 1) / GEMM_BM;
        if (!cu_check(cuLaunchKernel(fn, gx, gy, 1, GEMM_NTHREADS, 1, 1, 0, 0, params, NULL), "cuLaunchKernel"))
            goto cleanup;
        if (!cu_check(cuCtxSynchronize(), "cuCtxSynchronize")) goto cleanup;
    }

    result = create_tensor(NULL, (int[]){M, N}, 2);
    if (!result || !cu_check(cuMemcpyDtoH(result->data, d_out, (size_t)M * N * sizeof(float)), "D2H out")) {
        free_tensor(result); result = NULL;
    }

cleanup:
    if (d_a) cuMemFree(d_a);
    if (d_b) cuMemFree(d_b);
    if (d_out) cuMemFree(d_out);
    if (mod) cuModuleUnload(mod);
    free(ptx);
    free_graph(fused);
    return result;
}

static int approx(const Tensor* a, const Tensor* b, float tol) {
    if (a->size != b->size) { fprintf(stderr, "size mismatch\n"); return 0; }
    for (size_t i = 0; i < a->size; i++) {
        if (fabsf(a->data[i] - b->data[i]) > tol) {
            fprintf(stderr, "mismatch at %zu: got %g want %g\n", i, a->data[i], b->data[i]);
            return 0;
        }
    }
    return 1;
}

static void check_shape(int M, int K, int N, float tol) {
    float* ad = malloc((size_t)M * K * sizeof(float));
    float* bd = malloc((size_t)K * N * sizeof(float));
    unsigned seed = 12345 + M * 7 + K * 13 + N * 17;
    for (int i = 0; i < M * K; i++) { seed = seed * 1103515245u + 12345u; ad[i] = ((float)((seed >> 16) % 2000) - 1000.0f) / 1000.0f; }
    for (int i = 0; i < K * N; i++) { seed = seed * 1103515245u + 12345u; bd[i] = ((float)((seed >> 16) % 2000) - 1000.0f) / 1000.0f; }

    Tensor* a = create_tensor(ad, (int[]){M, K}, 2);
    Tensor* b = create_tensor(bd, (int[]){K, N}, 2);
    free(ad); free(bd);

    Tensor* cpu = matmul(a, b);
    Tensor* gpu = run_blocked(a, b);

    assert(cpu && gpu);
    int ok = approx(cpu, gpu, tol);
    printf("check_shape M=%d K=%d N=%d: %s\n", M, K, N, ok ? "PASS" : "FAIL");
    assert(ok);

    free_tensor(a); free_tensor(b); free_tensor(cpu); free_tensor(gpu);
}

// Kernel-only timing (upload once, time the launch alone) -- same
// methodology as gpu_time_kernel_ms / bench_loop.c, so the number is
// directly comparable to the recorded baseline.
static double time_blocked_ms(int S, int warmup, int reps) {
    float* ad = malloc((size_t)S * S * sizeof(float));
    float* bd = malloc((size_t)S * S * sizeof(float));
    unsigned seed = 24680;
    for (int i = 0; i < S * S; i++) { seed = seed * 1103515245u + 12345u; ad[i] = ((float)((seed >> 16) % 2000) - 1000.0f) / 1000.0f; }
    for (int i = 0; i < S * S; i++) { seed = seed * 1103515245u + 12345u; bd[i] = ((float)((seed >> 16) % 2000) - 1000.0f) / 1000.0f; }
    Tensor* a = create_tensor(ad, (int[]){S, S}, 2);
    Tensor* b = create_tensor(bd, (int[]){S, S}, 2);
    free(ad); free(bd);

    Node* fused = fused_node(input_node(a), input_node(b), NULL, 0);
    char* ptx = emit_ptx_blocked(fused);

    CUdevice dev; CUcontext ctx; CUmodule mod = NULL; CUfunction fn = NULL;
    CUdeviceptr d_a = 0, d_b = 0, d_out = 0;
    double result = -1.0;

    if (!cu_check(cuInit(0), "cuInit")) goto cleanup;
    if (!cu_check(cuDeviceGet(&dev, 0), "cuDeviceGet")) goto cleanup;
    if (!cu_check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain")) goto cleanup;
    if (!cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) goto cleanup;
    if (!cu_check(cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL), "cuModuleLoadDataEx")) goto cleanup;
    if (!cu_check(cuModuleGetFunction(&fn, mod, "fused"), "cuModuleGetFunction")) goto cleanup;

    if (!cu_check(cuMemAlloc(&d_a, a->size * sizeof(float)), "alloc a")) goto cleanup;
    if (!cu_check(cuMemcpyHtoD(d_a, a->data, a->size * sizeof(float)), "H2D a")) goto cleanup;
    if (!cu_check(cuMemAlloc(&d_b, b->size * sizeof(float)), "alloc b")) goto cleanup;
    if (!cu_check(cuMemcpyHtoD(d_b, b->data, b->size * sizeof(float)), "H2D b")) goto cleanup;
    if (!cu_check(cuMemAlloc(&d_out, (size_t)S * S * sizeof(float)), "alloc out")) goto cleanup;

    {
        unsigned uM = S, uN = S, uK = S;
        void* params[] = {&d_a, &d_b, &d_out, &uM, &uN, &uK};
        unsigned gx = ((unsigned)S + GEMM_BN - 1) / GEMM_BN;
        unsigned gy = ((unsigned)S + GEMM_BM - 1) / GEMM_BM;

        for (int i = 0; i < warmup; i++) {
            cuLaunchKernel(fn, gx, gy, 1, GEMM_NTHREADS, 1, 1, 0, 0, params, NULL);
            cuCtxSynchronize();
        }

        CUevent t0, t1;
        cuEventCreate(&t0, CU_EVENT_DEFAULT);
        cuEventCreate(&t1, CU_EVENT_DEFAULT);
        double best = 1e30;
        for (int i = 0; i < reps; i++) {
            cuEventRecord(t0, 0);
            cuLaunchKernel(fn, gx, gy, 1, GEMM_NTHREADS, 1, 1, 0, 0, params, NULL);
            cuEventRecord(t1, 0);
            cuEventSynchronize(t1);
            float ms; cuEventElapsedTime(&ms, t0, t1);
            if (ms < best) best = ms;
        }
        cuEventDestroy(t0); cuEventDestroy(t1);
        result = best;
    }

cleanup:
    if (d_a) cuMemFree(d_a);
    if (d_b) cuMemFree(d_b);
    if (d_out) cuMemFree(d_out);
    if (mod) cuModuleUnload(mod);
    free(ptx);
    free_graph(fused);
    free_tensor(a); free_tensor(b);
    return result;
}

int main(void) {
    // Sizes chosen to stress boundary handling: smaller than one block tile,
    // not a multiple of BM/BN/BK (64/64/8), exactly one tile, and several
    // tiles with a ragged remainder.
    check_shape(2, 3, 2, 1e-3f);
    check_shape(3, 2, 3, 1e-3f);
    check_shape(17, 5, 23, 1e-2f);
    check_shape(63, 63, 63, 1e-2f);
    check_shape(64, 64, 64, 1e-2f);
    check_shape(65, 9, 65, 1e-2f);
    check_shape(128, 128, 128, 5e-2f);
    check_shape(200, 77, 130, 5e-2f);
    check_shape(256, 256, 256, 1e-1f);
    check_shape(2048, 2048, 2048, 1.0f);   // fp32 accumulation drift at scale (sum of ~2048 terms)
    printf("all shape checks passed\n");

    double ms = time_blocked_ms(2048, 5, 10);
    double gflops = 2.0 * 2048 * 2048 * 2048 / (ms * 1e6);
    printf("minixla_gpu_blocked,2048,%.4f,%.2f\n", ms, gflops);
    return 0;
}
