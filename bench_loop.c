// Fast single-size benchmark for the GEMM autoresearch loop (see
// docs/research/autoresearch-gemm.md). Not the published report's harness
// (that's bench.c, full sweep, min-of-20-after-10, best of 4 runs) -- this
// is the lighter, faster-iterating version used between experiments.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "graph.h"
#include "optimizer.h"
#include "ptx.h"
#include "gpu_exec.h"

#define S 2048
#define WARMUP 5
#define REPS 10

static float* rand_buf(int n, unsigned* seed) {
    float* d = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        d[i] = ((float)((*seed >> 16) % 2000) - 1000.0f) / 1000.0f;
    }
    return d;
}

int main(void) {
    unsigned seed = 24680;
    float* ad = rand_buf(S * S, &seed);
    float* bd = rand_buf(S * S, &seed);
    Tensor* ta = create_tensor(ad, (int[]){S, S}, 2);
    Tensor* tb = create_tensor(bd, (int[]){S, S}, 2);
    free(ad); free(bd);

    cuInit(0);
    GpuCtx* ctx = gpu_ctx_create();
    if (!ctx) { fprintf(stderr, "gpu_ctx_create failed\n"); return 1; }

    Node* root = fused_node(input_node(ta), input_node(tb), NULL, 0);
    gpu_autotune(root, ctx);

    double ms = gpu_time_kernel_ms(root, ctx, WARMUP, REPS);
    if (ms < 0) { fprintf(stderr, "gpu_time_kernel_ms failed\n"); return 1; }
    double gflops = 2.0 * S * S * S / (ms * 1e6);
    printf("minixla_gpu_matmul_only,%d,%.4f,%.2f\n", S, ms, gflops);

    free_graph(root);
    free_tensor(ta); free_tensor(tb);
    gpu_ctx_destroy(ctx);
    return 0;
}
