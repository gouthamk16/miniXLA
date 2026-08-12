#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "graph.h"
#include "optimizer.h"
#include "ptx.h"
#include "runtime.h"
#include "gpu_exec.h"

static int approx_tensors(const Tensor* a, const Tensor* b, float tol) {
    if (a->size != b->size) return 0;
    for (size_t i = 0; i < a->size; i++)
        if (fabsf(a->data[i] - b->data[i]) > tol) return 0;
    return 1;
}

// ---- Test 1: single fused node (2x3 * 3x2 + relu) via runtime.c ----

static void test_single_fused(void) {
    Tensor* a = create_tensor((float[]){1,2,3,4,5,6}, (int[]){2,3}, 2);
    Tensor* b = create_tensor((float[]){1,2,3,4,5,6}, (int[]){3,2}, 2);
    Tensor* c = create_tensor((float[]){-100,0,0,-100}, (int[]){2,2}, 2);

    Node* root = g_relu(g_add(g_matmul(input_node(a), input_node(b)), input_node(c)));
    root = optimize(root);
    assert(root->op == OP_FUSED);

    Tensor* cpu_out = execute(root);
    char*   ptx     = emit_ptx(root);
    Tensor* gpu_out = gpu_run_fused(root, ptx);
    free(ptx);

    assert(gpu_out);
    assert(approx_tensors(cpu_out, gpu_out, 1e-4f));
    free_tensor(gpu_out);
    free_graph(root);
    free_tensor(a); free_tensor(b); free_tensor(c);
    printf("test_single_fused PASS\n");
}

// ---- Test 1b: broadcast-bias fused graph is rejected on GPU, not silently
// wrong. CPU fusion supports it (see tests.c); the PTX emitter doesn't yet
// (Phase 7 design doc), so both GPU entry points must fail loudly instead
// of reading past the smaller bias buffer.

static void test_broadcast_bias_rejected_on_gpu(void) {
    Tensor* a = create_tensor((float[]){1,2,3,4,5,6}, (int[]){2,3}, 2);
    Tensor* b = create_tensor((float[]){1,2,3,4,5,6}, (int[]){3,2}, 2);
    Tensor* bias = create_tensor((float[]){-30,-60}, (int[]){2}, 1);

    Node* root = g_relu(g_add(g_matmul(input_node(a), input_node(b)), input_node(bias)));
    root = optimize(root);
    assert(root->op == OP_FUSED);
    Tensor* cpu_out = execute(root);
    assert(cpu_out);   // CPU fusion handles the broadcast correctly

    char* ptx = emit_ptx(root);
    Tensor* via_runtime = gpu_run_fused(root, ptx);
    free(ptx);
    assert(via_runtime == NULL);

    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);
    Tensor* via_exec = gpu_execute(root, ctx);
    assert(via_exec == NULL);
    gpu_ctx_destroy(ctx);

    free_graph(root);
    free_tensor(a); free_tensor(b); free_tensor(bias);
    printf("test_broadcast_bias_rejected_on_gpu PASS\n");
}

// ---- Test 2: two-layer graph (multi-kernel) via gpu_exec.c ----
// hidden = relu(x * W1 + b1)
// output = hidden * W2 + b2

static void test_two_layer(void) {
    // x: [4, 8]   W1: [8, 16]   b1: [4, 16]
    // hidden: [4, 16]
    // W2: [16, 4]  b2: [4, 4]
    // output: [4, 4]
    int M = 4, K1 = 8, N1 = 16, K2 = N1, N2 = 4;

    float* xd  = calloc(M  * K1, sizeof(float));
    float* w1d = calloc(K1 * N1, sizeof(float));
    float* b1d = calloc(M  * N1, sizeof(float));
    float* w2d = calloc(K2 * N2, sizeof(float));
    float* b2d = calloc(M  * N2, sizeof(float));
    for (int i = 0; i < M  * K1; i++) xd[i]  = (float)(i % 5) * 0.1f;
    for (int i = 0; i < K1 * N1; i++) w1d[i] = (float)(i % 7) * 0.05f - 0.1f;
    for (int i = 0; i < M  * N1; i++) b1d[i] = (float)(i % 3) * 0.02f;
    for (int i = 0; i < K2 * N2; i++) w2d[i] = (float)(i % 11) * 0.03f - 0.15f;
    for (int i = 0; i < M  * N2; i++) b2d[i] = (float)(i % 4) * 0.01f;

    Tensor* tx  = create_tensor(xd,  (int[]){M,  K1}, 2);
    Tensor* tw1 = create_tensor(w1d, (int[]){K1, N1}, 2);
    Tensor* tb1 = create_tensor(b1d, (int[]){M,  N1}, 2);
    Tensor* tw2 = create_tensor(w2d, (int[]){K2, N2}, 2);
    Tensor* tb2 = create_tensor(b2d, (int[]){M,  N2}, 2);

    Node* nx  = input_node(tx);
    Node* nw1 = input_node(tw1);
    Node* nb1 = input_node(tb1);
    Node* nw2 = input_node(tw2);
    Node* nb2 = input_node(tb2);

    Node* hidden = g_relu(g_add(g_matmul(nx, nw1), nb1));
    Node* output = g_add(g_matmul(hidden, nw2), nb2);
    Node* root   = optimize(output);

    // Root must be OP_FUSED (fused2 fuses matmul2 + add_b2)
    assert(root->op == OP_FUSED);

    // CPU reference
    Tensor* cpu_out = execute(root);
    assert(cpu_out);

    // GPU via gpu_execute
    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);
    Tensor* gpu_out = gpu_execute(root, ctx);
    gpu_ctx_destroy(ctx);

    assert(gpu_out);
    assert(approx_tensors(cpu_out, gpu_out, 1e-3f));  // slightly looser: two kernel launches

    free_tensor(gpu_out);
    free_graph(root);
    free_tensor(tx); free_tensor(tw1); free_tensor(tb1);
    free_tensor(tw2); free_tensor(tb2);
    free(xd); free(w1d); free(b1d); free(w2d); free(b2d);
    printf("test_two_layer PASS\n");
}

// ---- Test 3: autotuner — picks best tile, subsequent execute uses it ----

static void test_autotune(void) {
    const int S = 256;
    float* ad = malloc(S * S * sizeof(float));
    float* bd = malloc(S * S * sizeof(float));
    float* cd = malloc(S * S * sizeof(float));
    for (int i = 0; i < S * S; i++) {
        ad[i] = (float)(i % 7) * 0.01f;
        bd[i] = (float)(i % 11) * 0.01f - 0.05f;
        cd[i] = (float)(i % 5) * 0.02f;
    }
    Tensor* ta = create_tensor(ad, (int[]){S, S}, 2);
    Tensor* tb = create_tensor(bd, (int[]){S, S}, 2);
    Tensor* tc = create_tensor(cd, (int[]){S, S}, 2);

    Node* root = optimize(g_relu(g_add(g_matmul(input_node(ta),
                                                 input_node(tb)),
                                       input_node(tc))));
    assert(root->op == OP_FUSED);
    Tensor* cpu_out = execute(root);

    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);

    printf("autotune 256x256 relu(matmul+add):\n");
    int best = gpu_autotune(root, ctx);
    assert(best == 8 || best == 16 || best == 32);

    // Execute using the tuned tile
    Tensor* gpu_out = gpu_execute(root, ctx);
    assert(gpu_out && approx_tensors(cpu_out, gpu_out, 1e-3f));
    free_tensor(gpu_out);

    gpu_ctx_destroy(ctx);
    free_graph(root);
    free_tensor(ta); free_tensor(tb); free_tensor(tc);
    free(ad); free(bd); free(cd);
    printf("test_autotune PASS (best tile = %d)\n", best);
}

// ---- Test 4: module cache — same graph executed twice, results agree ----
// The second run reuses the compiled CUmodule instead of re-JIT-ing.

static void test_module_cache(void) {
    Tensor* a = create_tensor((float[]){1,2,3,4,5,6}, (int[]){2,3}, 2);
    Tensor* b = create_tensor((float[]){1,2,3,4,5,6}, (int[]){3,2}, 2);
    Tensor* c = create_tensor((float[]){-100,0,0,-100}, (int[]){2,2}, 2);

    Node* root = optimize(g_relu(g_add(g_matmul(input_node(a), input_node(b)), input_node(c))));
    assert(root->op == OP_FUSED);

    Tensor* cpu_out = execute(root);

    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);

    // First call: JIT-compiles and caches the module
    Tensor* r1 = gpu_execute(root, ctx);
    assert(r1 && approx_tensors(cpu_out, r1, 1e-4f));
    free_tensor(r1);

    // Second call: must hit the cache (same PTX string → same CUmodule)
    Tensor* r2 = gpu_execute(root, ctx);
    assert(r2 && approx_tensors(cpu_out, r2, 1e-4f));
    free_tensor(r2);

    gpu_ctx_destroy(ctx);
    free_graph(root);
    free_tensor(a); free_tensor(b); free_tensor(c);
    printf("test_module_cache PASS\n");
}

// ---- Test 4: large matmul benchmark (512x512 * 512x512) ----
// Timed with CUDA events; prints ms. Verifies correctness, not a pass/fail perf gate.

static void bench_large_matmul(void) {
    const int S = 512;
    float* ad = malloc(S * S * sizeof(float));
    float* bd = malloc(S * S * sizeof(float));
    for (int i = 0; i < S * S; i++) {
        ad[i] = (float)(i % 17) * 0.01f - 0.08f;
        bd[i] = (float)(i % 13) * 0.01f - 0.06f;
    }

    Tensor* ta = create_tensor(ad, (int[]){S, S}, 2);
    Tensor* tb = create_tensor(bd, (int[]){S, S}, 2);
    Node*   root = optimize(g_relu(g_matmul(input_node(ta), input_node(tb))));
    assert(root->op == OP_FUSED);

    Tensor* cpu_out = execute(root);

    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);

    // Warm up (also caches the module)
    Tensor* warm = gpu_execute(root, ctx);
    assert(warm);
    assert(approx_tensors(cpu_out, warm, 1e-2f));  // fp32 accumulation diverges at this size
    free_tensor(warm);

    // Timed run via CUDA events
    CUevent ev_start, ev_stop;
    cuEventCreate(&ev_start, CU_EVENT_DEFAULT);
    cuEventCreate(&ev_stop,  CU_EVENT_DEFAULT);

    cuEventRecord(ev_start, 0);
    Tensor* timed = gpu_execute(root, ctx);
    cuEventRecord(ev_stop, 0);
    cuEventSynchronize(ev_stop);

    float ms = 0;
    cuEventElapsedTime(&ms, ev_start, ev_stop);
    printf("bench_large_matmul: %dx%d tiled GPU %.2f ms (cache hit)\n", S, S, ms);

    cuEventDestroy(ev_start);
    cuEventDestroy(ev_stop);
    assert(timed);
    free_tensor(timed);

    gpu_ctx_destroy(ctx);
    free_graph(root);
    free_tensor(ta); free_tensor(tb);
    free(ad); free(bd);
    printf("bench_large_matmul PASS\n");
}

int main(void) {
    test_single_fused();
    test_broadcast_bias_rejected_on_gpu();
    test_two_layer();
    test_autotune();
    test_module_cache();
    bench_large_matmul();
    printf("all gpu tests PASS\n");
    return 0;
}
