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

// ---- Test 1c: diamond-shaped graph (a shared leaf feeding two independent
// fused branches that a third fused node then combines) via gpu_exec.c.
// gpu_execute used to derive its execution order by reversing
// graph_collect's preorder, which is only a valid topological order for a
// tree/chain -- for a real DAG like this one, the shared leaf `x` could be
// scheduled after a node that needs it already uploaded. See the Phase 7
// design doc / graph_topo_order in graph.c.

static void test_diamond_shared_input(void) {
    int M = 2, K = 2, N = 2;
    float* xd  = calloc(M * K, sizeof(float));
    float* w1d = calloc(K * N, sizeof(float));
    float* w2d = calloc(K * N, sizeof(float));
    float* w3d = calloc(N * N, sizeof(float));
    for (int i = 0; i < M * K; i++) xd[i]  = (float)(i % 5) * 0.1f + 0.1f;
    for (int i = 0; i < K * N; i++) w1d[i] = (float)(i % 3) * 0.2f - 0.1f;
    for (int i = 0; i < K * N; i++) w2d[i] = (float)(i % 4) * 0.15f + 0.05f;
    for (int i = 0; i < N * N; i++) w3d[i] = (float)(i % 3) * 0.1f;

    Tensor* tx  = create_tensor(xd,  (int[]){M, K}, 2);
    Tensor* tw1 = create_tensor(w1d, (int[]){K, N}, 2);
    Tensor* tw2 = create_tensor(w2d, (int[]){K, N}, 2);
    Tensor* tw3 = create_tensor(w3d, (int[]){N, N}, 2);

    Node* nx  = input_node(tx);
    Node* nw1 = input_node(tw1);
    Node* nw2 = input_node(tw2);
    Node* nw3 = input_node(tw3);

    // h1 and h2 both read `nx` directly -- the diamond.
    Node* h1  = g_relu(g_matmul(nx, nw1));
    Node* h2  = g_relu(g_matmul(nx, nw2));
    Node* root = optimize(g_add(g_matmul(h1, nw3), h2));
    assert(root->op == OP_FUSED);

    Tensor* cpu_out = execute(root);
    assert(cpu_out);

    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);
    Tensor* gpu_out = gpu_execute(root, ctx);
    gpu_ctx_destroy(ctx);

    assert(gpu_out);
    assert(approx_tensors(cpu_out, gpu_out, 1e-3f));

    free_tensor(gpu_out);
    free_graph(root);
    free_tensor(tx); free_tensor(tw1); free_tensor(tw2); free_tensor(tw3);
    free(xd); free(w1d); free(w2d); free(w3d);
    printf("test_diamond_shared_input PASS\n");
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

// ---- Test 2b: deep chain (8 fused layers) — stresses last-use buffer
// freeing over more hops than the 2-layer/diamond tests (more chances for
// an off-by-one in last_use to corrupt a result), and confirms device
// memory returns to baseline once gpu_execute returns. Note this does NOT
// prove peak usage during the run is lower than the old always-free-at-
// the-end behavior -- cuMemGetInfo here only samples before/after the
// whole call, not mid-run. That peak-memory difference is what the
// benchmark suite measures with sampling across the run instead.

#define DEEP_CHAIN_LAYERS 8

static void test_deep_chain(void) {
    const int LAYERS = DEEP_CHAIN_LAYERS, S = 256;
    Tensor* tensors[DEEP_CHAIN_LAYERS];
    Node*   weights[DEEP_CHAIN_LAYERS];

    float* xd = malloc((size_t)S * S * sizeof(float));
    for (int i = 0; i < S * S; i++) xd[i] = (float)(i % 13) * 0.01f - 0.06f;
    Tensor* tx = create_tensor(xd, (int[]){S, S}, 2);
    free(xd);
    Node* nx = input_node(tx);

    Node* h = nx;
    for (int l = 0; l < LAYERS; l++) {
        float* wd = malloc((size_t)S * S * sizeof(float));
        for (int i = 0; i < S * S; i++) wd[i] = (float)((i + l) % 11) * 0.01f - 0.05f;
        tensors[l] = create_tensor(wd, (int[]){S, S}, 2);
        free(wd);
        weights[l] = input_node(tensors[l]);
        h = g_relu(g_matmul(h, weights[l]));
    }

    Node* root = optimize(h);
    assert(root->op == OP_FUSED);
    Tensor* cpu_out = execute(root);
    assert(cpu_out);

    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);

    size_t free_before, free_after, total;
    cuMemGetInfo(&free_before, &total);
    Tensor* gpu_out = gpu_execute(root, ctx);
    cuMemGetInfo(&free_after, &total);

    assert(gpu_out);
    assert(approx_tensors(cpu_out, gpu_out, 5e-2f));  // fp32 drift over 8 chained matmuls

    // Baseline should be restored -- everything gpu_execute allocated for
    // this call is freed by the time it returns, whether freed eagerly
    // (last-use) or all at the end.
    assert(free_after >= free_before - (16 * 1024 * 1024));  // small CUDA-internal slack

    free_tensor(gpu_out);
    gpu_ctx_destroy(ctx);
    free_graph(root);
    free_tensor(tx);
    for (int l = 0; l < LAYERS; l++) free_tensor(tensors[l]);
    printf("test_deep_chain PASS\n");
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

// ---- Test: TF32 tensor-core kernel correctness (emit_ptx_tensorcore,
// docs/research/tensorcore-and-fusion.md) ----
//
// Tolerance is wider than the CUDA-core kernel's and explicitly scales with
// sqrt(K): TF32 keeps ~11 mantissa bits (vs FP32's 23), so each of the K
// per-element products carries relative error of order 2^-11: for random
// signs those errors accumulate like a random walk, so absolute error in
// the K-reduction grows ~sqrt(K), not ~K. atol below is calibrated for this
// test's operand range (|value| <= 10, so per-product magnitude <= 100);
// a different operand range would need a different constant, not a
// different sqrt(K) shape. rtol covers large-magnitude outputs where the
// absolute term alone would be too tight.
static int approx_tensorcore(const Tensor* cpu, const Tensor* gpu, int K) {
    if (cpu->size != gpu->size) return 0;
    double atol = 0.05 * sqrt((double)K), rtol = 0.02;
    for (size_t i = 0; i < cpu->size; i++) {
        double d = fabs(cpu->data[i] - gpu->data[i]);
        if (d > atol + rtol * fabs(cpu->data[i])) return 0;
    }
    return 1;
}

static Tensor* rand_tensor_tc(int r, int c) {
    float* d = malloc(sizeof(float) * r * c);
    for (int i = 0; i < r * c; i++) d[i] = ((float)(rand() % 2000) - 1000) / 100.0f;
    Tensor* t = create_tensor(d, (int[]){r, c}, 2);
    free(d);
    return t;
}

static void tc_case(int M, int K, int N, const char* label) {
    Tensor* a = rand_tensor_tc(M, K);
    Tensor* b = rand_tensor_tc(K, N);
    Node* root = fused_node(input_node(a), input_node(b), NULL, 0);

    Tensor* cpu_out = matmul(a, b);
    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);
    Tensor* gpu_out = gpu_run_tensorcore(root, ctx);
    gpu_ctx_destroy(ctx);

    assert(gpu_out);
    if (!approx_tensorcore(cpu_out, gpu_out, K)) {
        fprintf(stderr, "test_tensorcore_correctness: %s (%dx%dx%d) FAILED\n", label, M, K, N);
        assert(0);
    }

    free_tensor(cpu_out); free_tensor(gpu_out);
    free_graph(root); free_tensor(a); free_tensor(b);
}

// Boundary-ragged shapes (not just clean multiples of the 16x8x8 tile), plus
// one large enough to exercise many K-loop iterations -- same philosophy as
// the diamond-DAG tests above: the bugs that matter show up off the clean
// cases, and a fragment-layout mistake here would be a *plausible-looking*
// wrong number, not a crash, so ragged coverage matters more than usual.
static void test_tensorcore_correctness(void) {
    srand(42);
    tc_case(16, 8, 8, "exact tile");
    tc_case(17, 8, 8, "M ragged");
    tc_case(16, 8, 9, "N ragged");
    tc_case(16, 5, 8, "K ragged");
    tc_case(100, 100, 100, "ragged everywhere");
    tc_case(1024, 1024, 1024, "1024^3");
    tc_case(2048, 2048, 2049, "2048x2048x2049 ragged N");
    printf("test_tensorcore_correctness PASS\n");
}

// Same shape but with an ADD+RELU epilogue, to cover the tensor-core
// emitter's epilogue path (untested by the matmul-only cases above).
static void test_tensorcore_epilogue(void) {
    srand(7);
    int M = 100, K = 100, N = 100;
    Tensor* a = rand_tensor_tc(M, K);
    Tensor* b = rand_tensor_tc(K, N);
    Tensor* c = rand_tensor_tc(M, N);
    EpStep ep[2] = {{OP_ADD, input_node(c)}, {OP_RELU, NULL}};
    Node* root = fused_node(input_node(a), input_node(b), ep, 2);

    Tensor* mm = matmul(a, b);
    Tensor* added = tensor_add(mm, c);
    Tensor* cpu_out = relu(added);
    GpuCtx* ctx = gpu_ctx_create();
    assert(ctx);
    Tensor* gpu_out = gpu_run_tensorcore(root, ctx);
    gpu_ctx_destroy(ctx);

    assert(gpu_out);
    assert(approx_tensorcore(cpu_out, gpu_out, K));

    free_tensor(mm); free_tensor(added); free_tensor(cpu_out); free_tensor(gpu_out);
    free_graph(root); free_tensor(a); free_tensor(b); free_tensor(c);
    printf("test_tensorcore_epilogue PASS\n");
}

int main(void) {
    test_single_fused();
    test_broadcast_bias_rejected_on_gpu();
    test_diamond_shared_input();
    test_two_layer();
    test_deep_chain();
    test_module_cache();
    bench_large_matmul();
    test_tensorcore_correctness();
    test_tensorcore_epilogue();
    printf("all gpu tests PASS\n");
    return 0;
}
