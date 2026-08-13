// Benchmark harness: MiniXLA's GPU path (fused, autotuned) against cuBLAS
// and MiniXLA's own CPU path, across a range of sizes. Every timed number
// here comes from actually running the thing being measured on this
// machine's RTX 4060 -- see docs/superpowers/specs for methodology notes.
// Prints CSV to stdout; bench_report.py combines it with bench_pytorch.py's
// output into the published report.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "graph.h"
#include "optimizer.h"
#include "ptx.h"
#include "gpu_exec.h"

// WARMUP is deliberately high, not the 10 an isolated benchmark would need:
// this GPU idles down to ~210MHz between bursts (nvidia-smi-confirmed, see
// the benchmark report's methodology notes) and a short run of sub-ms
// kernels can finish before the clock governor reacts and ramps back up --
// confirmed by reproducing a spurious ~7x slowdown at one size (cuBLAS and
// MiniXLA both, isolated repro gave the expected fast number) that a 10x
// warmup wasn't consistently absorbing across a full 5-size sweep.
#define WARMUP 150
#define REPS   20

static float* rand_buf(int n, unsigned* seed) {
    float* d = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        d[i] = ((float)((*seed >> 16) % 2000) - 1000.0f) / 1000.0f;
    }
    return d;
}

static double median(double* v, int n) {
    for (int i = 1; i < n; i++) {
        double key = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j+1] = v[j]; j--; }
        v[j+1] = key;
    }
    return v[n / 2];
}

// Minimum-of-many is standard practice for GPU kernel benchmarks: it's far
// less sensitive than the median to transient clock-boost/scheduling
// hiccups (this laptop GPU idles down to 210MHz between bursts and needs a
// few launches to ramp back up -- confirmed by nvidia-smi, not a bug) and
// reports the steady-state performance the kernel can actually sustain.
static double minv(double* v, int n) {
    double m = v[0];
    for (int i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}

// ---- MiniXLA GPU: fused relu(matmul(a,b)+bias) ----
// Kernel-only timing (gpu_time_kernel_ms: upload once, time the launch
// alone) -- the fair comparison point against cublasSgemm's own methodology
// below. An earlier version of this function timed gpu_execute in a loop
// instead, which re-uploads its inputs on every call; that measured upload+
// compute+download against cuBLAS's compute-only number, understating
// MiniXLA's actual kernel throughput. See docs/research/gemm-optimization.md.

static double bench_minixla_gpu(int M, int K, int N, GpuCtx* ctx) {
    unsigned seed = 12345 + M * 7 + K * 13 + N * 17;
    float* ad = rand_buf(M * K, &seed);
    float* bd = rand_buf(K * N, &seed);
    float* cd = rand_buf(M * N, &seed);
    Tensor* ta = create_tensor(ad, (int[]){M, K}, 2);
    Tensor* tb = create_tensor(bd, (int[]){K, N}, 2);
    Tensor* tc = create_tensor(cd, (int[]){M, N}, 2);
    free(ad); free(bd); free(cd);

    Node* root = optimize(g_relu(g_add(g_matmul(input_node(ta), input_node(tb)), input_node(tc))));
    double ms = gpu_time_kernel_ms(root, ctx, WARMUP, REPS);

    free_graph(root);
    free_tensor(ta); free_tensor(tb); free_tensor(tc);
    return ms;
}

// ---- MiniXLA GPU: matmul only, no epilogue (fused_node built directly --
// optimize() never fuses a bare matmul with no elementwise consumer, so
// this bypasses the optimizer rather than going through it). Same kernel
// infrastructure as bench_minixla_gpu, isolating what the add+relu
// epilogue costs on top of the matmul the two share, independent of how
// MiniXLA's raw matmul compares to cuBLAS's.

static double bench_minixla_gpu_matmul_only(int M, int K, int N, GpuCtx* ctx) {
    unsigned seed = 24680 + M * 7 + K * 13 + N * 17;
    float* ad = rand_buf(M * K, &seed);
    float* bd = rand_buf(K * N, &seed);
    Tensor* ta = create_tensor(ad, (int[]){M, K}, 2);
    Tensor* tb = create_tensor(bd, (int[]){K, N}, 2);
    free(ad); free(bd);

    Node* root = fused_node(input_node(ta), input_node(tb), NULL, 0);
    double ms = gpu_time_kernel_ms(root, ctx, WARMUP, REPS);

    free_graph(root);
    free_tensor(ta); free_tensor(tb);
    return ms;
}

// ---- MiniXLA GPU: TF32 tensor-core kernel, matmul only (docs/research/
// tensorcore-and-fusion.md). Same isolation/timing contract as
// bench_minixla_gpu_matmul_only, additive kernel path.

static double bench_minixla_gpu_tc_matmul_only(int M, int K, int N, GpuCtx* ctx) {
    unsigned seed = 24680 + M * 7 + K * 13 + N * 17;
    float* ad = rand_buf(M * K, &seed);
    float* bd = rand_buf(K * N, &seed);
    Tensor* ta = create_tensor(ad, (int[]){M, K}, 2);
    Tensor* tb = create_tensor(bd, (int[]){K, N}, 2);
    free(ad); free(bd);

    Node* root = fused_node(input_node(ta), input_node(tb), NULL, 0);
    double ms = gpu_time_kernel_tc_ms(root, ctx, WARMUP, REPS);

    free_graph(root);
    free_tensor(ta); free_tensor(tb);
    return ms;
}

// ---- cuBLAS: raw sgemm (matmul only, no bias/relu epilogue -- cuBLAS'
// plain Sgemm doesn't fuse one; that's exactly the comparison point) ----

static double bench_cublas(int M, int K, int N, cublasHandle_t handle) {
    unsigned seed = 99999 + M * 7 + K * 13 + N * 17;
    float* ad = rand_buf(M * K, &seed);
    float* bd = rand_buf(K * N, &seed);

    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, (size_t)M * K * sizeof(float));
    cuMemAlloc(&d_b, (size_t)K * N * sizeof(float));
    cuMemAlloc(&d_c, (size_t)M * N * sizeof(float));
    cuMemcpyHtoD(d_a, ad, (size_t)M * K * sizeof(float));
    cuMemcpyHtoD(d_b, bd, (size_t)K * N * sizeof(float));
    free(ad); free(bd);

    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major; row-major C[MxN]=A[MxK]*B[KxN] is computed as
    // the column-major product B^T * A^T = (A*B)^T, i.e. swap operands and
    // dimensions -- the standard trick, not a MiniXLA-specific workaround.
    for (int i = 0; i < WARMUP; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                    (float*)d_b, N, (float*)d_a, K, &beta, (float*)d_c, N);

    CUevent t0, t1;
    cuEventCreate(&t0, CU_EVENT_DEFAULT);
    cuEventCreate(&t1, CU_EVENT_DEFAULT);
    double samples[REPS];
    for (int i = 0; i < REPS; i++) {
        cuEventRecord(t0, 0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                    (float*)d_b, N, (float*)d_a, K, &beta, (float*)d_c, N);
        cuEventRecord(t1, 0);
        cuEventSynchronize(t1);
        float ms; cuEventElapsedTime(&ms, t0, t1);
        samples[i] = ms;
    }
    cuEventDestroy(t0); cuEventDestroy(t1);

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    return minv(samples, REPS);
}

// ---- MiniXLA CPU: the tensor.c triple-loop matmul, via graph execute() ----

static double bench_minixla_cpu(int M, int K, int N) {
    unsigned seed = 55555 + M * 7 + K * 13 + N * 17;
    float* ad = rand_buf(M * K, &seed);
    float* bd = rand_buf(K * N, &seed);
    Tensor* ta = create_tensor(ad, (int[]){M, K}, 2);
    Tensor* tb = create_tensor(bd, (int[]){K, N}, 2);
    free(ad); free(bd);

    Node* root = g_matmul(input_node(ta), input_node(tb));

    for (int i = 0; i < WARMUP && i < 1; i++) execute(root);

    // clock() on this platform has ~15ms resolution, so a single execute()
    // of a small matmul finishes inside one tick and reads back as 0 --
    // batch enough iterations into one timed bracket that the bracket
    // itself is comfortably above the clock's resolution, then divide.
    int batch = M <= 256 ? 100 : (M <= 512 ? 20 : (M <= 1024 ? 5 : 1));
    int reps  = (M >= 1024) ? 3 : REPS;
    double samples[REPS];
    for (int i = 0; i < reps; i++) {
        clock_t c0 = clock();
        for (int b = 0; b < batch; b++) {
            if (root->owns_output) free_tensor(root->output);   // drop prior run's result
            root->output = NULL; root->owns_output = 0;          // force recompute
            execute(root);
        }
        clock_t c1 = clock();
        samples[i] = 1000.0 * (double)(c1 - c0) / CLOCKS_PER_SEC / batch;
    }

    double result = median(samples, reps);
    free_graph(root);
    free_tensor(ta); free_tensor(tb);
    return result;
}

// Running every (system, size) pair in one process/CUDA context -- the
// original design -- turned out not to be neutral: a run of the full sweep
// reproducibly showed cuBLAS (and separately, MiniXLA) reading ~7x slower
// at exactly one size (512) than an isolated single-pair run of the exact
// same call sequence, with GPU temperature ruling out thermal throttling
// and a 5x larger warmup count not fixing it either. That points at a
// cross-system interaction (allocator state, driver/cuBLAS internal
// heuristics) specific to sharing one process across systems, not a
// warmup or thermal issue -- and it's exactly the kind of confound
// process isolation is supposed to rule out. `--pair <kind> <size>` runs
// exactly one (system, size) measurement in a fresh process with nothing
// else ever having touched this CUDA context; the driver script
// (docs/research or a shell loop) invokes this repeatedly instead of
// relying on the single-process sweep below, which is kept only as a
// quick-look default for interactive use, not for numbers that get
// published.
static int run_pair(const char* kind, int S) {
    if (strcmp(kind, "cublas") == 0) {
        cuInit(0);
        cublasHandle_t h;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 1; }
        double ms = bench_cublas(S, S, S, h);
        printf("cublas,%d,%d,%d,%.4f\n", S, S, S, ms);
        cublasDestroy(h);
    } else if (strcmp(kind, "minixla_cpu") == 0) {
        double ms = bench_minixla_cpu(S, S, S);
        printf("minixla_cpu,%d,%d,%d,%.4f\n", S, S, S, ms);
    } else if (strcmp(kind, "minixla_gpu") == 0 || strcmp(kind, "minixla_gpu_matmul_only") == 0 ||
               strcmp(kind, "minixla_gpu_tc_matmul_only") == 0) {
        cuInit(0);
        GpuCtx* ctx = gpu_ctx_create();
        if (!ctx) { fprintf(stderr, "gpu_ctx_create failed\n"); return 1; }
        double ms = strcmp(kind, "minixla_gpu") == 0 ? bench_minixla_gpu(S, S, S, ctx)
            : strcmp(kind, "minixla_gpu_matmul_only") == 0 ? bench_minixla_gpu_matmul_only(S, S, S, ctx)
            : bench_minixla_gpu_tc_matmul_only(S, S, S, ctx);
        printf("%s,%d,%d,%d,%.4f\n", kind, S, S, S, ms);
        gpu_ctx_destroy(ctx);
    } else {
        fprintf(stderr, "unknown kind: %s\n", kind);
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && strcmp(argv[1], "--pair") == 0)
        return run_pair(argv[2], atoi(argv[3]));

    int sizes[] = {128, 256, 512, 1024, 2048};
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int cpu_cap = 1024;   // skip CPU matmul above this -- O(n^3) triple loop, minutes otherwise

    cuInit(0);
    GpuCtx* ctx = gpu_ctx_create();
    if (!ctx) { fprintf(stderr, "gpu_ctx_create failed\n"); return 1; }

    cublasHandle_t cublas_h;
    if (cublasCreate(&cublas_h) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n"); return 1;
    }

    printf("kind,M,K,N,ms\n");
    for (int i = 0; i < n_sizes; i++) {
        int S = sizes[i];
        double gpu_ms        = bench_minixla_gpu(S, S, S, ctx);
        double gpu_matmul_ms = bench_minixla_gpu_matmul_only(S, S, S, ctx);
        double cublas_ms     = bench_cublas(S, S, S, cublas_h);
        printf("minixla_gpu,%d,%d,%d,%.4f\n", S, S, S, gpu_ms);
        printf("minixla_gpu_matmul_only,%d,%d,%d,%.4f\n", S, S, S, gpu_matmul_ms);
        printf("cublas,%d,%d,%d,%.4f\n", S, S, S, cublas_ms);
        fflush(stdout);

        if (S <= cpu_cap) {
            double cpu_ms = bench_minixla_cpu(S, S, S);
            printf("minixla_cpu,%d,%d,%d,%.4f\n", S, S, S, cpu_ms);
            fflush(stdout);
        }
    }

    cublasDestroy(cublas_h);
    gpu_ctx_destroy(ctx);
    return 0;
}
