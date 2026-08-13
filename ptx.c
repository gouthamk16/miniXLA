// PTX emitter: 2D register-blocked shared-memory matmul + fused elementwise
// epilogue (docs/research/gemm-optimization.md). Handles arbitrary M, N, K
// via predicated boundary loads + output guard. Superseded the original
// one-thread-per-output tiled kernel (~12.9% of cuBLAS) once this one was
// correctness-verified and benchmarked at ~45% -- see that doc for the
// autoresearch log instead of keeping the old kernel around as dead code.
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "ptx.h"

typedef struct { char* s; int len; int cap; } Buf;

static void bprintf(Buf* b, const char* fmt, ...) {
    for (;;) {
        char* dst   = (b->cap > b->len) ? b->s + b->len : NULL;
        int   avail = (b->cap > b->len) ? b->cap - b->len : 0;
        va_list ap;
        va_start(ap, fmt);
        int need = vsnprintf(dst, avail, fmt, ap);
        va_end(ap);
        if (need < avail) { b->len += need; return; }
        int nc = b->cap ? b->cap * 2 : 8192;
        while (nc - b->len <= need) nc *= 2;
        b->s = realloc(b->s, nc);
        b->cap = nc;
    }
}

// ---- 2D register-blocked kernel (docs/research/gemm-optimization.md) ----
// Each thread computes a GEMM_TM x GEMM_TN grid of output cells instead of
// one, so shared-memory traffic amortizes over TM*TN FMAs per pair of
// loads instead of 1. Fixed BM/BN/BK/TM/TN for now (see ptx.h) -- not yet
// autotuned. blockDim is 1D (GEMM_NTHREADS threads); gridDim is
// ceil(N/BN) x ceil(M/BM), same convention as the tiled kernel.
//
// Register plan (kept fixed and reused rather than counter-incremented,
// unlike emit_ptx_tiled -- this kernel has far more live values and an
// unrolled inner loop, so a never-reuse counter would need hundreds of
// names for no benefit; every reused slot below is fully consumed in
// program order before its next reuse, so this is safe non-SSA PTX, not a
// hazard):
//   .b32  r0 M, r1 N, r2 K, r3 tid, r4 blockCol, r5 blockRow,
//         r6 threadRow, r7 threadCol, r8 rowBase, r9 colBase, r10 kTile,
//         r11-16 cooperative-load scratch (reused x4: A it0/it1, B it0/it1),
//         r17-18 setup-only scratch, r19 outRowBase, r20 outColBase,
//         r21-24 outCol[0..TN), r25-26 epilogue scratch, r27-28 setup-only
//   .b64  rd0 A ptr, rd1 B ptr, rd2 As base, rd3 Bs base,
//         rd4 As write addr (scratch), rd5 Bs write addr (scratch),
//         rd6 global read addr (scratch), rd7 As read base (persistent),
//         rd8 Bs read base (persistent), rd9 out ptr,
//         rd10.. one per ADD epilogue operand (persistent),
//         rd50-51 setup-only scratch, rd60 epilogue read scratch, rd61 epilogue store scratch
//   .f32  f0-15 accumulators (TM*TN), f16-19 regM (TM), f20-23 regN (TN),
//         f30 cooperative-load scratch, f31 epilogue bias scratch, f32 epilogue zero scratch
//   .pred p0 tile-loop test, p1-3 load boundary (reused x4), p4-6 epilogue boundary (reused per cell)
char* emit_ptx_blocked(const Node* fused) {
    Buf b = {0};
    const int BM = GEMM_BM, BN = GEMM_BN, BK = GEMM_BK, TM = GEMM_TM, TN = GEMM_TN;
    const int NTHREADS = GEMM_NTHREADS;
    const int A_ELEMS = BM * BK, B_ELEMS = BK * BN;
    const int A_ITERS = A_ELEMS / NTHREADS, B_ITERS = B_ELEMS / NTHREADS;

    int n_add = 0;
    for (int i = 0; i < fused->n_epilogue; i++)
        if (fused->epilogue[i].op == OP_ADD) n_add++;

    bprintf(&b,
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n\n"
        ".entry fused(\n"
        "    .param .u64 p_a,\n"
        "    .param .u64 p_b,\n");
    for (int i = 0; i < n_add; i++)
        bprintf(&b, "    .param .u64 p_op%d,\n", i);
    bprintf(&b,
        "    .param .u64 p_out,\n"
        "    .param .u32 p_M,\n"
        "    .param .u32 p_N,\n"
        "    .param .u32 p_K\n"
        ")\n"
        "{\n"
        "    .reg .f32  %%f<64>;\n"
        "    .reg .b32  %%r<40>;\n"
        "    .reg .b64  %%rd<80>;\n"
        "    .reg .pred %%p<8>;\n"
        "    .shared .align 16 .b8 As[%d];\n"
        "    .shared .align 16 .b8 Bs[%d];\n\n",
        A_ELEMS * 4, B_ELEMS * 4);

    // ---- params, thread/block indices ----
    bprintf(&b,
        "    ld.param.u32 %%r0, [p_M];\n"
        "    ld.param.u32 %%r1, [p_N];\n"
        "    ld.param.u32 %%r2, [p_K];\n"
        "    ld.param.u64 %%rd0, [p_a];\n"
        "    ld.param.u64 %%rd1, [p_b];\n\n"
        "    mov.u32     %%r3, %%tid.x;\n"
        "    mov.u32     %%r4, %%ctaid.x;\n"
        "    mov.u32     %%r5, %%ctaid.y;\n"
        "    div.u32     %%r6, %%r3, %d;\n"      // threadRow = tid / (BN/TN)
        "    rem.u32     %%r7, %%r3, %d;\n"      // threadCol = tid % (BN/TN)
        "    mul.lo.u32  %%r8, %%r5, %d;\n"      // rowBase = blockRow*BM
        "    mul.lo.u32  %%r9, %%r4, %d;\n\n",   // colBase = blockCol*BN
        BN / TN, BN / TN, BM, BN);

    // ---- shared mem bases + persistent read-base addresses ----
    bprintf(&b,
        "    mov.u64     %%rd2, As;\n"
        "    mov.u64     %%rd3, Bs;\n"
        "    mul.lo.u32  %%r17, %%r6, %d;\n"     // rowOff_A = threadRow*TM (As is stored transposed: [k][m])
        "    cvt.u64.u32 %%rd50, %%r17;\n"
        "    shl.b64     %%rd50, %%rd50, 2;\n"
        "    add.u64     %%rd7, %%rd2, %%rd50;\n" // As read base
        "    mul.lo.u32  %%r18, %%r7, %d;\n"     // colOff_B = threadCol*TN
        "    cvt.u64.u32 %%rd51, %%r18;\n"
        "    shl.b64     %%rd51, %%rd51, 2;\n"
        "    add.u64     %%rd8, %%rd3, %%rd51;\n\n", // Bs read base
        TM, TN);

    // ---- accumulators ----
    for (int i = 0; i < TM * TN; i++)
        bprintf(&b, "    mov.f32     %%f%d, 0f00000000;\n", i);
    bprintf(&b, "\n");

    // ---- outer k-tile loop ----
    bprintf(&b,
        "    mov.u32     %%r10, 0;\n"
        "TILE_LOOP:\n"
        "    setp.ge.u32 %%p0, %%r10, %%r2;\n"
        "    @%%p0 bra   TILE_DONE;\n\n");

    // Cooperative load of As from A[rowBase.., kTile..], stored TRANSPOSED
    // in shared memory as As[k][m] (not As[m][k]) -- this is what makes the
    // regM read below a single contiguous ld.shared.v4.f32 instead of 4
    // strided scalar loads (see docs/research/gemm-optimization.md): for a
    // fixed kk, the TM values across i=0..TM-1 are BM*4 bytes apart in the
    // natural [m][k] layout but 4 bytes apart (contiguous) in [k][m].
    for (int it = 0; it < A_ITERS; it++) {
        const char* e = it == 0 ? "%r3" : "%r11";
        if (it > 0) bprintf(&b, "    add.u32     %%r11, %%r3, %d;\n", it * NTHREADS);
        bprintf(&b,
            "    div.u32     %%r12, %s, %d;\n"      // arow
            "    rem.u32     %%r13, %s, %d;\n"      // acol
            "    add.u32     %%r14, %%r8, %%r12;\n" // globalRow
            "    add.u32     %%r15, %%r10, %%r13;\n" // globalCol
            "    setp.lt.u32 %%p1, %%r14, %%r0;\n"  // globalRow < M
            "    setp.lt.u32 %%p2, %%r15, %%r2;\n"  // globalCol < K
            "    and.pred    %%p3, %%p1, %%p2;\n"
            "    mad.lo.u32  %%r16, %%r14, %%r2, %%r15;\n"
            "    cvt.u64.u32 %%rd6, %%r16;\n"
            "    shl.b64     %%rd6, %%rd6, 2;\n"
            "    add.u64     %%rd6, %%rd0, %%rd6;\n"
            "    mov.f32     %%f30, 0f00000000;\n"
            "    @%%p3 ld.global.f32 %%f30, [%%rd6];\n"
            "    mad.lo.u32  %%r16, %%r13, %d, %%r12;\n" // writeIdx = acol*BM + arow (transposed)
            "    cvt.u64.u32 %%rd4, %%r16;\n"
            "    shl.b64     %%rd4, %%rd4, 2;\n"
            "    add.u64     %%rd4, %%rd2, %%rd4;\n"
            "    st.shared.f32 [%%rd4], %%f30;\n\n",
            e, BK, e, BK, BM);
    }

    // Cooperative load of Bs[BK][BN] from B[kTile.., colBase..]
    for (int it = 0; it < B_ITERS; it++) {
        const char* e = it == 0 ? "%r3" : "%r11";
        if (it > 0) bprintf(&b, "    add.u32     %%r11, %%r3, %d;\n", it * NTHREADS);
        bprintf(&b,
            "    div.u32     %%r12, %s, %d;\n"      // brow
            "    rem.u32     %%r13, %s, %d;\n"      // bcol
            "    add.u32     %%r14, %%r10, %%r12;\n" // globalRow
            "    add.u32     %%r15, %%r9, %%r13;\n"  // globalCol
            "    setp.lt.u32 %%p1, %%r14, %%r2;\n"  // globalRow < K
            "    setp.lt.u32 %%p2, %%r15, %%r1;\n"  // globalCol < N
            "    and.pred    %%p3, %%p1, %%p2;\n"
            "    mad.lo.u32  %%r16, %%r14, %%r1, %%r15;\n"
            "    cvt.u64.u32 %%rd6, %%r16;\n"
            "    shl.b64     %%rd6, %%rd6, 2;\n"
            "    add.u64     %%rd6, %%rd1, %%rd6;\n"
            "    mov.f32     %%f30, 0f00000000;\n"
            "    @%%p3 ld.global.f32 %%f30, [%%rd6];\n"
            "    cvt.u64.u32 %%rd5, %s;\n"
            "    shl.b64     %%rd5, %%rd5, 2;\n"
            "    add.u64     %%rd5, %%rd3, %%rd5;\n"
            "    st.shared.f32 [%%rd5], %%f30;\n\n",
            e, BN, e, BN, e);
    }

    bprintf(&b, "    bar.sync    0;\n\n");

    // ---- compute: unrolled kk = 0..BK-1 ----
    // regM/regN loads are vectorized (ld.shared.v4.f32, one instruction for
    // all TM/TN values instead of TM/TN scalar loads) when TM==TN==4 -- the
    // only per-thread tile size this emitter currently generates. As is
    // stored transposed specifically so this regM load is contiguous (see
    // the cooperative-load comment above); regN's is contiguous in B's
    // natural layout already. Falls back to scalar loads if that ever
    // changes, so a future TM/TN tweak fails to compile cleanly rather than
    // silently emitting a wrong vector load.
    for (int kk = 0; kk < BK; kk++) {
        if (TM == 4) {
            bprintf(&b, "    ld.shared.v4.f32 {%%f16,%%f17,%%f18,%%f19}, [%%rd7+%d];\n", kk * BM * 4);
        } else {
            for (int i = 0; i < TM; i++)
                bprintf(&b, "    ld.shared.f32 %%f%d, [%%rd7+%d];\n", 16 + i, (kk * BM + i) * 4);
        }
        if (TN == 4) {
            bprintf(&b, "    ld.shared.v4.f32 {%%f20,%%f21,%%f22,%%f23}, [%%rd8+%d];\n", kk * BN * 4);
        } else {
            for (int j = 0; j < TN; j++)
                bprintf(&b, "    ld.shared.f32 %%f%d, [%%rd8+%d];\n", 20 + j, (kk * BN + j) * 4);
        }
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                bprintf(&b, "    fma.rn.f32  %%f%d, %%f%d, %%f%d, %%f%d;\n",
                        i * TN + j, 16 + i, 20 + j, i * TN + j);
        bprintf(&b, "\n");
    }

    bprintf(&b,
        "    bar.sync    0;\n"
        "    add.u32     %%r10, %%r10, %d;\n"
        "    bra         TILE_LOOP;\n"
        "TILE_DONE:\n\n",
        BK);

    // ---- epilogue + store ----
    bprintf(&b, "    ld.param.u64 %%rd9, [p_out];\n");
    for (int i = 0; i < n_add; i++)
        bprintf(&b, "    ld.param.u64 %%rd%d, [p_op%d];\n", 10 + i, i);

    bprintf(&b,
        "    mul.lo.u32  %%r27, %%r6, %d;\n"     // threadRow*TM
        "    add.u32     %%r19, %%r8, %%r27;\n"  // outRowBase
        "    mul.lo.u32  %%r28, %%r7, %d;\n"      // threadCol*TN
        "    add.u32     %%r20, %%r9, %%r28;\n\n", // outColBase
        TM, TN);
    for (int j = 0; j < TN; j++)
        bprintf(&b, "    add.u32     %%r%d, %%r20, %d;\n", 21 + j, j);
    bprintf(&b, "\n");

    for (int i = 0; i < TM; i++) {
        bprintf(&b,
            "    add.u32     %%r25, %%r19, %d;\n"
            "    setp.lt.u32 %%p4, %%r25, %%r0;\n",
            i);
        for (int j = 0; j < TN; j++) {
            int acc = i * TN + j;
            bprintf(&b,
                "    setp.lt.u32 %%p5, %%r%d, %%r1;\n"
                "    and.pred    %%p6, %%p4, %%p5;\n"
                "    mad.lo.u32  %%r26, %%r25, %%r1, %%r%d;\n",
                21 + j, 21 + j);

            int add_idx = 0;
            for (int e = 0; e < fused->n_epilogue; e++) {
                if (fused->epilogue[e].op == OP_ADD) {
                    bprintf(&b,
                        "    cvt.u64.u32 %%rd60, %%r26;\n"
                        "    shl.b64     %%rd60, %%rd60, 2;\n"
                        "    add.u64     %%rd60, %%rd%d, %%rd60;\n"
                        "    mov.f32     %%f31, 0f00000000;\n"
                        "    @%%p6 ld.global.f32 %%f31, [%%rd60];\n"
                        "    add.f32     %%f%d, %%f%d, %%f31;\n",
                        10 + add_idx, acc, acc);
                    add_idx++;
                } else if (fused->epilogue[e].op == OP_RELU) {
                    bprintf(&b,
                        "    mov.f32     %%f32, 0f00000000;\n"
                        "    max.f32     %%f%d, %%f%d, %%f32;\n",
                        acc, acc);
                }
            }

            bprintf(&b,
                "    cvt.u64.u32 %%rd61, %%r26;\n"
                "    shl.b64     %%rd61, %%rd61, 2;\n"
                "    add.u64     %%rd61, %%rd9, %%rd61;\n"
                "    @%%p6 st.global.f32 [%%rd61], %%f%d;\n\n",
                acc);
        }
    }

    bprintf(&b, "    ret;\n}\n");
    return b.s;
}

char* emit_ptx(const Node* fused) { return emit_ptx_blocked(fused); }
