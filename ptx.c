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
// hazard). The .f32 side and the epilogue's outCol array scale with TM/TN
// (computed as F_*/R_* locals below, not hardcoded) since accumulators
// alone need TM*TN registers -- a fixed layout sized for TM=TN=4 would
// silently collide the moment either grows (e.g. regM overlapping
// still-live accumulators). The .b64/.pred side doesn't depend on TM/TN
// and stays fixed:
//   .b32  r0 M, r1 N, r2 K, r3 tid, r4 blockCol, r5 blockRow,
//         r6 threadRow, r7 threadCol, r8 rowBase, r9 colBase, r10 kTile,
//         r11-16 cooperative-load scratch (reused x4: A it0/it1, B it0/it1),
//         r17-18 setup-only scratch, r19 outRowBase, r20 outColBase,
//         R_OUTCOL.. outCol[0..TN) (dynamic base), then R_EPI_ROW/R_EPI_IDX
//         (epilogue scratch), then R_SETUP_A/R_SETUP_B (setup-only)
//   .b64  rd0 A ptr, rd1 B ptr, rd2 As base, rd3 Bs base,
//         rd4 As write addr (scratch), rd5 Bs write addr (scratch),
//         rd6 global read addr (scratch), rd7 As read base (persistent),
//         rd8 Bs read base (persistent), rd9 out ptr,
//         rd10.. one per ADD epilogue operand (persistent),
//         rd50-51 setup-only scratch, rd60 epilogue read scratch, rd61 epilogue store scratch
//   .f32  F_ACC.. accumulators (TM*TN), F_REGM.. regM (TM), F_REGN.. regN (TN),
//         F_SCRATCH cooperative-load scratch, F_BIAS epilogue bias, F_ZERO epilogue zero
//   .pred p0 tile-loop test, p1-3 load boundary (reused x4), p4-6 epilogue boundary (reused per cell)
char* emit_ptx_blocked(const Node* fused) {
    Buf b = {0};
    const int BM = GEMM_BM, BN = GEMM_BN, BK = GEMM_BK, TM = GEMM_TM, TN = GEMM_TN;
    const int NTHREADS = GEMM_NTHREADS;
    const int A_ELEMS = BM * BK, B_ELEMS = BK * BN;
    const int A_ITERS = A_ELEMS / NTHREADS, B_ITERS = B_ELEMS / NTHREADS;

    // Register bases scale with TM/TN (accumulators alone need TM*TN
    // registers) so they're computed, not hardcoded -- a fixed layout sized
    // for TM=TN=4 would silently collide (e.g. regM overlapping still-live
    // accumulators) the moment either grows. See the register-plan comment
    // above the struct-less layout below for the .b64/.pred side, which
    // doesn't scale with TM/TN and stays fixed.
    const int F_ACC = 0, F_REGM = TM * TN, F_REGN = F_REGM + TM;
    const int F_SCRATCH = F_REGN + TN, F_BIAS = F_SCRATCH + 1, F_ZERO = F_SCRATCH + 2;
    const int F_POOL = F_ZERO + 1;
    const int R_OUTCOL = 21, R_EPI_ROW = R_OUTCOL + TN, R_EPI_IDX = R_EPI_ROW + 1;
    const int R_SETUP_A = R_EPI_IDX + 1, R_SETUP_B = R_SETUP_A + 1;
    const int R_POOL = R_SETUP_B + 1;

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
        "    .reg .f32  %%f<%d>;\n"
        "    .reg .b32  %%r<%d>;\n"
        "    .reg .b64  %%rd<80>;\n"
        "    .reg .pred %%p<8>;\n"
        "    .shared .align 16 .b8 As[%d];\n"
        "    .shared .align 16 .b8 Bs[%d];\n\n",
        F_POOL, R_POOL, A_ELEMS * 4, B_ELEMS * 4);

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
        bprintf(&b, "    mov.f32     %%f%d, 0f00000000;\n", F_ACC + i);
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
            "    mov.f32     %%f%d, 0f00000000;\n"
            "    @%%p3 ld.global.f32 %%f%d, [%%rd6];\n"
            "    mad.lo.u32  %%r16, %%r13, %d, %%r12;\n" // writeIdx = acol*BM + arow (transposed)
            "    cvt.u64.u32 %%rd4, %%r16;\n"
            "    shl.b64     %%rd4, %%rd4, 2;\n"
            "    add.u64     %%rd4, %%rd2, %%rd4;\n"
            "    st.shared.f32 [%%rd4], %%f%d;\n\n",
            e, BK, e, BK, F_SCRATCH, F_SCRATCH, BM, F_SCRATCH);
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
            "    mov.f32     %%f%d, 0f00000000;\n"
            "    @%%p3 ld.global.f32 %%f%d, [%%rd6];\n"
            "    cvt.u64.u32 %%rd5, %s;\n"
            "    shl.b64     %%rd5, %%rd5, 2;\n"
            "    add.u64     %%rd5, %%rd3, %%rd5;\n"
            "    st.shared.f32 [%%rd5], %%f%d;\n\n",
            e, BN, e, BN, F_SCRATCH, F_SCRATCH, e, F_SCRATCH);
    }

    bprintf(&b, "    bar.sync    0;\n\n");

    // ---- compute: unrolled kk = 0..BK-1 ----
    // regM/regN loads are vectorized (ld.shared.v4.f32 per 4-element chunk,
    // one instruction per 4 values instead of 4 scalar loads) whenever
    // TM/TN is a multiple of 4 -- true for every per-thread tile size this
    // emitter has generated so far (4 and 8). As is stored transposed
    // specifically so the regM chunks are contiguous (see the cooperative-
    // load comment above); regN's are contiguous in B's natural layout
    // already. Falls back to scalar loads for a TM/TN this doesn't evenly
    // divide, so a future odd tile size fails to compile cleanly rather
    // than silently emitting a wrong vector load.
    for (int kk = 0; kk < BK; kk++) {
        if (TM % 4 == 0) {
            for (int c = 0; c < TM; c += 4)
                bprintf(&b, "    ld.shared.v4.f32 {%%f%d,%%f%d,%%f%d,%%f%d}, [%%rd7+%d];\n",
                        F_REGM + c, F_REGM + c + 1, F_REGM + c + 2, F_REGM + c + 3,
                        (kk * BM + c) * 4);
        } else {
            for (int i = 0; i < TM; i++)
                bprintf(&b, "    ld.shared.f32 %%f%d, [%%rd7+%d];\n", F_REGM + i, (kk * BM + i) * 4);
        }
        if (TN % 4 == 0) {
            for (int c = 0; c < TN; c += 4)
                bprintf(&b, "    ld.shared.v4.f32 {%%f%d,%%f%d,%%f%d,%%f%d}, [%%rd8+%d];\n",
                        F_REGN + c, F_REGN + c + 1, F_REGN + c + 2, F_REGN + c + 3,
                        (kk * BN + c) * 4);
        } else {
            for (int j = 0; j < TN; j++)
                bprintf(&b, "    ld.shared.f32 %%f%d, [%%rd8+%d];\n", F_REGN + j, (kk * BN + j) * 4);
        }
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                bprintf(&b, "    fma.rn.f32  %%f%d, %%f%d, %%f%d, %%f%d;\n",
                        F_ACC + i * TN + j, F_REGM + i, F_REGN + j, F_ACC + i * TN + j);
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
        "    mul.lo.u32  %%r%d, %%r6, %d;\n"     // threadRow*TM
        "    add.u32     %%r19, %%r8, %%r%d;\n"  // outRowBase
        "    mul.lo.u32  %%r%d, %%r7, %d;\n"      // threadCol*TN
        "    add.u32     %%r20, %%r9, %%r%d;\n\n", // outColBase
        R_SETUP_A, TM, R_SETUP_A, R_SETUP_B, TN, R_SETUP_B);
    for (int j = 0; j < TN; j++)
        bprintf(&b, "    add.u32     %%r%d, %%r20, %d;\n", R_OUTCOL + j, j);
    bprintf(&b, "\n");

    for (int i = 0; i < TM; i++) {
        bprintf(&b,
            "    add.u32     %%r%d, %%r19, %d;\n"
            "    setp.lt.u32 %%p4, %%r%d, %%r0;\n",
            R_EPI_ROW, i, R_EPI_ROW);
        for (int j = 0; j < TN; j++) {
            int acc = F_ACC + i * TN + j;
            bprintf(&b,
                "    setp.lt.u32 %%p5, %%r%d, %%r1;\n"
                "    and.pred    %%p6, %%p4, %%p5;\n"
                "    mad.lo.u32  %%r%d, %%r%d, %%r1, %%r%d;\n",
                R_OUTCOL + j, R_EPI_IDX, R_EPI_ROW, R_OUTCOL + j);

            int add_idx = 0;
            for (int e = 0; e < fused->n_epilogue; e++) {
                if (fused->epilogue[e].op == OP_ADD) {
                    bprintf(&b,
                        "    cvt.u64.u32 %%rd60, %%r%d;\n"
                        "    shl.b64     %%rd60, %%rd60, 2;\n"
                        "    add.u64     %%rd60, %%rd%d, %%rd60;\n"
                        "    mov.f32     %%f%d, 0f00000000;\n"
                        "    @%%p6 ld.global.f32 %%f%d, [%%rd60];\n"
                        "    add.f32     %%f%d, %%f%d, %%f%d;\n",
                        R_EPI_IDX, 10 + add_idx, F_BIAS, F_BIAS, acc, acc, F_BIAS);
                    add_idx++;
                } else if (fused->epilogue[e].op == OP_RELU) {
                    bprintf(&b,
                        "    mov.f32     %%f%d, 0f00000000;\n"
                        "    max.f32     %%f%d, %%f%d, %%f%d;\n",
                        F_ZERO, acc, acc, F_ZERO);
                }
            }

            bprintf(&b,
                "    cvt.u64.u32 %%rd61, %%r%d;\n"
                "    shl.b64     %%rd61, %%rd61, 2;\n"
                "    add.u64     %%rd61, %%rd9, %%rd61;\n"
                "    @%%p6 st.global.f32 [%%rd61], %%f%d;\n\n",
                R_EPI_IDX, acc);
        }
    }

    bprintf(&b, "    ret;\n}\n");
    return b.s;
}

char* emit_ptx(const Node* fused) { return emit_ptx_blocked(fused); }

// ---- TF32 tensor-core kernel (docs/research/tensorcore-and-fusion.md) ----
// One warp (32 threads) per block computes one 16x8 output tile via a chain
// of mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32, accumulating over
// K in steps of 8. No shared memory: every k-step's A/B fragment elements
// are read straight from global memory, converted f32->tf32 with
// cvt.rna.tf32.f32 (RNA: keeps more mantissa than RZ, confirmed against the
// PTX ISA cvt reference), then fed to mma. Fragment row/col formulas below
// are transcribed verbatim from the PTX ISA "Matrix Fragments for
// mma.m16n8k8" section (.tf32 subsection) -- not derived, not guessed, a
// mistake here produces plausible-looking wrong numbers rather than a
// crash, so this is copied, not adapted.
//
// Register plan (non-SSA, same convention as emit_ptx_blocked):
//   .b32  r0 M, r1 N, r2 K, r3 laneid, r4 ctaid.x, r5 ctaid.y,
//         r6 groupID, r7 threadID_in_group, r8 blockRowBase, r9 blockColBase,
//         r10 kk (loop var), r11-15 address-computation scratch (reused
//         per element), r16-19 A fragment (a0..a3, tf32-in-b32),
//         r20-21 B fragment (b0..b1, tf32-in-b32), r22-24 epilogue scratch
//   .b64  rd0 A ptr, rd1 B ptr, rd9 out ptr, rd10.. ADD epilogue operand
//         ptrs, rd20 address scratch (reused)
//   .f32  f0-3 accumulator (c0..c3, doubles as mma's C and D operand list
//         across k-steps), f4-7 raw f32 scratch before tf32 conversion,
//         f8 zero (relu), f9 bias (add epilogue)
//   .pred p0 k-loop test, p1-3 load boundary (reused per element),
//         p4-6 store/epilogue boundary (reused per output cell)
char* emit_ptx_tensorcore(const Node* fused) {
    Buf b = {0};
    const int F_ACC = 0, F_RAW = 4, F_ZERO = 8, F_BIAS = 9;
    const int R_A = 16, R_B = 20;

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
        "    .reg .f32  %%f<10>;\n"
        "    .reg .b32  %%r<25>;\n"
        "    .reg .b64  %%rd<21>;\n"
        "    .reg .pred %%p<7>;\n\n");

    bprintf(&b,
        "    ld.param.u32 %%r0, [p_M];\n"
        "    ld.param.u32 %%r1, [p_N];\n"
        "    ld.param.u32 %%r2, [p_K];\n"
        "    ld.param.u64 %%rd0, [p_a];\n"
        "    ld.param.u64 %%rd1, [p_b];\n\n"
        "    mov.u32     %%r3, %%laneid;\n"
        "    mov.u32     %%r4, %%ctaid.x;\n"
        "    mov.u32     %%r5, %%ctaid.y;\n"
        "    shr.u32     %%r6, %%r3, 2;\n"       // groupID
        "    rem.u32     %%r7, %%r3, 4;\n"       // threadID_in_group
        "    mul.lo.u32  %%r8, %%r5, %d;\n"      // blockRowBase = ctaid.y*TC_BM
        "    mul.lo.u32  %%r9, %%r4, %d;\n\n",   // blockColBase = ctaid.x*TC_BN
        TC_BM, TC_BN);

    for (int i = 0; i < 4; i++)
        bprintf(&b, "    mov.f32     %%f%d, 0f00000000;\n", F_ACC + i);
    bprintf(&b, "\n");

    bprintf(&b,
        "    mov.u32     %%r10, 0;\n"
        "TC_LOOP:\n"
        "    setp.ge.u32 %%p0, %%r10, %%r2;\n"
        "    @%%p0 bra   TC_DONE;\n\n");

    // A fragment: a0 row=groupID col=tid_in_group, a1 row=groupID+8
    // col=tid_in_group, a2 row=groupID col=tid_in_group+4, a3
    // row=groupID+8 col=tid_in_group+4 (PTX ISA 9.7.15.5.7, .tf32).
    static const int a_row_off[4] = {0, 8, 0, 8};
    static const int a_col_off[4] = {0, 0, 4, 4};
    for (int i = 0; i < 4; i++) {
        bprintf(&b,
            "    add.u32     %%r11, %%r6, %d;\n"   // localRow
            "    add.u32     %%r12, %%r7, %d;\n"   // localCol
            "    add.u32     %%r13, %%r8, %%r11;\n" // globalRow
            "    add.u32     %%r14, %%r10, %%r12;\n" // globalCol = kk + localCol
            "    setp.lt.u32 %%p1, %%r13, %%r0;\n"
            "    setp.lt.u32 %%p2, %%r14, %%r2;\n"
            "    and.pred    %%p3, %%p1, %%p2;\n"
            "    mad.lo.u32  %%r15, %%r13, %%r2, %%r14;\n"
            "    cvt.u64.u32 %%rd20, %%r15;\n"
            "    shl.b64     %%rd20, %%rd20, 2;\n"
            "    add.u64     %%rd20, %%rd0, %%rd20;\n"
            "    mov.f32     %%f%d, 0f00000000;\n"
            "    @%%p3 ld.global.f32 %%f%d, [%%rd20];\n"
            "    cvt.rna.tf32.f32 %%r%d, %%f%d;\n\n",
            a_row_off[i], a_col_off[i], F_RAW, F_RAW, R_A + i, F_RAW);
    }

    // B fragment: b0 row=tid_in_group col=groupID, b1 row=tid_in_group+4
    // col=groupID (same section).
    static const int b_row_off[2] = {0, 4};
    for (int i = 0; i < 2; i++) {
        bprintf(&b,
            "    add.u32     %%r11, %%r7, %d;\n"   // localRow
            "    add.u32     %%r13, %%r10, %%r11;\n" // globalRow = kk + localRow
            "    add.u32     %%r14, %%r9, %%r6;\n"  // globalCol = blockColBase + groupID
            "    setp.lt.u32 %%p1, %%r13, %%r2;\n"
            "    setp.lt.u32 %%p2, %%r14, %%r1;\n"
            "    and.pred    %%p3, %%p1, %%p2;\n"
            "    mad.lo.u32  %%r15, %%r13, %%r1, %%r14;\n"
            "    cvt.u64.u32 %%rd20, %%r15;\n"
            "    shl.b64     %%rd20, %%rd20, 2;\n"
            "    add.u64     %%rd20, %%rd1, %%rd20;\n"
            "    mov.f32     %%f%d, 0f00000000;\n"
            "    @%%p3 ld.global.f32 %%f%d, [%%rd20];\n"
            "    cvt.rna.tf32.f32 %%r%d, %%f%d;\n\n",
            b_row_off[i], F_RAW, F_RAW, R_B + i, F_RAW);
    }

    bprintf(&b,
        "    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%%f%d,%%f%d,%%f%d,%%f%d}, {%%r%d,%%r%d,%%r%d,%%r%d}, {%%r%d,%%r%d}, {%%f%d,%%f%d,%%f%d,%%f%d};\n\n",
        F_ACC, F_ACC+1, F_ACC+2, F_ACC+3,
        R_A, R_A+1, R_A+2, R_A+3, R_B, R_B+1,
        F_ACC, F_ACC+1, F_ACC+2, F_ACC+3);

    bprintf(&b,
        "    add.u32     %%r10, %%r10, %d;\n"
        "    bra         TC_LOOP;\n"
        "TC_DONE:\n\n",
        TC_BK);

    // Epilogue + store. C/D fragment: c0 row=groupID col=2*tid_in_group,
    // c1 row=groupID col=2*tid_in_group+1, c2 row=groupID+8
    // col=2*tid_in_group, c3 row=groupID+8 col=2*tid_in_group+1.
    bprintf(&b, "    ld.param.u64 %%rd9, [p_out];\n");
    for (int i = 0; i < n_add; i++)
        bprintf(&b, "    ld.param.u64 %%rd%d, [p_op%d];\n", 10 + i, i);
    bprintf(&b, "\n");

    static const int c_row_off[4] = {0, 0, 8, 8};
    static const int c_col_off[4] = {0, 1, 0, 1};
    for (int i = 0; i < 4; i++) {
        int acc = F_ACC + i;
        bprintf(&b,
            "    add.u32     %%r22, %%r6, %d;\n"          // localRow
            "    mul.lo.u32  %%r23, %%r7, 2;\n"
            "    add.u32     %%r23, %%r23, %d;\n"          // localCol
            "    add.u32     %%r22, %%r8, %%r22;\n"        // globalRow
            "    add.u32     %%r23, %%r9, %%r23;\n"        // globalCol
            "    setp.lt.u32 %%p4, %%r22, %%r0;\n"
            "    setp.lt.u32 %%p5, %%r23, %%r1;\n"
            "    and.pred    %%p6, %%p4, %%p5;\n"
            "    mad.lo.u32  %%r24, %%r22, %%r1, %%r23;\n",
            c_row_off[i], c_col_off[i]);

        int add_idx = 0;
        for (int e = 0; e < fused->n_epilogue; e++) {
            if (fused->epilogue[e].op == OP_ADD) {
                bprintf(&b,
                    "    cvt.u64.u32 %%rd20, %%r24;\n"
                    "    shl.b64     %%rd20, %%rd20, 2;\n"
                    "    add.u64     %%rd20, %%rd%d, %%rd20;\n"
                    "    mov.f32     %%f%d, 0f00000000;\n"
                    "    @%%p6 ld.global.f32 %%f%d, [%%rd20];\n"
                    "    add.f32     %%f%d, %%f%d, %%f%d;\n",
                    10 + add_idx, F_BIAS, F_BIAS, acc, acc, F_BIAS);
                add_idx++;
            } else if (fused->epilogue[e].op == OP_RELU) {
                bprintf(&b,
                    "    mov.f32     %%f%d, 0f00000000;\n"
                    "    max.f32     %%f%d, %%f%d, %%f%d;\n",
                    F_ZERO, acc, acc, F_ZERO);
            }
        }

        bprintf(&b,
            "    cvt.u64.u32 %%rd20, %%r24;\n"
            "    shl.b64     %%rd20, %%rd20, 2;\n"
            "    add.u64     %%rd20, %%rd9, %%rd20;\n"
            "    @%%p6 st.global.f32 [%%rd20], %%f%d;\n\n",
            acc);
    }

    bprintf(&b, "    ret;\n}\n");
    return b.s;
}
