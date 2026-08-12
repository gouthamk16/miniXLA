// PTX emitter: 16x16 tiled shared-memory matmul + fused elementwise epilogue.
// One thread per output cell, organised as a 2-D (TILE x TILE) block.
// Handles arbitrary M, N, K via predicated boundary loads + output guard.
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

char* emit_ptx_tiled(const Node* fused, int tile) {
    Buf b = {0};
    const int T = tile;

    int n_add = 0;
    for (int i = 0; i < fused->n_epilogue; i++)
        if (fused->epilogue[i].op == OP_ADD) n_add++;

    // ---- Header and entry signature ----
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
        // .reg pools (over-declared so emission never runs out)
        "    .reg .f32  %%f<64>;\n"
        "    .reg .b32  %%r<32>;\n"
        "    .reg .b64  %%rd<128>;\n"
        "    .reg .pred %%p<8>;\n"
        // Two shared-memory tiles: s_a[T][T], s_b[T][T]
        "    .shared .align 16 .b8 s_a[%d];\n"
        "    .shared .align 16 .b8 s_b[%d];\n\n",
        T * T * 4, T * T * 4);

    // ---- Load dimensions and base pointers ----
    bprintf(&b,
        "    ld.param.u32 %%r0, [p_M];\n"    // r0 = M
        "    ld.param.u32 %%r1, [p_N];\n"    // r1 = N
        "    ld.param.u32 %%r2, [p_K];\n"    // r2 = K
        "    ld.param.u64 %%rd0, [p_a];\n"   // rd0 = A*
        "    ld.param.u64 %%rd1, [p_b];\n\n"); // rd1 = B*

    // ---- Thread/block indices → row and col ----
    bprintf(&b,
        "    mov.u32     %%r3, %%tid.x;\n"               // tx
        "    mov.u32     %%r4, %%tid.y;\n"               // ty
        "    mov.u32     %%r5, %%ctaid.x;\n"             // bx
        "    mov.u32     %%r6, %%ctaid.y;\n"             // by
        "    mad.lo.u32  %%r7, %%r6, %d, %%r4;\n"       // row = by*T + ty
        "    mad.lo.u32  %%r8, %%r5, %d, %%r3;\n\n",    // col = bx*T + tx
        T, T);

    // ---- Shared memory write pointers (one per thread) ----
    // Each thread writes the same smem slot each tile iteration:
    //   s_a[ty*T + tx]  and  s_b[ty*T + tx]
    bprintf(&b,
        "    mov.u64     %%rd2, s_a;\n"                  // rd2 = s_a base
        "    mov.u64     %%rd3, s_b;\n"                  // rd3 = s_b base
        "    mad.lo.u32  %%r9, %%r4, %d, %%r3;\n"       // smem_idx = ty*T + tx
        "    cvt.u64.u32 %%rd4, %%r9;\n"
        "    shl.b64     %%rd5, %%rd4, 2;\n"             // byte offset
        "    add.u64     %%rd6, %%rd2, %%rd5;\n"         // rd6 = s_a write ptr
        "    add.u64     %%rd7, %%rd3, %%rd5;\n\n",      // rd7 = s_b write ptr
        T);

    // ---- acc = 0; tile_k loop ----
    bprintf(&b,
        "    mov.f32     %%f0, 0f00000000;\n"
        "    mov.u32     %%r10, 0;\n"                    // tile_k = 0
        "TILE_LOOP:\n"
        "    setp.ge.u32 %%p0, %%r10, %%r2;\n"
        "    @%%p0 bra   TILE_DONE;\n\n");

    // Load A tile: s_a[ty*T + tx] = A[row, tile_k + tx]  (zero-pad if OOB)
    bprintf(&b,
        "    add.u32     %%r11, %%r10, %%r3;\n"          // a_col = tile_k + tx
        "    setp.lt.u32 %%p1, %%r7, %%r0;\n"            // row < M
        "    setp.lt.u32 %%p2, %%r11, %%r2;\n"           // a_col < K
        "    and.pred    %%p3, %%p1, %%p2;\n"
        "    mad.lo.u32  %%r12, %%r7, %%r2, %%r11;\n"    // A[row*K + a_col]
        "    cvt.u64.u32 %%rd8, %%r12;\n"
        "    shl.b64     %%rd9, %%rd8, 2;\n"
        "    add.u64     %%rd10, %%rd0, %%rd9;\n"
        "    mov.f32     %%f1, 0f00000000;\n"
        "    @%%p3 ld.global.f32 %%f1, [%%rd10];\n"
        "    st.shared.f32 [%%rd6], %%f1;\n\n");

    // Load B tile: s_b[ty*T + tx] = B[tile_k + ty, col]  (zero-pad if OOB)
    bprintf(&b,
        "    add.u32     %%r13, %%r10, %%r4;\n"          // b_row = tile_k + ty
        "    setp.lt.u32 %%p4, %%r13, %%r2;\n"           // b_row < K
        "    setp.lt.u32 %%p5, %%r8, %%r1;\n"            // col < N
        "    and.pred    %%p6, %%p4, %%p5;\n"
        "    mad.lo.u32  %%r14, %%r13, %%r1, %%r8;\n"    // B[b_row*N + col]
        "    cvt.u64.u32 %%rd11, %%r14;\n"
        "    shl.b64     %%rd12, %%rd11, 2;\n"
        "    add.u64     %%rd13, %%rd1, %%rd12;\n"
        "    mov.f32     %%f2, 0f00000000;\n"
        "    @%%p6 ld.global.f32 %%f2, [%%rd13];\n"
        "    st.shared.f32 [%%rd7], %%f2;\n\n");

    // Sync, inner k-loop (k = 0 .. T-1)
    bprintf(&b,
        "    bar.sync    0;\n\n"
        "    mov.u32     %%r15, 0;\n"                    // k = 0
        "INNER_LOOP:\n"
        "    setp.ge.u32 %%p7, %%r15, %d;\n"
        "    @%%p7 bra   INNER_DONE;\n\n",
        T);

    // s_a[ty*T + k]
    bprintf(&b,
        "    mad.lo.u32  %%r16, %%r4, %d, %%r15;\n"     // ty*T + k
        "    cvt.u64.u32 %%rd14, %%r16;\n"
        "    shl.b64     %%rd15, %%rd14, 2;\n"
        "    add.u64     %%rd16, %%rd2, %%rd15;\n"
        "    ld.shared.f32 %%f3, [%%rd16];\n\n",
        T);

    // s_b[k*T + tx]
    bprintf(&b,
        "    mad.lo.u32  %%r17, %%r15, %d, %%r3;\n"     // k*T + tx
        "    cvt.u64.u32 %%rd17, %%r17;\n"
        "    shl.b64     %%rd18, %%rd17, 2;\n"
        "    add.u64     %%rd19, %%rd3, %%rd18;\n"
        "    ld.shared.f32 %%f4, [%%rd19];\n\n",
        T);

    bprintf(&b,
        "    fma.rn.f32  %%f0, %%f3, %%f4, %%f0;\n"
        "    add.u32     %%r15, %%r15, 1;\n"
        "    bra         INNER_LOOP;\n"
        "INNER_DONE:\n\n"
        "    bar.sync    0;\n"
        "    add.u32     %%r10, %%r10, %d;\n"
        "    bra         TILE_LOOP;\n"
        "TILE_DONE:\n\n",
        T);

    // ---- Output guard ----
    bprintf(&b,
        "    setp.ge.u32 %%p0, %%r7, %%r0;\n"
        "    @%%p0 bra   DONE;\n"
        "    setp.ge.u32 %%p0, %%r8, %%r1;\n"
        "    @%%p0 bra   DONE;\n\n");

    // Flat output index t = row*N + col  (r18)
    bprintf(&b, "    mad.lo.u32  %%r18, %%r7, %%r1, %%r8;\n\n");

    // ---- Epilogue ----
    // rd starts at 20 (rd0..rd19 consumed), f at 5 (f0..f4 consumed)
    int rd = 20, f = 5, add_idx = 0;
    for (int i = 0; i < fused->n_epilogue; i++) {
        if (fused->epilogue[i].op == OP_ADD) {
            bprintf(&b,
                "    ld.param.u64 %%rd%d, [p_op%d];\n"
                "    cvt.u64.u32 %%rd%d, %%r18;\n"
                "    shl.b64     %%rd%d, %%rd%d, 2;\n"
                "    add.u64     %%rd%d, %%rd%d, %%rd%d;\n"
                "    ld.global.f32 %%f%d, [%%rd%d];\n"
                "    add.f32     %%f0, %%f0, %%f%d;\n\n",
                rd, add_idx,
                rd+1,
                rd+2, rd+1,
                rd+3, rd, rd+2,
                f, rd+3,
                f);
            rd += 4; f++; add_idx++;
        } else if (fused->epilogue[i].op == OP_RELU) {
            bprintf(&b,
                "    mov.f32     %%f%d, 0f00000000;\n"
                "    max.f32     %%f0, %%f0, %%f%d;\n\n",
                f, f);
            f++;
        }
    }

    // ---- Store out[t] = acc ----
    bprintf(&b,
        "    ld.param.u64 %%rd%d, [p_out];\n"
        "    cvt.u64.u32 %%rd%d, %%r18;\n"
        "    shl.b64     %%rd%d, %%rd%d, 2;\n"
        "    add.u64     %%rd%d, %%rd%d, %%rd%d;\n"
        "    st.global.f32 [%%rd%d], %%f0;\n\n",
        rd, rd+1, rd+2, rd+1, rd+3, rd, rd+2, rd+3);

    bprintf(&b, "DONE:\n    ret;\n}\n");
    return b.s;
}

char* emit_ptx(const Node* fused) { return emit_ptx_tiled(fused, PTX_TILE); }
