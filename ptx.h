#ifndef PTX_H
#define PTX_H

#include "graph.h"

// ---- 2D register-blocked kernel (docs/research/gemm-optimization.md) ----
// Block tile BMxBN, K-slice depth BK per outer iteration, each thread
// computes a TMxTN grid of output cells instead of one. Fixed constants for
// now (not yet autotuned -- see the research doc's priority list). Threads
// per block is a 1D count, not a tile x tile square.
#define GEMM_BM 128
#define GEMM_BN 128
#define GEMM_BK 8
#define GEMM_TM 8
#define GEMM_TN 8
#define GEMM_NTHREADS ((GEMM_BM / GEMM_TM) * (GEMM_BN / GEMM_TN))

char* emit_ptx_blocked(const Node* fused);

// Alias kept stable across kernel-implementation changes -- callers that
// just want "the current best kernel" use this name.
char* emit_ptx(const Node* fused);

#endif
