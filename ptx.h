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

// ---- TF32 tensor-core kernel (docs/research/tensorcore-and-fusion.md) ----
// One warp per block, one mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
// per K-step of 8, no shared memory (every k-step reads A/B straight from
// global memory). Deliberately the simplest correct thing, not the fastest
// one: this is the highest-risk kernel in the codebase (a fragment-layout
// mistake produces plausible-looking wrong numbers, not a crash), so
// correctness-first with a wide-open follow-up (shared-memory blocking for
// warp reuse) rather than trying to land both at once. Additive: does not
// replace emit_ptx_blocked, which stays the default for every other caller.
#define TC_BM 16
#define TC_BN 8
#define TC_BK 8
#define TC_NTHREADS 32

char* emit_ptx_tensorcore(const Node* fused);

#endif
