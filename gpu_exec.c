#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "gpu_exec.h"
#include "ptx.h"

// ---- Device pointer map (node → CUdeviceptr) ----

typedef struct { const Node* node; CUdeviceptr ptr; } DevEntry;
typedef struct { DevEntry* data; int count; int cap; } DevMap;

static void map_put(DevMap* m, const Node* n, CUdeviceptr p) {
    if (m->count == m->cap) {
        m->cap  = m->cap ? m->cap * 2 : 8;
        m->data = realloc(m->data, m->cap * sizeof(DevEntry));
    }
    m->data[m->count++] = (DevEntry){n, p};
}

static CUdeviceptr map_get(const DevMap* m, const Node* n) {
    for (int i = 0; i < m->count; i++)
        if (m->data[i].node == n) return m->data[i].ptr;
    return 0;
}

// ---- Module cache (PTX string → CUmodule) ----
// Different tile sizes produce different PTX strings so the cache key is the
// full PTX; no extra discriminator needed.

typedef struct { char* key; CUmodule mod; } CacheEntry;
typedef struct { CacheEntry* data; int count; int cap; } ModCache;

static CUmodule cache_lookup(const ModCache* c, const char* ptx) {
    for (int i = 0; i < c->count; i++)
        if (strcmp(c->data[i].key, ptx) == 0) return c->data[i].mod;
    return NULL;
}

static void cache_insert(ModCache* c, char* ptx_owned, CUmodule mod) {
    if (c->count == c->cap) {
        c->cap  = c->cap ? c->cap * 2 : 4;
        c->data = realloc(c->data, c->cap * sizeof(CacheEntry));
    }
    c->data[c->count++] = (CacheEntry){ptx_owned, mod};
}

// ---- Tile-size map (epilogue signature → optimal tile) ----

typedef struct { char ep_key[32]; int tile; } TuneEntry;

// Epilogue key: compact string encoding op sequence, e.g. "AR" = ADD,RELU.
static void ep_key_str(const Node* n, char key[32]) {
    int i = 0;
    for (int e = 0; e < n->n_epilogue && i < 31; e++)
        key[i++] = (n->epilogue[e].op == OP_ADD) ? 'A' : 'R';
    key[i] = 0;
}

// ---- GpuCtx ----

struct GpuCtx {
    CUdevice  dev;
    CUcontext ctx;
    ModCache  mod_cache;
    TuneEntry tune[32];
    int       n_tune;
};

static int cu_check(CUresult r, const char* label) {
    if (r == CUDA_SUCCESS) return 1;
    const char* s = NULL;
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuda: %s: %s\n", label, s ? s : "unknown");
    return 0;
}

GpuCtx* gpu_ctx_create(void) {
    GpuCtx* g = calloc(1, sizeof(GpuCtx));
    if (!g) return NULL;
    if (!cu_check(cuInit(0),                                  "cuInit"))                   goto fail;
    if (!cu_check(cuDeviceGet(&g->dev, 0),                    "cuDeviceGet"))              goto fail;
    if (!cu_check(cuDevicePrimaryCtxRetain(&g->ctx, g->dev),  "cuDevicePrimaryCtxRetain")) goto fail;
    if (!cu_check(cuCtxSetCurrent(g->ctx),                    "cuCtxSetCurrent"))          goto fail;
    return g;
fail:
    free(g);
    return NULL;
}

void gpu_ctx_destroy(GpuCtx* g) {
    if (!g) return;
    for (int i = 0; i < g->mod_cache.count; i++) {
        cuModuleUnload(g->mod_cache.data[i].mod);
        free(g->mod_cache.data[i].key);
    }
    free(g->mod_cache.data);
    if (g->ctx) cuDevicePrimaryCtxRelease(g->dev);
    free(g);
}

// ---- Shape inference ----

static void node_dims(const Node* n, int* rows, int* cols) {
    if (n->output) { *rows = n->output->shape[0]; *cols = n->output->shape[1]; return; }
    int k1, k2;
    node_dims(n->inputs[0], rows, &k1);
    node_dims(n->inputs[1], &k2, cols);
}

// ---- Tile lookup (returns PTX_TILE if not autotuned) ----

static int tile_for(const GpuCtx* g, const Node* nd) {
    char key[32];
    ep_key_str(nd, key);
    for (int i = 0; i < g->n_tune; i++)
        if (strcmp(g->tune[i].ep_key, key) == 0) return g->tune[i].tile;
    return PTX_TILE;
}

// ---- Shared: build + launch a fused kernel (used by both execute and autotune) ----

static CUresult launch_fused(CUfunction fn, int tile,
                              int M, int N, int K,
                              CUdeviceptr d_a, CUdeviceptr d_b,
                              CUdeviceptr* op_ptrs, int n_add,
                              CUdeviceptr d_out) {
    unsigned uM = M, uN = N, uK = K;
    void** params = malloc((n_add + 6) * sizeof(void*));
    if (!params) return CUDA_ERROR_OUT_OF_MEMORY;
    // Stable heap storage for operand ptrs (params holds address-of elements)
    CUdeviceptr* op_copy = n_add ? malloc(n_add * sizeof(CUdeviceptr)) : NULL;
    if (n_add && !op_copy) { free(params); return CUDA_ERROR_OUT_OF_MEMORY; }
    if (op_copy) memcpy(op_copy, op_ptrs, n_add * sizeof(CUdeviceptr));

    int pi = 0;
    params[pi++] = &d_a;
    params[pi++] = &d_b;
    for (int i = 0; i < n_add; i++) params[pi++] = &op_copy[i];
    params[pi++] = &d_out;
    params[pi++] = &uM;
    params[pi++] = &uN;
    params[pi++] = &uK;

    unsigned gx = ((unsigned)N + tile - 1) / tile;
    unsigned gy = ((unsigned)M + tile - 1) / tile;
    CUresult r = cuLaunchKernel(fn, gx, gy, 1, tile, tile, 1, 0, 0, params, NULL);
    free(params);
    free(op_copy);
    return r;
}

// ---- gpu_autotune ----

int gpu_autotune(Node* root, GpuCtx* g) {
    if (root->op != OP_FUSED) return PTX_TILE;

    char key[32];
    ep_key_str(root, key);
    for (int i = 0; i < g->n_tune; i++)
        if (strcmp(g->tune[i].ep_key, key) == 0) return g->tune[i].tile;

    // Require inputs to have host data available
    if (!root->inputs[0]->output || !root->inputs[1]->output) return PTX_TILE;

    int M, K, N, dummy;
    node_dims(root->inputs[0], &M, &K);
    node_dims(root->inputs[1], &dummy, &N);

    int n_add = 0;
    for (int i = 0; i < root->n_epilogue; i++)
        if (root->epilogue[i].op == OP_ADD) n_add++;

    // Upload inputs once for all tile variants
    Tensor* ta = root->inputs[0]->output;
    Tensor* tb = root->inputs[1]->output;
    CUdeviceptr d_a = 0, d_b = 0, d_out = 0;
    CUdeviceptr* d_ops = n_add ? calloc(n_add, sizeof(CUdeviceptr)) : NULL;

    if (!cu_check(cuMemAlloc(&d_a, ta->size * sizeof(float)), "autotune alloc a") ||
        !cu_check(cuMemcpyHtoD(d_a, ta->data, ta->size * sizeof(float)), "autotune H2D a") ||
        !cu_check(cuMemAlloc(&d_b, tb->size * sizeof(float)), "autotune alloc b") ||
        !cu_check(cuMemcpyHtoD(d_b, tb->data, tb->size * sizeof(float)), "autotune H2D b") ||
        !cu_check(cuMemAlloc(&d_out, (size_t)M * N * sizeof(float)), "autotune alloc out"))
        goto at_cleanup;

    {
        int ai = 0;
        for (int i = 0; i < root->n_epilogue; i++) {
            if (root->epilogue[i].op != OP_ADD) continue;
            Tensor* op = root->epilogue[i].operand->output;
            if (!cu_check(cuMemAlloc(&d_ops[ai], op->size * sizeof(float)), "autotune alloc op") ||
                !cu_check(cuMemcpyHtoD(d_ops[ai], op->data, op->size * sizeof(float)), "autotune H2D op"))
                goto at_cleanup;
            ai++;
        }
    }

    {
        int best_tile = PTX_TILE;
        float best_ms = 1e30f;
        const int TILES[] = {8, 16, 32};

        for (int ti = 0; ti < 3; ti++) {
            int tile = TILES[ti];
            char* ptx = emit_ptx_tiled(root, tile);
            CUmodule  mod = NULL;
            CUfunction fn = NULL;
            if (cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL) != CUDA_SUCCESS ||
                cuModuleGetFunction(&fn, mod, "fused") != CUDA_SUCCESS) {
                free(ptx); if (mod) cuModuleUnload(mod); continue;
            }
            free(ptx);

            // 2 warmup runs
            for (int w = 0; w < 2; w++) {
                launch_fused(fn, tile, M, N, K, d_a, d_b, d_ops, n_add, d_out);
                cuCtxSynchronize();
            }

            // 5 timed runs — take the minimum to avoid OS jitter
            CUevent ev0, ev1;
            cuEventCreate(&ev0, CU_EVENT_DEFAULT);
            cuEventCreate(&ev1, CU_EVENT_DEFAULT);
            float min_ms = 1e30f;
            for (int r = 0; r < 5; r++) {
                cuEventRecord(ev0, 0);
                launch_fused(fn, tile, M, N, K, d_a, d_b, d_ops, n_add, d_out);
                cuEventRecord(ev1, 0);
                cuEventSynchronize(ev1);
                float ms; cuEventElapsedTime(&ms, ev0, ev1);
                if (ms < min_ms) min_ms = ms;
            }
            cuEventDestroy(ev0); cuEventDestroy(ev1);

            printf("  autotune tile=%2d: %.3f ms\n", tile, min_ms);
            if (min_ms < best_ms) { best_ms = min_ms; best_tile = tile; }

            cuModuleUnload(mod);
        }

        if (g->n_tune < 32) {
            strncpy(g->tune[g->n_tune].ep_key, key, 31);
            g->tune[g->n_tune].tile = best_tile;
            g->n_tune++;
        }
        printf("  autotune best tile=%d (%.3f ms)\n", best_tile, best_ms);

at_cleanup:
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
        if (d_ops) {
            for (int i = 0; i < n_add; i++) if (d_ops[i]) cuMemFree(d_ops[i]);
            free(d_ops);
        }
        return best_tile;
    }
}

// ---- gpu_execute ----

Tensor* gpu_execute(Node* root, GpuCtx* g) {
    Node** nodes;
    int n = graph_collect(root, &nodes);
    // Reverse pre-order → topological order (leaves first)
    for (int i = 0, j = n-1; i < j; i++, j--) {
        Node* t = nodes[i]; nodes[i] = nodes[j]; nodes[j] = t;
    }

    DevMap  map  = {0};
    Tensor* result = NULL;
    int     ok   = 1;

    for (int ni = 0; ni < n && ok; ni++) {
        Node* nd = nodes[ni];

        if (nd->op == OP_INPUT) {
            size_t nb = nd->output->size * sizeof(float);
            CUdeviceptr ptr = 0;
            ok = cu_check(cuMemAlloc(&ptr, nb),                    "cuMemAlloc input") &&
                 cu_check(cuMemcpyHtoD(ptr, nd->output->data, nb), "cuMemcpyHtoD input");
            if (!ok) { if (ptr) cuMemFree(ptr); break; }
            map_put(&map, nd, ptr);

        } else if (nd->op == OP_FUSED) {
            int M, K, N, tmp;
            node_dims(nd->inputs[0], &M, &K);
            node_dims(nd->inputs[1], &tmp, &N);

            int n_add = 0;
            for (int i = 0; i < nd->n_epilogue; i++)
                if (nd->epilogue[i].op == OP_ADD) n_add++;

            // Use tuned tile if available, else default
            int tile = tile_for(g, nd);

            char* ptx  = emit_ptx_tiled(nd, tile);
            CUmodule  mod = cache_lookup(&g->mod_cache, ptx);
            CUfunction fn = NULL;
            if (mod) {
                free(ptx);
            } else {
                ok = cu_check(cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL), "cuModuleLoadDataEx");
                if (!ok) { free(ptx); break; }
                cache_insert(&g->mod_cache, ptx, mod);
            }
            ok = cu_check(cuModuleGetFunction(&fn, mod, "fused"), "cuModuleGetFunction");
            if (!ok) break;

            CUdeviceptr d_a   = map_get(&map, nd->inputs[0]);
            CUdeviceptr d_b   = map_get(&map, nd->inputs[1]);
            CUdeviceptr d_out = 0;
            ok = cu_check(cuMemAlloc(&d_out, (size_t)M * N * sizeof(float)), "cuMemAlloc out");
            if (!ok) break;
            map_put(&map, nd, d_out);

            CUdeviceptr* op_ptrs = n_add ? malloc(n_add * sizeof(CUdeviceptr)) : NULL;
            if (n_add && !op_ptrs) { ok = 0; break; }
            int ai = 0;
            for (int i = 0; i < nd->n_epilogue; i++)
                if (nd->epilogue[i].op == OP_ADD)
                    op_ptrs[ai++] = map_get(&map, nd->epilogue[i].operand);

            CUresult lr = launch_fused(fn, tile, M, N, K, d_a, d_b, op_ptrs, n_add, d_out);
            free(op_ptrs);
            ok = cu_check(lr, "cuLaunchKernel");

        } else {
            fprintf(stderr, "gpu_execute: op %d not OP_INPUT/OP_FUSED; call optimize() first\n", nd->op);
            ok = 0;
        }
    }

    if (ok) ok = cu_check(cuCtxSynchronize(), "cuCtxSynchronize");

    if (ok) {
        int M, N;
        node_dims(root, &M, &N);
        result = create_tensor(NULL, (int[]){M, N}, 2);
        CUdeviceptr d_root = map_get(&map, root);
        if (!result ||
            !cu_check(cuMemcpyDtoH(result->data, d_root,
                                   (size_t)M * N * sizeof(float)), "cuMemcpyDtoH")) {
            free_tensor(result);
            result = NULL;
        }
    }

    for (int i = 0; i < map.count; i++) if (map.data[i].ptr) cuMemFree(map.data[i].ptr);
    free(map.data);
    free(nodes);
    return result;
}
