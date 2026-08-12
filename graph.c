// Computational graph (DAG) layer over the tensor kernels.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "graph.h"

static Node* make_node(OpType op, Node** inputs, int n_inputs) {
    Node* n = (Node*)calloc(1, sizeof(Node));
    if (!n) return NULL;
    n->op = op;
    n->n_inputs = n_inputs;
    if (n_inputs > 0) {
        n->inputs = (Node**)malloc(n_inputs * sizeof(Node*));
        if (!n->inputs) { free(n); return NULL; }
        for (int i = 0; i < n_inputs; i++) n->inputs[i] = inputs[i];
    }
    return n;
}

Node* input_node(Tensor* t) {
    Node* n = make_node(OP_INPUT, NULL, 0);
    if (n) n->output = t; // owns_output stays 0: caller owns this tensor
    return n;
}

Node* const_node(Tensor* t) {
    Node* n = make_node(OP_INPUT, NULL, 0);
    if (n) { n->output = t; n->is_const = 1; } // owns_output stays 0: caller owns t
    return n;
}

Node* fused_node(Node* a, Node* b, EpStep* epilogue, int n_epilogue) {
    int n_operands = 0;
    for (int i = 0; i < n_epilogue; i++) if (epilogue[i].op == OP_ADD) n_operands++;

    int n_in = 2 + n_operands;
    Node** ins = (Node**)malloc(n_in * sizeof(Node*));
    if (!ins) return NULL;
    ins[0] = a; ins[1] = b;
    int k = 2;
    for (int i = 0; i < n_epilogue; i++) if (epilogue[i].op == OP_ADD) ins[k++] = epilogue[i].operand;

    Node* n = make_node(OP_FUSED, ins, n_in);
    free(ins);
    if (!n) return NULL;

    n->epilogue = (EpStep*)malloc(n_epilogue * sizeof(EpStep));
    if (!n->epilogue) { free(n->inputs); free(n); return NULL; }
    memcpy(n->epilogue, epilogue, n_epilogue * sizeof(EpStep));
    n->n_epilogue = n_epilogue;
    return n;
}

Node* g_matmul(Node* a, Node* b) { return make_node(OP_MATMUL, (Node*[]){a, b}, 2); }
Node* g_add(Node* a, Node* b)    { return make_node(OP_ADD,    (Node*[]){a, b}, 2); }
Node* g_mul(Node* a, Node* b)    { return make_node(OP_MUL,    (Node*[]){a, b}, 2); }
Node* g_relu(Node* x)            { return make_node(OP_RELU,      (Node*[]){x}, 1); }
Node* g_softmax(Node* x)         { return make_node(OP_SOFTMAX,   (Node*[]){x}, 1); }
Node* g_transpose(Node* x)       { return make_node(OP_TRANSPOSE, (Node*[]){x}, 1); }

// Execute a fused region in one loop per output cell: matmul dot-product, then
// the element-wise epilogue applied in a register, then a single store. No
// intermediate tensors are allocated.
static Tensor* eval_fused(Node* node) {
    Tensor* a = node->inputs[0]->output;
    Tensor* b = node->inputs[1]->output;
    if (a->ndim < 2 || b->ndim < 2 || a->ndim != b->ndim ||
        a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) {
        fprintf(stderr, "eval_fused: incompatible matmul operand shapes\n");
        return NULL;
    }
    for (int i = 0; i < a->ndim - 2; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "eval_fused: batch dimension %d mismatch\n", i);
            return NULL;
        }
    }

    int M = a->shape[a->ndim - 2], K = a->shape[a->ndim - 1], N = b->shape[b->ndim - 1];
    size_t batch = a->size / ((size_t)M * K);

    int* os = (int*)malloc(a->ndim * sizeof(int));
    if (!os) return NULL;
    memcpy(os, a->shape, a->ndim * sizeof(int));
    os[a->ndim - 1] = N;
    Tensor* r = create_tensor(NULL, os, a->ndim);
    free(os);
    if (!r) return NULL;

    for (size_t bt = 0; bt < batch; bt++) {
        const float* A = a->data + bt * M * K;
        const float* B = b->data + bt * K * N;
        float* C = r->data + bt * M * N;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float x = 0;
                for (int k = 0; k < K; k++) x += A[i * K + k] * B[k * N + j];
                for (int e = 0; e < node->n_epilogue; e++) {
                    EpStep s = node->epilogue[e];
                    if (s.op == OP_ADD) {
                        // Same broadcast trick as tensor_add: an operand
                        // smaller than the full [batch,M,N] output (a bias
                        // vector shaped [N]) repeats via modulo.
                        Tensor* op = s.operand->output;
                        x += op->data[(bt * M * N + i * N + j) % op->size];
                    } else if (s.op == OP_RELU) {
                        x = x > 0 ? x : 0;
                    }
                }
                C[i * N + j] = x;
            }
        }
    }
    return r;
}

Tensor* eval_op(Node* node, Tensor** in) {
    switch (node->op) {
        case OP_MATMUL:    return matmul(in[0], in[1]);
        case OP_ADD:       return tensor_add(in[0], in[1]);
        case OP_MUL:       return tensor_mul(in[0], in[1]);
        case OP_RELU:      return relu(in[0]);
        case OP_SOFTMAX:   return softmax(in[0]);
        case OP_TRANSPOSE: return transpose(in[0]);
        case OP_FUSED:     return eval_fused(node);
        case OP_INPUT:     return NULL;   // output was set at creation
    }
    return NULL;
}

Tensor* execute(Node* node) {
    if (!node) return NULL;
    if (node->output) return node->output; // memoized (incl. all OP_INPUT)

    for (int i = 0; i < node->n_inputs; i++)
        if (!execute(node->inputs[i])) return NULL;

    Tensor** in = (Tensor**)malloc(node->n_inputs * sizeof(Tensor*));
    if (node->n_inputs > 0 && !in) return NULL;
    for (int i = 0; i < node->n_inputs; i++) in[i] = node->inputs[i]->output;

    Tensor* out = eval_op(node, in);
    free(in);

    if (!out) { fprintf(stderr, "execute: op %d failed\n", node->op); return NULL; }
    node->output = out;
    node->owns_output = 1;
    return out;
}

// Collect each reachable node once (dedup via the visited flag) into `*out`.
static void collect_rec(Node* n, Node*** out, int* count, int* cap) {
    if (!n || n->visited) return;
    n->visited = 1;
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 8;
        *out = (Node**)realloc(*out, *cap * sizeof(Node*));
    }
    (*out)[(*count)++] = n;
    for (int i = 0; i < n->n_inputs; i++) collect_rec(n->inputs[i], out, count, cap);
}

int graph_collect(Node* root, Node*** out) {
    *out = NULL;
    int count = 0, cap = 0;
    collect_rec(root, out, &count, &cap);
    for (int i = 0; i < count; i++) (*out)[i]->visited = 0;
    return count;
}

void free_graph(Node* root) {
    Node** seen;
    int count = graph_collect(root, &seen);
    for (int i = 0; i < count; i++) {
        if (seen[i]->owns_output) free_tensor(seen[i]->output);
        free(seen[i]->inputs);
        free(seen[i]->epilogue);
        free(seen[i]);
    }
    free(seen);
}
