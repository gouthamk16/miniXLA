// Computational graph (DAG) layer over the tensor kernels.
#include <stdio.h>
#include <stdlib.h>
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

Node* g_matmul(Node* a, Node* b) { return make_node(OP_MATMUL, (Node*[]){a, b}, 2); }
Node* g_add(Node* a, Node* b)    { return make_node(OP_ADD,    (Node*[]){a, b}, 2); }
Node* g_relu(Node* x)            { return make_node(OP_RELU,      (Node*[]){x}, 1); }
Node* g_softmax(Node* x)         { return make_node(OP_SOFTMAX,   (Node*[]){x}, 1); }
Node* g_transpose(Node* x)       { return make_node(OP_TRANSPOSE, (Node*[]){x}, 1); }

Tensor* execute(Node* node) {
    if (!node) return NULL;
    if (node->output) return node->output; // memoized (incl. all OP_INPUT)

    for (int i = 0; i < node->n_inputs; i++)
        if (!execute(node->inputs[i])) return NULL;

    Tensor** in = (Tensor**)malloc(node->n_inputs * sizeof(Tensor*));
    if (node->n_inputs > 0 && !in) return NULL;
    for (int i = 0; i < node->n_inputs; i++) in[i] = node->inputs[i]->output;

    Tensor* out = NULL;
    switch (node->op) {
        case OP_MATMUL:    out = matmul(in[0], in[1]); break;
        case OP_ADD:       out = tensor_add(in[0], in[1]); break;
        case OP_RELU:      out = relu(in[0]); break;
        case OP_SOFTMAX:   out = softmax(in[0]); break;
        case OP_TRANSPOSE: out = transpose(in[0]); break;
        case OP_INPUT:     break;   // unreachable: output was set at creation
    }
    free(in);

    if (!out) { fprintf(stderr, "execute: op %d failed\n", node->op); return NULL; }
    node->output = out;
    node->owns_output = 1;
    return out;
}

// Collect each reachable node once (dedup via the visited flag) into `seen`.
static void collect(Node* node, Node*** seen, int* count, int* cap) {
    if (!node || node->visited) return;
    node->visited = 1;
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 8;
        *seen = (Node**)realloc(*seen, *cap * sizeof(Node*));
    }
    (*seen)[(*count)++] = node;
    for (int i = 0; i < node->n_inputs; i++) collect(node->inputs[i], seen, count, cap);
}

void free_graph(Node* root) {
    Node** seen = NULL;
    int count = 0, cap = 0;
    collect(root, &seen, &count, &cap);
    for (int i = 0; i < count; i++) {
        if (seen[i]->owns_output) free_tensor(seen[i]->output);
        free(seen[i]->inputs);
        free(seen[i]);
    }
    free(seen);
}

int main(void) {
    // Build the idea.md example: relu(matmul(a, b) + c), with b shaped so the
    // matmul is valid: a[2,3] @ b[3,2] = [2,2], then + c[2,2], then relu.
    Tensor* a = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    Tensor* b = create_tensor((float[]){1, 2, 3, 4, 5, 6}, (int[]){3, 2}, 2);
    Tensor* c = create_tensor((float[]){-100, 0, 0, -100}, (int[]){2, 2}, 2);

    Node* na = input_node(a);
    Node* nb = input_node(b);
    Node* nc = input_node(c);
    Node* n1 = g_matmul(na, nb); // [[22,28],[49,64]]
    Node* n2 = g_add(n1, nc); // [[-78,28],[49,-36]]
    Node* n3 = g_relu(n2); // [[0,28],[49,0]]

    Tensor* result = execute(n3);
    if (!result) { fprintf(stderr, "graph execution failed\n"); return EXIT_FAILURE; }

    printf("relu(matmul(a,b) + c) (expect [[0,28],[49,0]]):\n");
    print_tensor(result);

    free_graph(n3);
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    return EXIT_SUCCESS;
}
