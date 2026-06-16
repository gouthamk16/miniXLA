#ifndef GRAPH_H
#define GRAPH_H

#include "tensor.h"

typedef enum {
    OP_INPUT,
    OP_MATMUL,
    OP_ADD,
    OP_RELU,
    OP_SOFTMAX,
    OP_TRANSPOSE
} OpType;

typedef struct Node {
    OpType op;
    struct Node** inputs;
    int n_inputs;
    Tensor* output;
    int owns_output; // 1 if this node allocated output and must free it
    char visited;    // scratch flag for graph traversals; reset by the caller
} Node;

Node* input_node(Tensor* t);

Node* g_matmul(Node* a, Node* b);
Node* g_add(Node* a, Node* b);
Node* g_relu(Node* x);
Node* g_softmax(Node* x);
Node* g_transpose(Node* x);

Tensor* execute(Node* node);

void free_graph(Node* root);

#endif
