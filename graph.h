#ifndef GRAPH_H
#define GRAPH_H

#include "tensor.h"

typedef enum {
    OP_INPUT,
    OP_MATMUL,
    OP_ADD,
    OP_MUL,
    OP_RELU,
    OP_SOFTMAX,
    OP_TRANSPOSE,
    OP_FUSED
} OpType;

struct Node;
typedef struct { OpType op; struct Node* operand; } EpStep;

typedef struct Node {
    OpType op;
    struct Node** inputs;
    int n_inputs;
    Tensor* output;
    int owns_output;  // 1 if this node allocated output and must free it
    char visited;     // scratch flag for graph traversals; reset by the caller
    int uses;         // scratch consumer count (set by the optimizer)
    char is_const;    // 1 for const_node leaves
    EpStep* epilogue; // OP_FUSED only
    int n_epilogue;
} Node;

Node* input_node(Tensor* t);
Node* const_node(Tensor* t);
Node* fused_node(Node* a, Node* b, EpStep* epilogue, int n_epilogue);

Node* g_matmul(Node* a, Node* b);
Node* g_add(Node* a, Node* b);
Node* g_mul(Node* a, Node* b);
Node* g_relu(Node* x);
Node* g_softmax(Node* x);
Node* g_transpose(Node* x);

// Run one node's op given its already-computed input tensors. Used by execute
// and by constant folding.
Tensor* eval_op(Node* node, Tensor** in);

Tensor* execute(Node* node);

// Collect every node reachable from `root` exactly once into *out (caller frees
// *out). Returns the count. Resets each node's visited flag before returning.
int graph_collect(Node* root, Node*** out);

// Topological order of every node reachable from `root`: each node appears
// only after all of its own inputs -- a valid execution/dependency order.
// NOT the same as reversing graph_collect's preorder, which only happens to
// be valid for a tree/chain; a node with two consumers can be discovered
// through one before the other is even visited. Reverse this result for a
// reverse-mode traversal (e.g. backprop), where a node instead needs to come
// after all of its *consumers*. Same caller-frees/visited-reset contract as
// graph_collect.
int graph_topo_order(Node* root, Node*** out);

void free_graph(Node* root);

#endif
