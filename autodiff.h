#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "graph.h"

typedef struct { const Node* node; Tensor* grad; } GradEntry;
typedef struct { GradEntry* entries; int count; } GradTape;

// Reverse-mode autodiff: computes d(sum(output))/d(node) for every node
// reachable from `output`, via already-memoized forward tensors. Requires
// execute(output) to have been called first. Call before optimize() --
// OP_FUSED nodes (which only optimize() ever creates) aren't handled.
GradTape backward(Node* output);

// Linear scan; NULL if `node` has no entry.
Tensor* backward_grad(const GradTape* tape, const Node* node);

void backward_free(GradTape* tape);

#endif
