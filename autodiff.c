// Reverse-mode autodiff over an already-executed graph. Computes gradient
// Tensors directly rather than building a second, differentiable graph --
// see the Phase 6 design doc for why that generality isn't needed here.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "autodiff.h"

// Add `contrib` into node `n`'s accumulated gradient, creating the entry on
// first contribution. Takes ownership of `contrib`.
static void tape_accumulate(GradTape* tape, const Node* n, Tensor* contrib) {
    if (!contrib) return;
    for (int i = 0; i < tape->count; i++) {
        if (tape->entries[i].node == n) {
            Tensor* sum = tensor_add(tape->entries[i].grad, contrib);
            free_tensor(contrib);
            free_tensor(tape->entries[i].grad);
            tape->entries[i].grad = sum;
            return;
        }
    }
    tape->entries = (GradEntry*)realloc(tape->entries, (tape->count + 1) * sizeof(GradEntry));
    tape->entries[tape->count].node = n;
    tape->entries[tape->count].grad = contrib;
    tape->count++;
}

static Tensor* ones_like(const Tensor* t) {
    Tensor* r = create_tensor(NULL, t->shape, t->ndim);
    if (!r) return NULL;
    for (size_t i = 0; i < r->size; i++) r->data[i] = 1.0f;
    return r;
}

static Tensor* relu_backward(const Tensor* gy, const Tensor* x) {
    Tensor* r = create_tensor(NULL, x->shape, x->ndim);
    if (!r) return NULL;
    for (size_t i = 0; i < r->size; i++) r->data[i] = x->data[i] > 0 ? gy->data[i] : 0.0f;
    return r;
}

// Undo tensor_add's broadcast: sum gy down to `shape`. `shape` is either a
// trailing suffix of gy's shape (the broadcast case) or exactly gy's shape,
// in which case every gy element maps to a distinct output slot and this is
// a plain copy -- same one-pass trick tensor_add itself uses.
static Tensor* sum_to_shape(const Tensor* gy, const int* shape, int ndim) {
    Tensor* r = create_tensor(NULL, (int*)shape, ndim);
    if (!r) return NULL;
    memset(r->data, 0, r->size * sizeof(float));
    for (size_t i = 0; i < gy->size; i++) r->data[i % r->size] += gy->data[i];
    return r;
}

// dx_j = y_j * (gy_j - sum_k gy_k*y_k), applied per row.
static Tensor* softmax_backward(const Tensor* gy, const Tensor* y) {
    Tensor* r = create_tensor(NULL, y->shape, y->ndim);
    if (!r) return NULL;
    int axis = y->shape[y->ndim - 1];
    size_t rows = y->size / (size_t)axis;
    for (size_t row = 0; row < rows; row++) {
        const float* yr  = y->data  + row * (size_t)axis;
        const float* gyr = gy->data + row * (size_t)axis;
        float* out = r->data + row * (size_t)axis;
        float dot = 0;
        for (int j = 0; j < axis; j++) dot += gyr[j] * yr[j];
        for (int j = 0; j < axis; j++) out[j] = yr[j] * (gyr[j] - dot);
    }
    return r;
}

GradTape backward(Node* output) {
    GradTape tape = {0};
    if (!output->output) {
        fprintf(stderr, "backward: execute(output) must be called first\n");
        return tape;
    }

    // graph_topo_order puts every node after its own inputs; walking it in
    // reverse instead puts every node after all of its *consumers* -- what
    // backward propagation needs, since a node's gradient must be fully
    // accumulated before it pushes gradient further to its own inputs.
    Node** order;
    int n = graph_topo_order(output, &order);

    tape_accumulate(&tape, output, ones_like(output->output));

    for (int i = n - 1; i >= 0; i--) {
        Node* x = order[i];
        Tensor* gy = backward_grad(&tape, x);
        if (!gy) continue;

        switch (x->op) {
            case OP_MATMUL: {
                Tensor* A = x->inputs[0]->output;
                Tensor* B = x->inputs[1]->output;
                Tensor* Bt = transpose(B);
                Tensor* gA = matmul(gy, Bt);
                free_tensor(Bt);
                Tensor* At = transpose(A);
                Tensor* gB = matmul(At, gy);
                free_tensor(At);
                tape_accumulate(&tape, x->inputs[0], gA);
                tape_accumulate(&tape, x->inputs[1], gB);
                break;
            }
            case OP_ADD: {
                Tensor* oa = x->inputs[0]->output;
                Tensor* ob = x->inputs[1]->output;
                tape_accumulate(&tape, x->inputs[0], sum_to_shape(gy, oa->shape, oa->ndim));
                tape_accumulate(&tape, x->inputs[1], sum_to_shape(gy, ob->shape, ob->ndim));
                break;
            }
            case OP_MUL: {
                Tensor* ga = tensor_mul(gy, x->inputs[1]->output);
                Tensor* gb = tensor_mul(gy, x->inputs[0]->output);
                tape_accumulate(&tape, x->inputs[0], ga);
                tape_accumulate(&tape, x->inputs[1], gb);
                break;
            }
            case OP_RELU:
                tape_accumulate(&tape, x->inputs[0], relu_backward(gy, x->inputs[0]->output));
                break;
            case OP_TRANSPOSE:
                tape_accumulate(&tape, x->inputs[0], transpose(gy));
                break;
            case OP_SOFTMAX:
                tape_accumulate(&tape, x->inputs[0], softmax_backward(gy, x->output));
                break;
            case OP_INPUT:
                break;   // leaf: nothing further to propagate
            case OP_FUSED:
                fprintf(stderr, "backward: OP_FUSED not supported -- call backward() before optimize()\n");
                break;
        }
    }

    free(order);
    return tape;
}

Tensor* backward_grad(const GradTape* tape, const Node* node) {
    for (int i = 0; i < tape->count; i++)
        if (tape->entries[i].node == node) return tape->entries[i].grad;
    return NULL;
}

void backward_free(GradTape* tape) {
    for (int i = 0; i < tape->count; i++) free_tensor(tape->entries[i].grad);
    free(tape->entries);
    tape->entries = NULL;
    tape->count = 0;
}
