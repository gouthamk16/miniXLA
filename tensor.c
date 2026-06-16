// Minimal tensor library in C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"

Tensor* create_tensor(float* data, int* shape, int ndim) {
    if (ndim <= 0) {
        fprintf(stderr, "Invalid number of dimensions: %d\n", ndim);
        return NULL;
    }
    if (!shape) {
        fprintf(stderr, "Shape cannot be NULL\n");
        return NULL;
    }
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "Invalid shape dimension: %d\n", shape[i]);
            return NULL;
        }
    }
    Tensor* tensor = (Tensor*)calloc(1, sizeof(Tensor)); if (!tensor) {
        fprintf(stderr, "Failed to allocate memory for tensor\n");
        return NULL;
    }
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        fprintf(stderr, "Failed to allocate memory for tensor dimensions\n");
        free_tensor(tensor);
        return NULL;
    }

    size_t size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->shape[i] = shape[i];
        tensor->strides[i] = size;
        size *= shape[i];
    }
    tensor->size = size;
    tensor->data = (float*)malloc(size * sizeof(float));
    if (!tensor->data) {
        fprintf(stderr, "Failed to allocate memory for tensor data\n");
        free_tensor(tensor);
        return NULL;
    }
    if (data != NULL) {
        memcpy(tensor->data, data, size * sizeof(float));
    }
    return tensor;
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Tensors must have the same number of dimensions\n");
        return NULL;
    }
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Tensors must have the same shape\n");
            return NULL;
        }
    }
    Tensor* result = create_tensor(NULL, a->shape, a->ndim);
    if (!result) {
        fprintf(stderr, "Failed to create result tensor\n");
        return NULL;
    }
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

Tensor* relu(const Tensor* x) {
    Tensor* result = create_tensor(NULL, x->shape, x->ndim);
    if (!result) {
        fprintf(stderr, "Failed to create result tensor\n");
        return NULL;
    }
    for (size_t i = 0; i < x->size; i++) {
        result->data[i] = x->data[i] > 0 ? x->data[i] : 0;
    }
    return result;
}

Tensor* matmul(const Tensor* a, const Tensor* b) {
    // Multupliying batch dimensions in parallel?
    int M = a->shape[a->ndim-2];
    int K = a->shape[a->ndim-1];
    int N = b->shape[b->ndim-1];

    if (a->ndim != b->ndim || b->shape[b->ndim-2] != K) {
        fprintf(stderr, "Incompatible shapes for matmul: [%d,%d] x [%d,%d]\n",
                a->shape[a->ndim-2], a->shape[a->ndim-1],
                b->shape[b->ndim-2], b->shape[b->ndim-1]);
        return NULL;
    }

    size_t batch_size = a->size / (a->shape[a->ndim-2] * a->shape[a->ndim-1]);

    int* out_shape = (int*)malloc(a->ndim * sizeof(int));
    if (!out_shape) {
        fprintf(stderr, "Failed to allocate memory for output shape\n");
        return NULL;
    }
    memcpy(out_shape, a->shape, a->ndim * sizeof(int));
    out_shape[a->ndim-1] = N;
    Tensor* result = create_tensor(NULL, out_shape, a->ndim);
    free(out_shape);
    if (!result) {
        fprintf(stderr, "Failed to create tensor to store the result\n");
        return NULL;
    }
    memset(result->data, 0, result->size * sizeof(float));

    for (size_t bt = 0; bt < batch_size; bt++) {
        const float* A = a->data + bt * M * K;
        const float* B = b->data + bt * K * N;
        float* C = result->data + bt * M * N;
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float aik = A[i * K + k];
                for (int j = 0; j < N; j++) {
                    C[i * N + j] += aik * B[k * N + j];
                }
            }
        }
    }
    return result;
}

// Reorder axes by `order` (a permutation of 0..ndim-1), Output axis d maps to input axis order[d].
Tensor* permute(const Tensor* x, const int* order) {
    char* seen = (char*)calloc(x->ndim, 1);
    if (!seen) return NULL;
    for (int d = 0; d < x->ndim; d++) {
        if (order[d] < 0 || order[d] >= x->ndim || seen[order[d]]) {
            fprintf(stderr, "permute: order is not a valid permutation\n");
            free(seen);
            return NULL;
        }
        seen[order[d]] = 1;
    }
    free(seen);

    int* out_shape = (int*)malloc(x->ndim * sizeof(int));
    if (!out_shape) return NULL;
    for (int d = 0; d < x->ndim; d++) out_shape[d] = x->shape[order[d]];
    Tensor* result = create_tensor(NULL, out_shape, x->ndim);
    free(out_shape);
    if (!result) return NULL;

    int* idx = (int*)calloc(x->ndim, sizeof(int)); // current output multi-index
    if (!idx) { free_tensor(result); return NULL; }
    for (size_t n = 0; n < result->size; n++) {
        size_t src = 0;
        for (int d = 0; d < x->ndim; d++) src += idx[d] * x->strides[order[d]];
        result->data[n] = x->data[src];
        for (int d = x->ndim - 1; d >= 0; d--) {     
            if (++idx[d] < result->shape[d]) break;
            idx[d] = 0;
        }
    }
    free(idx);
    return result;
}

// Transpose: swap the last two axes
Tensor* transpose(const Tensor* x) {
    if (x->ndim < 2) {
        fprintf(stderr, "transpose: need rank >= 2\n");
        return NULL;
    }
    int* order = (int*)malloc(x->ndim * sizeof(int));
    if (!order) return NULL;
    for (int d = 0; d < x->ndim; d++) order[d] = d;
    order[x->ndim - 2] = x->ndim - 1;
    order[x->ndim - 1] = x->ndim - 2;
    Tensor* result = permute(x, order);
    free(order);
    return result;
}

// Softmax over the last axis.
Tensor* softmax(const Tensor* x) {
    Tensor* result = create_tensor(NULL, x->shape, x->ndim);
    if (!result) return NULL;

    int axis = x->shape[x->ndim - 1];
    size_t rows = x->size / (size_t)axis;
    for (size_t r = 0; r < rows; r++) {
        const float* in = x->data + r * axis;
        float* out = result->data + r * axis;

        float max = in[0];
        for (int j = 1; j < axis; j++) if (in[j] > max) max = in[j];

        float sum = 0;
        for (int j = 0; j < axis; j++) {
            out[j] = expf(in[j] - max);
            sum += out[j];
        }
        for (int j = 0; j < axis; j++) out[j] /= sum;
    }
    return result;
}

size_t tensor_offset(Tensor* tensor, int* indices) {
    size_t offset = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
            fprintf(stderr, "Index out of bounds: %d\n", indices[i]);
            return (size_t)-1;
        }
        offset += indices[i] * tensor->strides[i];
    }
    return offset;
}


static void print_dim(const Tensor* t, int d, size_t offset) {
    putchar('[');
    for (int i = 0; i < t->shape[d]; i++) {
        if (d == t->ndim - 1) {
            printf("%g", t->data[offset + i * t->strides[d]]);
        } else {
            print_dim(t, d + 1, offset + i * t->strides[d]);
        }
        if (i < t->shape[d] - 1) printf(d == t->ndim - 1 ? ", " : ",\n");
    }
    putchar(']');
}

void print_tensor(const Tensor* tensor) {
    if (!tensor) {
        fprintf(stderr, "Tensor is NULL\n");
        return;
    }
    printf("shape=[");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf("]\n");
    print_dim(tensor, 0, 0);
    putchar('\n');
}

void free_tensor(Tensor* tensor) {
    if (!tensor) return;
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}