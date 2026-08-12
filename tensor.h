#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    float* data;
    int* shape;
    size_t* strides;
    int ndim;
    size_t size;
} Tensor;

Tensor* create_tensor(float* data, int* shape, int ndim);
void free_tensor(Tensor* tensor);

// `b` may either exactly match `a`'s shape, or be a trailing suffix of it
// (e.g. a->shape=[M,N], b->shape=[N]) -- the standard bias-broadcast case.
Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_mul(const Tensor* a, const Tensor* b);
Tensor* relu(const Tensor* x);
Tensor* matmul(const Tensor* a, const Tensor* b);
Tensor* transpose(const Tensor* x);
Tensor* permute(const Tensor* x, const int* order);
Tensor* softmax(const Tensor* x);

size_t tensor_offset(Tensor* tensor, int* indices);
void print_tensor(const Tensor* tensor);

#endif
