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

Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* relu(const Tensor* x);
Tensor* matmul(const Tensor* a, const Tensor* b);
Tensor* transpose(const Tensor* x);
Tensor* permute(const Tensor* x, const int* order);
Tensor* softmax(const Tensor* x);

size_t tensor_offset(Tensor* tensor, int* indices);
void print_tensor(const Tensor* tensor);

#endif
