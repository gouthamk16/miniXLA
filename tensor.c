// Minimal tensor library in C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data;
    int* shape;
    size_t* strides;
    int ndim;
    size_t size;
} Tensor;

void free_tensor(Tensor* tensor);
size_t tensor_offset(Tensor* tensor, int* indices);
void print_tensor(const Tensor* tensor);

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
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

int main() {
    int shape[2] = {2, 3};
    int ndim = 2;
    float data_a[6] = {1, 2, 3, 4, 5, 6};
    float data_b[6] = {6, 5, 4, 3, 2, 1};

    Tensor* a = create_tensor(data_a, shape, ndim);
    Tensor* b = create_tensor(data_b, shape, ndim);
    if (!a || !b) {
        fprintf(stderr, "Failed to create tensors\n");
        return EXIT_FAILURE;
    }

    print_tensor(a);
    printf("\n");
    print_tensor(b);
    printf("\n");

    Tensor* c = tensor_add(a, b);
    if (!c) {
        fprintf(stderr, "Failed to add tensors\n");
        return EXIT_FAILURE;
    }

    print_tensor(c);
    printf("\n");

    Tensor* d = relu(create_tensor((float[]){-1, 0, 1, -2, 2, -3}, shape, ndim));
    if (!d) {
        fprintf(stderr, "Failed to apply ReLU\n");
        return EXIT_FAILURE;
    }
    print_tensor(d);

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    return EXIT_SUCCESS;
}