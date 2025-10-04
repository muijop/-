#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static size_t calculate_size(const size_t* shape, size_t ndim) {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

static void calculate_strides(size_t* strides, const size_t* shape, size_t ndim) {
    size_t stride = 1;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

Tensor* tensor_create(const float* data, const size_t* shape, size_t ndim, bool requires_grad) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    
    tensor->ndim = ndim;
    tensor->size = calculate_size(shape, ndim);
    tensor->requires_grad = requires_grad;
    
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    
    if (!tensor->shape || !tensor->strides || !tensor->data) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor->data);
        free(tensor);
        return NULL;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    calculate_strides(tensor->strides, shape, ndim);
    
    if (data) {
        memcpy(tensor->data, data, tensor->size * sizeof(float));
    } else {
        memset(tensor->data, 0, tensor->size * sizeof(float));
    }
    
    tensor->grad = NULL;
    tensor->grad_fn = NULL;
    
    if (requires_grad) {
        tensor->grad = (float*)calloc(tensor->size, sizeof(float));
    }
    
    return tensor;
}

Tensor* tensor_zeros(const size_t* shape, size_t ndim, bool requires_grad) {
    return tensor_create(NULL, shape, ndim, requires_grad);
}

Tensor* tensor_ones(const size_t* shape, size_t ndim, bool requires_grad) {
    Tensor* tensor = tensor_create(NULL, shape, ndim, requires_grad);
    if (tensor) {
        for (size_t i = 0; i < tensor->size; i++) {
            tensor->data[i] = 1.0f;
        }
    }
    return tensor;
}

Tensor* tensor_randn(const size_t* shape, size_t ndim, bool requires_grad) {
    Tensor* tensor = tensor_create(NULL, shape, ndim, requires_grad);
    if (tensor) {
        srand(time(NULL));
        for (size_t i = 0; i < tensor->size; i++) {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
            tensor->data[i] = z0;
        }
    }
    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor->strides);
        free(tensor->grad);
        free(tensor);
    }
}

void tensor_zero_grad(Tensor* tensor) {
    if (tensor && tensor->grad) {
        memset(tensor->grad, 0, tensor->size * sizeof(float));
    }
}

size_t tensor_get_index(const Tensor* tensor, const size_t* indices) {
    size_t index = 0;
    for (size_t i = 0; i < tensor->ndim; i++) {
        index += indices[i] * tensor->strides[i];
    }
    return index;
}

float tensor_get_item(const Tensor* tensor, const size_t* indices) {
    size_t index = tensor_get_index(tensor, indices);
    return tensor->data[index];
}

void tensor_set_item(Tensor* tensor, const size_t* indices, float value) {
    size_t index = tensor_get_index(tensor, indices);
    tensor->data[index] = value;
}

void tensor_print(const Tensor* tensor, const char* name) {
    printf("Tensor %s: shape=[", name ? name : "tensor");
    for (size_t i = 0; i < tensor->ndim; i++) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf("], data=\n");
    
    if (tensor->ndim == 1) {
        printf("[");
        for (size_t i = 0; i < tensor->shape[0]; i++) {
            printf("%.4f", tensor->data[i]);
            if (i < tensor->shape[0] - 1) printf(", ");
        }
        printf("]\n");
    } else if (tensor->ndim == 2) {
        printf("[");
        for (size_t i = 0; i < tensor->shape[0]; i++) {
            printf("[");
            for (size_t j = 0; j < tensor->shape[1]; j++) {
                printf("%.4f", tensor->data[i * tensor->strides[0] + j * tensor->strides[1]]);
                if (j < tensor->shape[1] - 1) printf(", ");
            }
            printf("]");
            if (i < tensor->shape[0] - 1) printf(",\n ");
        }
        printf("]\n");
    } else {
        printf("[... multi-dimensional tensor ...]\n");
    }
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->size != b->size) return NULL;
    
    size_t* shape = (size_t*)malloc(a->ndim * sizeof(size_t));
    memcpy(shape, a->shape, a->ndim * sizeof(size_t));
    
    Tensor* result = tensor_create(NULL, shape, a->ndim, a->requires_grad || b->requires_grad);
    free(shape);
    
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    
    return result;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (a->size != b->size) return NULL;
    
    size_t* shape = (size_t*)malloc(a->ndim * sizeof(size_t));
    memcpy(shape, a->shape, a->ndim * sizeof(size_t));
    
    Tensor* result = tensor_create(NULL, shape, a->ndim, a->requires_grad || b->requires_grad);
    free(shape);
    
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    
    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    
    size_t out_shape[2] = {a->shape[0], b->shape[1]};
    Tensor* result = tensor_zeros(out_shape, 2, a->requires_grad || b->requires_grad);
    
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->shape[0]; i++) {
        for (size_t j = 0; j < b->shape[1]; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->shape[1]; k++) {
                sum += a->data[i * a->strides[0] + k * a->strides[1]] * 
                       b->data[k * b->strides[0] + j * b->strides[1]];
            }
            result->data[i * result->strides[0] + j * result->strides[1]] = sum;
        }
    }
    
    return result;
}

Tensor* tensor_relu(Tensor* tensor) {
    Tensor* result = tensor_create(NULL, tensor->shape, tensor->ndim, tensor->requires_grad);
    if (!result) return NULL;
    
    for (size_t i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i] > 0 ? tensor->data[i] : 0;
    }
    
    return result;
}

Tensor* tensor_softmax(Tensor* tensor, size_t dim) {
    if (dim >= tensor->ndim) return NULL;
    
    Tensor* result = tensor_create(NULL, tensor->shape, tensor->ndim, tensor->requires_grad);
    if (!result) return NULL;
    
    memcpy(result->data, tensor->data, tensor->size * sizeof(float));
    
    if (tensor->ndim == 1) {
        float max_val = tensor->data[0];
        for (size_t i = 1; i < tensor->size; i++) {
            if (tensor->data[i] > max_val) max_val = tensor->data[i];
        }
        
        float sum = 0.0f;
        for (size_t i = 0; i < tensor->size; i++) {
            result->data[i] = exp(tensor->data[i] - max_val);
            sum += result->data[i];
        }
        
        for (size_t i = 0; i < tensor->size; i++) {
            result->data[i] /= sum;
        }
    }
    
    return result;
}

void tensor_backward(Tensor* tensor) {
    if (!tensor->requires_grad || !tensor->grad) return;
    
    for (size_t i = 0; i < tensor->size; i++) {
        tensor->grad[i] = 1.0f;
    }
}