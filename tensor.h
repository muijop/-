// Tensor operations for AI training framework
// Provides basic tensor operations similar to PyTorch

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

// Data type enumeration
typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_BOOL,
    DTYPE_UINT8,
    DTYPE_INT8,
    DTYPE_FLOAT16,
    DTYPE_BFLOAT16
} DataType;

// Device type enumeration
typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_MPS,
    DEVICE_META
} DeviceType;

// Tensor structure definition
typedef struct {
    float* data;                // Raw data array
    size_t* shape;              // Shape dimensions
    size_t* strides;            // Strides for indexing
    size_t ndim;                // Number of dimensions
    size_t size;                // Total number of elements
    bool requires_grad;         // Whether gradient computation is required
    float* grad;                // Gradient data
    void* grad_fn;              // Gradient function for autograd
} Tensor;

// Tensor list for parameter management
typedef struct {
    Tensor** tensors;           // Array of tensor pointers
    size_t count;               // Number of tensors
    size_t capacity;            // Capacity of the array
} TensorList;

// Tensor creation functions
Tensor* tensor_create(const float* data, const size_t* shape, size_t ndim, bool requires_grad);
Tensor* tensor_zeros(const size_t* shape, size_t ndim, bool requires_grad);
Tensor* tensor_ones(const size_t* shape, size_t ndim, bool requires_grad);
Tensor* tensor_randn(const size_t* shape, size_t ndim, bool requires_grad);

// Tensor memory management
void tensor_free(Tensor* tensor);
void tensor_zero_grad(Tensor* tensor);

// Tensor operations
Tensor* tensor_view(Tensor* tensor, const size_t* new_shape, size_t new_ndim);
Tensor* tensor_transpose(Tensor* tensor, size_t dim0, size_t dim1);
Tensor* tensor_reshape(Tensor* tensor, const size_t* new_shape, size_t new_ndim);

// Autograd operations
void tensor_backward(Tensor* tensor);

// Utility functions
void tensor_print(const Tensor* tensor, const char* name);

// Mathematical operations
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_sum(Tensor* tensor, size_t dim, bool keepdim);
Tensor* tensor_mean(Tensor* tensor, size_t dim, bool keepdim);
Tensor* tensor_max(Tensor* tensor, size_t dim, bool keepdim);
Tensor* tensor_relu(Tensor* tensor);
Tensor* tensor_softmax(Tensor* tensor, size_t dim);
Tensor* tensor_log(Tensor* tensor);
Tensor* tensor_exp(Tensor* tensor);
Tensor* tensor_pow(Tensor* tensor, float exponent);

// Indexing operations
size_t tensor_get_index(const Tensor* tensor, const size_t* indices);
float tensor_get_item(const Tensor* tensor, const size_t* indices);
void tensor_set_item(Tensor* tensor, const size_t* indices, float value);

#endif