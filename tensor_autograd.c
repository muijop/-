#include "tensor_autograd.h"
#include "kernels.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// 全局自动微分状态
static bool g_grad_enabled = true;
static AutogradEngine* g_default_engine = NULL;

// 内部辅助函数
static AutogradTensor* create_autograd_tensor_internal(Tensor* tensor, bool requires_grad, bool is_leaf);
static void ensure_grad_node(AutogradTensor* atensor);
static void backward_impl(AutogradTensor* atensor, Tensor* grad_output);

// 初始化默认自动微分引擎
static void init_default_engine(void) {
    if (g_default_engine == NULL) {
        g_default_engine = autograd_engine_create();
    }
}

// 创建支持自动微分的张量
AutogradTensor* autograd_tensor_create(const int* shape, int ndim, DataType dtype, bool requires_grad) {
    // 使用tensor_zeros创建张量
    Tensor* tensor = tensor_zeros((const size_t*)shape, ndim, requires_grad);
    return create_autograd_tensor_internal(tensor, requires_grad, true);
}

AutogradTensor* autograd_tensor_create_from_data(const int* shape, int ndim, DataType dtype, void* data, bool requires_grad) {
    // 使用tensor_create创建张量
    Tensor* tensor = tensor_create((const float*)data, (const size_t*)shape, ndim, requires_grad);
    return create_autograd_tensor_internal(tensor, requires_grad, true);
}

AutogradTensor* autograd_tensor_create_from_tensor(Tensor* tensor, bool requires_grad) {
    // 复制张量以避免共享内存
    Tensor* tensor_copy = tensor_create((const float*)tensor->data, tensor->shape, tensor->ndim, requires_grad);
    return create_autograd_tensor_internal(tensor_copy, requires_grad, true);
}

// 内部创建函数
static AutogradTensor* create_autograd_tensor_internal(Tensor* tensor, bool requires_grad, bool is_leaf) {
    AutogradTensor* atensor = (AutogradTensor*)malloc(sizeof(AutogradTensor));
    if (!atensor) {
        return NULL;
    }
    
    atensor->tensor = tensor;
    atensor->grad_node = NULL;
    atensor->requires_grad = requires_grad && g_grad_enabled;
    atensor->is_leaf = is_leaf;
    atensor->engine = g_default_engine;
    
    if (atensor->requires_grad) {
        ensure_grad_node(atensor);
    }
    
    return atensor;
}

// 确保梯度节点存在
static void ensure_grad_node(AutogradTensor* atensor) {
    if (atensor->grad_node == NULL && atensor->requires_grad) {
        atensor->grad_node = autograd_create_node(atensor->engine, atensor->tensor, OP_INPUT);
    }
}

// 销毁自动微分张量
void autograd_tensor_destroy(AutogradTensor* atensor) {
    if (atensor) {
        if (atensor->tensor) {
            tensor_destroy(atensor->tensor);
        }
        if (atensor->grad_node) {
            autograd_node_destroy(atensor->grad_node);
        }
        free(atensor);
    }
}

// 获取张量数据
void* autograd_tensor_data(AutogradTensor* atensor) {
    return atensor ? tensor_data(atensor->tensor) : NULL;
}

float* autograd_tensor_data_float(AutogradTensor* atensor) {
    return atensor ? tensor_data_float(atensor->tensor) : NULL;
}

int* autograd_tensor_data_int(AutogradTensor* atensor) {
    return atensor ? tensor_data_int(atensor->tensor) : NULL;
}

// 获取梯度
Tensor* autograd_tensor_grad(AutogradTensor* atensor) {
    if (!atensor || !atensor->grad_node || !atensor->grad_node->grads) {
        return NULL;
    }
    return atensor->grad_node->grads[0];
}

void* autograd_tensor_grad_data(AutogradTensor* atensor) {
    Tensor* grad = autograd_tensor_grad(atensor);
    return grad ? tensor_data(grad) : NULL;
}

// 基本操作（支持自动微分）
AutogradTensor* autograd_tensor_add(AutogradTensor* a, AutogradTensor* b) {
    if (!a || !b) return NULL;
    
    // 执行张量加法
    Tensor* result_tensor = tensor_add(a->tensor, b->tensor);
    if (!result_tensor) return NULL;
    
    // 创建结果张量
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, 
                                                             a->requires_grad || b->requires_grad, false);
    
    // 如果不需要梯度，直接返回
    if (!result->requires_grad) {
        return result;
    }
    
    // 创建计算图节点
    GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_ADD);
    autograd_add_input(node, a->grad_node);
    autograd_add_input(node, b->grad_node);
    result->grad_node = node;
    
    return result;
}

AutogradTensor* autograd_tensor_subtract(AutogradTensor* a, AutogradTensor* b) {
    if (!a || !b) return NULL;
    
    Tensor* result_tensor = tensor_subtract(a->tensor, b->tensor);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor,
                                                             a->requires_grad || b->requires_grad, false);
    
    if (!result->requires_grad) {
        return result;
    }
    
    GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_SUBTRACT);
    autograd_add_input(node, a->grad_node);
    autograd_add_input(node, b->grad_node);
    result->grad_node = node;
    
    return result;
}

AutogradTensor* autograd_tensor_multiply(AutogradTensor* a, AutogradTensor* b) {
    if (!a || !b) return NULL;
    
    Tensor* result_tensor = tensor_multiply(a->tensor, b->tensor);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor,
                                                             a->requires_grad || b->requires_grad, false);
    
    if (!result->requires_grad) {
        return result;
    }
    
    GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_MULTIPLY);
    autograd_add_input(node, a->grad_node);
    autograd_add_input(node, b->grad_node);
    result->grad_node = node;
    
    return result;
}

AutogradTensor* autograd_tensor_divide(AutogradTensor* a, AutogradTensor* b) {
    if (!a || !b) return NULL;
    
    Tensor* result_tensor = tensor_divide(a->tensor, b->tensor);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor,
                                                             a->requires_grad || b->requires_grad, false);
    
    if (!result->requires_grad) {
        return result;
    }
    
    GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_DIVIDE);
    autograd_add_input(node, a->grad_node);
    autograd_add_input(node, b->grad_node);
    result->grad_node = node;
    
    return result;
}

AutogradTensor* autograd_tensor_matmul(AutogradTensor* a, AutogradTensor* b) {
    if (!a || !b) return NULL;
    
    Tensor* result_tensor = tensor_matmul(a->tensor, b->tensor);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor,
                                                             a->requires_grad || b->requires_grad, false);
    
    if (!result->requires_grad) {
        return result;
    }
    
    GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_MATMUL);
    autograd_add_input(node, a->grad_node);
    autograd_add_input(node, b->grad_node);
    result->grad_node = node;
    
    return result;
}

// 标量操作
AutogradTensor* autograd_tensor_add_scalar(AutogradTensor* a, float scalar) {
    if (!a) return NULL;
    
    // 创建标量张量
    int scalar_shape[] = {1};
    float scalar_data = scalar;
    Tensor* scalar_tensor = tensor_create(&scalar_data, (const size_t*)scalar_shape, 1, false);
    
    AutogradTensor* scalar_atensor = create_autograd_tensor_internal(scalar_tensor, false, true);
    AutogradTensor* result = autograd_tensor_add(a, scalar_atensor);
    
    autograd_tensor_destroy(scalar_atensor);
    return result;
}

AutogradTensor* autograd_tensor_multiply_scalar(AutogradTensor* a, float scalar) {
    if (!a) return NULL;
    
    int scalar_shape[] = {1};
    float scalar_data = scalar;
    Tensor* scalar_tensor = tensor_create(&scalar_data, (const size_t*)scalar_shape, 1, false);
    
    AutogradTensor* scalar_atensor = create_autograd_tensor_internal(scalar_tensor, false, true);
    AutogradTensor* result = autograd_tensor_multiply(a, scalar_atensor);
    
    autograd_tensor_destroy(scalar_atensor);
    return result;
}

AutogradTensor* autograd_tensor_divide_scalar(AutogradTensor* a, float scalar) {
    if (!a) return NULL;
    
    int scalar_shape[] = {1};
    float scalar_data = scalar;
    Tensor* scalar_tensor = tensor_create(&scalar_data, (const size_t*)scalar_shape, 1, false);
    
    AutogradTensor* scalar_atensor = create_autograd_tensor_internal(scalar_tensor, false, true);
    AutogradTensor* result = autograd_tensor_divide(a, scalar_atensor);
    
    autograd_tensor_destroy(scalar_atensor);
    return result;
}

// 形状操作
AutogradTensor* autograd_tensor_reshape(AutogradTensor* a, const int* new_shape, int new_ndim) {
    if (!a) return NULL;
    
    Tensor* result_tensor = tensor_reshape(a->tensor, new_shape, new_ndim);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, a->requires_grad, false);
    
    if (result->requires_grad) {
        GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_RESHAPE);
        autograd_add_input(node, a->grad_node);
        result->grad_node = node;
    }
    
    return result;
}

AutogradTensor* autograd_tensor_transpose(AutogradTensor* a, int dim1, int dim2) {
    if (!a) return NULL;
    
    Tensor* result_tensor = tensor_transpose(a->tensor, dim1, dim2);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, a->requires_grad, false);
    
    if (result->requires_grad) {
        GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_TRANSPOSE);
        autograd_add_input(node, a->grad_node);
        result->grad_node = node;
    }
    
    return result;
}

// 聚合操作
AutogradTensor* autograd_tensor_sum(AutogradTensor* a, int dim, bool keepdim) {
    if (!a) return NULL;
    
    Tensor* result_tensor = tensor_sum(a->tensor, dim, keepdim);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, a->requires_grad, false);
    
    if (result->requires_grad) {
        GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_SUM);
        autograd_add_input(node, a->grad_node);
        result->grad_node = node;
    }
    
    return result;
}

// 兼容版本 - 返回Tensor*而不是AutogradTensor*
Tensor* autograd_tensor_sum_compat(Tensor* tensor, int dim, bool keepdim, AutogradEngine* engine) {
    if (!tensor) return NULL;
    
    Tensor* result_tensor = tensor_sum(tensor, dim, keepdim);
    return result_tensor;
}

AutogradTensor* autograd_tensor_mean(AutogradTensor* a, int dim, bool keepdim) {
    if (!a) return NULL;
    
    Tensor* result_tensor = tensor_mean(a->tensor, dim, keepdim);
    if (!result_tensor) return NULL;
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, a->requires_grad, false);
    
    if (result->requires_grad) {
        GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_MEAN);
        autograd_add_input(node, a->grad_node);
        result->grad_node = node;
    }
    
    return result;
}

// 兼容版本 - 返回Tensor*而不是AutogradTensor*
Tensor* autograd_tensor_mean_compat(Tensor* tensor, int dim, bool keepdim, AutogradEngine* engine) {
    if (!tensor) return NULL;
    
    Tensor* result_tensor = tensor_mean(tensor, dim, keepdim);
    return result_tensor;
}

// 激活函数
AutogradTensor* autograd_tensor_relu(AutogradTensor* a) {
    if (!a) return NULL;
    
    // 使用内核优化
    Tensor* result_tensor = tensor_create_like(a->tensor);
    float* input_data = tensor_data_float(a->tensor);
    float* output_data = tensor_data_float(result_tensor);
    size_t size = tensor_size(a->tensor);
    
    kernel_relu(size, input_data, output_data);
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, a->requires_grad, false);
    
    if (result->requires_grad) {
        GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_RELU);
        autograd_add_input(node, a->grad_node);
        result->grad_node = node;
    }
    
    return result;
}

// 兼容版本 - 返回Tensor*而不是AutogradTensor*
Tensor* autograd_tensor_relu_compat(Tensor* tensor, AutogradEngine* engine) {
    if (!tensor) return NULL;
    
    Tensor* result_tensor = tensor_create_like(tensor);
    float* input_data = tensor_data_float(tensor);
    float* output_data = tensor_data_float(result_tensor);
    size_t size = tensor_size(tensor);
    
    kernel_relu(size, input_data, output_data);
    return result_tensor;
}

AutogradTensor* autograd_tensor_sigmoid(AutogradTensor* a) {
    if (!a) return NULL;
    
    Tensor* result_tensor = tensor_create_like(a->tensor);
    float* input_data = tensor_data_float(a->tensor);
    float* output_data = tensor_data_float(result_tensor);
    size_t size = tensor_size(a->tensor);
    
    kernel_sigmoid(size, input_data, output_data);
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, a->requires_grad, false);
    
    if (result->requires_grad) {
        GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_SIGMOID);
        autograd_add_input(node, a->grad_node);
        result->grad_node = node;
    }
    
    return result;
}

// 兼容版本 - 返回Tensor*而不是AutogradTensor*
Tensor* autograd_tensor_sigmoid_compat(Tensor* tensor, AutogradEngine* engine) {
    if (!tensor) return NULL;
    
    Tensor* result_tensor = tensor_create_like(tensor);
    float* input_data = tensor_data_float(tensor);
    float* output_data = tensor_data_float(result_tensor);
    size_t size = tensor_size(tensor);
    
    kernel_sigmoid(size, input_data, output_data);
    return result_tensor;
}

AutogradTensor* autograd_tensor_tanh(AutogradTensor* a) {
    if (!a) return NULL;
    
    Tensor* result_tensor = tensor_create_like(a->tensor);
    float* input_data = tensor_data_float(a->tensor);
    float* output_data = tensor_data_float(result_tensor);
    size_t size = tensor_size(a->tensor);
    
    kernel_tanh(size, input_data, output_data);
    
    AutogradTensor* result = create_autograd_tensor_internal(result_tensor, a->requires_grad, false);
    
    if (result->requires_grad) {
        GraphNode* node = autograd_create_node(g_default_engine, result_tensor, OP_TANH);
        autograd_add_input(node, a->grad_node);
        result->grad_node = node;
    }
    
    return result;
}

// 兼容版本 - 返回Tensor*而不是AutogradTensor*
Tensor* autograd_tensor_tanh_compat(Tensor* tensor, AutogradEngine* engine) {
    if (!tensor) return NULL;
    
    Tensor* result_tensor = tensor_create_like(tensor);
    float* input_data = tensor_data_float(tensor);
    float* output_data = tensor_data_float(result_tensor);
    size_t size = tensor_size(tensor);
    
    kernel_tanh(size, input_data, output_data);
    return result_tensor;
}

// 反向传播
void autograd_tensor_backward(AutogradTensor* atensor, Tensor* grad_output) {
    if (!atensor || !atensor->requires_grad) {
        return;
    }
    
    // 如果还没有梯度，创建梯度张量
    if (!atensor->grad_node->grads || !atensor->grad_node->grads[0]) {
        atensor->grad_node->grads = malloc(sizeof(Tensor*));
        atensor->grad_node->grads[0] = tensor_create_like(atensor->tensor);
        tensor_fill(atensor->grad_node->grads[0], 0.0f);
    }
    
    // 累加梯度
    tensor_add(atensor->grad_node->grads[0], grad_output);
    
    // 执行反向传播
    autograd_backward(atensor->engine, grad_output);
}

void autograd_tensor_zero_grad(AutogradTensor* atensor) {
    if (!atensor || !atensor->grad_node) return;
    
    if (atensor->grad_node->grad) {
        tensor_fill(atensor->grad_node->grad, 0.0f);
    }
}

// 内部反向传播实现
static void backward_impl(AutogradTensor* atensor, Tensor* grad_output) {
    if (!atensor || !atensor->grad_node || !atensor->requires_grad) {
        return;
    }
    
    // 如果还没有梯度，创建梯度张量
    if (!atensor->grad_node->grads || !atensor->grad_node->grads[0]) {
        atensor->grad_node->grads = malloc(sizeof(Tensor*));
        atensor->grad_node->grads[0] = tensor_create_like(atensor->tensor);
        tensor_fill(atensor->grad_node->grads[0], 0.0f);
    }
    
    // 累加梯度
    tensor_add(atensor->grad_node->grads[0], grad_output);
    
    // 递归反向传播
    autograd_backward(atensor->engine, grad_output);
}

// 计算图操作
ComputationalGraph* autograd_tensor_graph(AutogradTensor* atensor) {
    return atensor && atensor->engine ? atensor->engine->graph : NULL;
}

void autograd_tensor_detach(AutogradTensor* atensor) {
    if (atensor) {
        atensor->requires_grad = false;
        if (atensor->grad_node) {
            autograd_node_destroy(atensor->grad_node);
            atensor->grad_node = NULL;
        }
    }
}

AutogradTensor* autograd_tensor_detach_copy(AutogradTensor* atensor) {
    if (!atensor) return NULL;
    
    Tensor* tensor_copy = tensor_copy(atensor->tensor);
    return create_autograd_tensor_internal(tensor_copy, false, true);
}

// 张量信息
const int* autograd_tensor_shape(AutogradTensor* atensor) {
    return atensor ? tensor_shape(atensor->tensor) : NULL;
}

int autograd_tensor_ndim(AutogradTensor* atensor) {
    return atensor ? tensor_ndim(atensor->tensor) : 0;
}

int autograd_tensor_size(AutogradTensor* atensor) {
    return atensor ? tensor_size(atensor->tensor) : 0;
}

DataType autograd_tensor_dtype(AutogradTensor* atensor) {
    return atensor ? tensor_dtype(atensor->tensor) : DTYPE_FLOAT32;
}

DeviceType autograd_tensor_device(AutogradTensor* atensor) {
    return atensor ? tensor_device(atensor->tensor) : DEVICE_CPU;
}

// 张量创建辅助函数
AutogradTensor* autograd_zeros(const int* shape, int ndim, DataType dtype, bool requires_grad) {
    Tensor* tensor = tensor_zeros(shape, ndim, dtype);
    return create_autograd_tensor_internal(tensor, requires_grad, true);
}

AutogradTensor* autograd_ones(const int* shape, int ndim, DataType dtype, bool requires_grad) {
    Tensor* tensor = tensor_ones(shape, ndim, dtype);
    return create_autograd_tensor_internal(tensor, requires_grad, true);
}

AutogradTensor* autograd_randn(const int* shape, int ndim, bool requires_grad) {
    Tensor* tensor = tensor_randn(shape, ndim);
    return create_autograd_tensor_internal(tensor, requires_grad, true);
}

AutogradTensor* autograd_uniform(const int* shape, int ndim, float min_val, float max_val, bool requires_grad) {
    Tensor* tensor = tensor_uniform(shape, ndim, min_val, max_val);
    return create_autograd_tensor_internal(tensor, requires_grad, true);
}

AutogradTensor* autograd_arange(float start, float stop, float step, bool requires_grad) {
    Tensor* tensor = tensor_arange(start, stop, step);
    return create_autograd_tensor_internal(tensor, requires_grad, true);
}

// 张量操作辅助函数
bool autograd_tensor_equal(AutogradTensor* a, AutogradTensor* b) {
    if (!a || !b) return false;
    return tensor_equal(a->tensor, b->tensor);
}

void autograd_tensor_print(AutogradTensor* atensor, const char* name) {
    if (!atensor) return;
    
    printf("AutogradTensor %s:\n", name ? name : "unnamed");
    printf("  shape: [");
    const int* shape = autograd_tensor_shape(atensor);
    int ndim = autograd_tensor_ndim(atensor);
    for (int i = 0; i < ndim; i++) {
        printf("%d", shape[i]);
        if (i < ndim - 1) printf(", ");
    }
    printf("]\n");
    printf("  dtype: %s\n", tensor_dtype_to_string(autograd_tensor_dtype(atensor)));
    printf("  device: %s\n", tensor_device_to_string(autograd_tensor_device(atensor)));
    printf("  requires_grad: %s\n", atensor->requires_grad ? "true" : "false");
    printf("  is_leaf: %s\n", atensor->is_leaf ? "true" : "false");
    
    if (tensor_size(atensor->tensor) <= 16) {
        printf("  data: ");
        tensor_print_data(atensor->tensor);
    }
    
    if (atensor->grad_node && atensor->grad_node->grad) {
        printf("  grad: ");
        tensor_print_data(atensor->grad_node->grad);
    }
}

void autograd_tensor_print_shape(AutogradTensor* atensor, const char* name) {
    if (!atensor) return;
    
    printf("%s: ", name ? name : "Tensor");
    const int* shape = autograd_tensor_shape(atensor);
    int ndim = autograd_tensor_ndim(atensor);
    printf("[");
    for (int i = 0; i < ndim; i++) {
        printf("%d", shape[i]);
        if (i < ndim - 1) printf(", ");
    }
    printf("]\n");
}

// 梯度检查
bool autograd_tensor_check_grad(AutogradTensor* atensor) {
    if (!atensor || !atensor->grad_node || !atensor->grad_node->grads || !atensor->grad_node->grads[0]) {
        return false;
    }
    
    return tensor_check_finite(atensor->grad_node->grads[0]);
}

void autograd_tensor_print_grad(AutogradTensor* atensor, const char* name) {
    if (!atensor || !atensor->grad_node || !atensor->grad_node->grad) {
        printf("%s: No gradient\n", name ? name : "Tensor");
        return;
    }
    
    printf("%s gradient:\n", name ? name : "Tensor");
    tensor_print_data(atensor->grad_node->grad);
}

// 内存管理
void autograd_tensor_set_grad_enabled(bool enabled) {
    g_grad_enabled = enabled;
}

bool autograd_tensor_is_grad_enabled(void) {
    return g_grad_enabled;
}

void autograd_tensor_set_grad(AutogradTensor* atensor, Tensor* grad) {
    if (!atensor || !atensor->grad_node) return;
    
    if (atensor->grad_node->grad) {
        tensor_destroy(atensor->grad_node->grad);
    }
    atensor->grad_node->grad = tensor_copy(grad);
}

// 上下文管理
void autograd_no_grad(void) {
    g_grad_enabled = false;
}

void autograd_enable_grad(void) {
    g_grad_enabled = true;
}

bool autograd_is_grad_enabled(void) {
    return g_grad_enabled;
}

// 性能优化
void autograd_tensor_use_kernel_optimization(AutogradTensor* atensor, bool use_kernels) {
    // 这里可以设置是否使用内核优化
    // 实际实现中会传递到计算内核
}

void autograd_tensor_set_parallel_threads(int num_threads) {
    kernel_init(&(KernelConfig){
        .simd_level = kernel_detect_simd(),
        .num_threads = num_threads,
        .cache_size = 32768,
        .use_mkl = false,
        .use_openblas = false,
        .use_cuda = false
    });
}

int autograd_tensor_get_parallel_threads(void) {
    // 返回当前线程数
    return 1; // 简化实现
}

// 初始化函数
void autograd_tensor_init(void) {
    init_default_engine();
    kernel_init(NULL);
}

void autograd_tensor_cleanup(void) {
    if (g_default_engine) {
        autograd_engine_destroy(g_default_engine);
        g_default_engine = NULL;
    }
    kernel_cleanup();
}