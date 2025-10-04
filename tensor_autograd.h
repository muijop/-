#ifndef TENSOR_AUTOGRAD_H
#define TENSOR_AUTOGRAD_H

#include "unified_tensor.h"
#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 自动梯度张量结构 ====================

// 自动梯度张量（支持自动微分）
typedef struct AutogradTensor {
    tensor_t* tensor;              // 基础张量
    tensor_t* grad;                // 梯度张量
    int requires_grad;            // 是否需要计算梯度
    int is_leaf;                  // 是否为叶子节点
    struct AutogradTensor* (*backward_fn)(struct AutogradTensor*, tensor_t*); // 反向传播函数
    void* backward_data;          // 反向传播数据
    struct AutogradTensor** children; // 子节点
    int num_children;             // 子节点数量
    char* name;                   // 张量名称
} AutogradTensor;

// ==================== 自动梯度张量创建和销毁 ====================

// 创建自动梯度张量
AutogradTensor* autograd_tensor_create(tensor_t* tensor, int requires_grad, const char* name);

// 销毁自动梯度张量
void autograd_tensor_destroy(AutogradTensor* atensor);

// ==================== 数据访问函数 ====================

// 获取基础张量
tensor_t* autograd_tensor_tensor(AutogradTensor* atensor);

// 获取张量数据
void* autograd_tensor_data(AutogradTensor* atensor);

// 获取浮点数据
float* autograd_tensor_data_float(AutogradTensor* atensor);

// 获取整型数据
int* autograd_tensor_data_int(AutogradTensor* atensor);

// 获取梯度张量
tensor_t* autograd_tensor_grad(AutogradTensor* atensor);

// 获取梯度数据
void* autograd_tensor_grad_data(AutogradTensor* atensor);

// ==================== 张量运算函数 ====================

// 张量加法
AutogradTensor* autograd_tensor_add_simple(AutogradTensor* a, AutogradTensor* b);

// 张量减法
AutogradTensor* autograd_tensor_sub_simple(AutogradTensor* a, AutogradTensor* b);

// 张量乘法
AutogradTensor* autograd_tensor_mul_simple(AutogradTensor* a, AutogradTensor* b);

// 张量除法
AutogradTensor* autograd_tensor_div_simple(AutogradTensor* a, AutogradTensor* b);

// 矩阵乘法
AutogradTensor* autograd_tensor_matmul_simple(AutogradTensor* a, AutogradTensor* b);

// 标量加法
AutogradTensor* autograd_tensor_add_scalar(AutogradTensor* a, float scalar);

// 标量乘法
AutogradTensor* autograd_tensor_multiply_scalar(AutogradTensor* a, float scalar);

// 标量除法
AutogradTensor* autograd_tensor_divide_scalar(AutogradTensor* a, float scalar);

// ==================== 张量形状操作 ====================

// 重塑张量
AutogradTensor* autograd_tensor_reshape(AutogradTensor* a, const int* new_shape, int new_ndim);

// 转置张量
AutogradTensor* autograd_tensor_transpose(AutogradTensor* a, int dim1, int dim2);

// 维度重排
AutogradTensor* autograd_tensor_permute(AutogradTensor* a, const int* dims);

// 切片操作
AutogradTensor* autograd_tensor_slice(AutogradTensor* a, const int* start, const int* end, const int* step);

// ==================== 归约操作 ====================

// 求和
AutogradTensor* autograd_tensor_sum_simple(AutogradTensor* a, int dim, int keepdim);

// 平均值
AutogradTensor* autograd_tensor_mean_simple(AutogradTensor* a, int dim, int keepdim);

// 最大值
AutogradTensor* autograd_tensor_max(AutogradTensor* a, int dim, int keepdim);

// 最小值
AutogradTensor* autograd_tensor_min(AutogradTensor* a, int dim, int keepdim);

// ==================== 激活函数 ====================

// ReLU激活函数
AutogradTensor* autograd_tensor_relu_simple(AutogradTensor* a);

// Sigmoid激活函数
AutogradTensor* autograd_tensor_sigmoid_simple(AutogradTensor* a);

// Tanh激活函数
AutogradTensor* autograd_tensor_tanh_simple(AutogradTensor* a);

// Softmax激活函数
AutogradTensor* autograd_tensor_softmax_simple(AutogradTensor* a, int dim);

// Leaky ReLU激活函数
AutogradTensor* autograd_tensor_leaky_relu(AutogradTensor* a, float negative_slope);

// ==================== 损失函数 ====================

// 均方误差损失
AutogradTensor* autograd_tensor_mse_loss(AutogradTensor* pred, AutogradTensor* target);

// 交叉熵损失
AutogradTensor* autograd_tensor_cross_entropy_loss(AutogradTensor* pred, AutogradTensor* target);

// 二元交叉熵损失
AutogradTensor* autograd_tensor_binary_cross_entropy_loss(AutogradTensor* pred, AutogradTensor* target);

// ==================== 梯度计算 ====================

// 反向传播
void autograd_tensor_backward(AutogradTensor* atensor, tensor_t* grad_output);

// 清零梯度
void autograd_tensor_zero_grad(AutogradTensor* atensor);

// ==================== 计算图管理 ====================

// 获取计算图
void* autograd_tensor_graph(AutogradTensor* atensor);

// 分离张量
void autograd_tensor_detach(AutogradTensor* atensor);

// 创建分离副本
AutogradTensor* autograd_tensor_detach_copy(AutogradTensor* atensor);

// ==================== 张量属性 ====================

// 获取形状
const int* autograd_tensor_shape(AutogradTensor* atensor);

// 获取维度数
int autograd_tensor_ndim(AutogradTensor* atensor);

// 获取元素总数
int autograd_tensor_size(AutogradTensor* atensor);

// ==================== 设备管理 ====================

// 移动到CPU
AutogradTensor* autograd_tensor_to_cpu(AutogradTensor* atensor);

// 移动到CUDA
AutogradTensor* autograd_tensor_to_cuda(AutogradTensor* atensor);

// ==================== 数据类型转换 ====================

// 转换为float32
AutogradTensor* autograd_tensor_to_float32(AutogradTensor* atensor);

// 转换为float16
AutogradTensor* autograd_tensor_to_float16(AutogradTensor* atensor);

// 转换为int32
AutogradTensor* autograd_tensor_to_int32(AutogradTensor* atensor);

// ==================== 张量比较 ====================

// 张量相等性检查
int autograd_tensor_equal(AutogradTensor* a, AutogradTensor* b);

// ==================== 打印和调试 ====================

// 打印张量
void autograd_tensor_print(AutogradTensor* atensor, const char* name);

// 打印形状
void autograd_tensor_print_shape(AutogradTensor* atensor, const char* name);

// 检查梯度
int autograd_tensor_check_grad(AutogradTensor* atensor);

// 打印梯度
void autograd_tensor_print_grad(AutogradTensor* atensor, const char* name);

// 设置梯度
void autograd_tensor_set_grad(AutogradTensor* atensor, tensor_t* grad);

// ==================== 神经网络操作 ====================

// 卷积操作
AutogradTensor* autograd_tensor_conv2d(AutogradTensor* input, AutogradTensor* weight, AutogradTensor* bias,
                                      int stride, int padding, int dilation);

// 最大池化
AutogradTensor* autograd_tensor_max_pool2d(AutogradTensor* input, int kernel_size, int stride, int padding);

// 平均池化
AutogradTensor* autograd_tensor_avg_pool2d(AutogradTensor* input, int kernel_size, int stride, int padding);

// 批归一化
AutogradTensor* autograd_tensor_batch_norm(AutogradTensor* input, AutogradTensor* weight, AutogradTensor* bias,
                                          AutogradTensor* running_mean, AutogradTensor* running_var,
                                          int training, float momentum, float eps);

// ==================== 优化器操作 ====================

// SGD优化器步骤
void autograd_tensor_step_sgd(AutogradTensor* atensor, float learning_rate, float weight_decay, float momentum);

// Adam优化器步骤
void autograd_tensor_step_adam(AutogradTensor* atensor, float learning_rate, float beta1, float beta2, float eps, float weight_decay);

// RMSprop优化器步骤
void autograd_tensor_step_rmsprop(AutogradTensor* atensor, float learning_rate, float alpha, float eps, float weight_decay);

// ==================== 序列化和反序列化 ====================

// 获取序列化大小
size_t autograd_tensor_serialize_size(AutogradTensor* atensor);

// 序列化张量
void autograd_tensor_serialize(AutogradTensor* atensor, void* buffer, size_t buffer_size);

// 反序列化张量
AutogradTensor* autograd_tensor_deserialize(const void* buffer, size_t buffer_size);

// ==================== 内核优化 ====================

// 设置内核优化
void autograd_tensor_use_kernel_optimization(AutogradTensor* atensor, int use_kernels);

// ==================== 实用函数 ====================

// 获取张量内存使用量
size_t autograd_tensor_memory_usage(AutogradTensor* atensor);

// 获取梯度内存使用量
size_t autograd_tensor_grad_memory_usage(AutogradTensor* atensor);

// 检查张量是否有效
int autograd_tensor_is_valid(AutogradTensor* atensor);

// 重置张量状态
void autograd_tensor_reset(AutogradTensor* atensor);

// 复制张量
AutogradTensor* autograd_tensor_copy(AutogradTensor* atensor);

// 深度复制张量
AutogradTensor* autograd_tensor_deep_copy(AutogradTensor* atensor);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_AUTOGRAD_H