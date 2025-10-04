#ifndef NN_LAYERS_AUTOGRAD_H
#define NN_LAYERS_AUTOGRAD_H

#include "nn.h"
#include "tensor.h"
#include "nn_layers.h"
#include "kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 自动梯度层基础结构 ====================

// 自动梯度层基础类
typedef struct AutogradLayer {
    int training;                    // 是否处于训练模式
    int use_kernel_optimization;     // 是否使用内核优化
    char* name;                      // 层名称
    void (*forward)(struct AutogradLayer*, tensor_t*, tensor_t**);  // 前向传播
    void (*backward)(struct AutogradLayer*, tensor_t*, tensor_t*, tensor_t**); // 反向传播
    void (*reset_parameters)(struct AutogradLayer*);  // 重置参数
    void (*to_device)(struct AutogradLayer*, DeviceType); // 移动到设备
    void (*free)(struct AutogradLayer*);  // 释放资源
} AutogradLayer;

// ==================== 全连接层自动梯度 ====================

// 全连接层（支持自动微分）
typedef struct AutogradLinear {
    AutogradLayer base;              // 基础层
    tensor_t* weight;          // 权重矩阵
    tensor_t* bias;            // 偏置向量
    int use_bias;                  // 是否使用偏置
    int in_features;                // 输入特征数
    int out_features;               // 输出特征数
} AutogradLinear;

// 全连接层函数声明
AutogradLinear* autograd_linear_create(int in_features, int out_features, int bias);
void autograd_linear_destroy(AutogradLinear* layer);
void autograd_linear_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_linear_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_linear_reset_parameters(AutogradLinear* layer);
void autograd_linear_to_device(AutogradLinear* layer, DeviceType device);

// ==================== 卷积层自动梯度 ====================

// 卷积层（支持自动微分）
typedef struct AutogradConv2d {
    AutogradLayer base;              // 基础层
    tensor_t* weight;          // 卷积核权重
    tensor_t* bias;            // 卷积核偏置
    int use_bias;                  // 是否使用偏置
    int in_channels;                // 输入通道数
    int out_channels;               // 输出通道数
    int kernel_size;                // 卷积核大小
    int stride;                     // 步长
    int padding;                    // 填充
    int dilation;                   // 膨胀
} AutogradConv2d;

// 卷积层函数声明
AutogradConv2d* autograd_conv2d_create(int in_channels, int out_channels, int kernel_size, 
                                       int stride, int padding, int dilation, int bias);
void autograd_conv2d_destroy(AutogradConv2d* layer);
void autograd_conv2d_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_conv2d_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_conv2d_reset_parameters(AutogradConv2d* layer);
void autograd_conv2d_to_device(AutogradConv2d* layer, DeviceType device);

// ==================== 批归一化层自动梯度 ====================

// 批归一化层（支持自动微分）
typedef struct AutogradBatchNorm2d {
    AutogradLayer base;              // 基础层
    tensor_t* weight;          // 缩放参数
    tensor_t* bias;            // 偏移参数
    tensor_t* running_mean;    // 运行均值
    tensor_t* running_var;     // 运行方差
    float eps;                      // 数值稳定性
    float momentum;                 // 动量
    int track_running_stats;       // 是否跟踪运行统计
    int num_features;               // 特征数
} AutogradBatchNorm2d;

// 批归一化层函数声明
AutogradBatchNorm2d* autograd_batch_norm2d_create(int num_features, float eps, float momentum, int affine, int track_running_stats);
void autograd_batch_norm2d_destroy(AutogradBatchNorm2d* layer);
void autograd_batch_norm2d_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_batch_norm2d_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_batch_norm2d_reset_parameters(AutogradBatchNorm2d* layer);
void autograd_batch_norm2d_to_device(AutogradBatchNorm2d* layer, DeviceType device);

// ==================== LSTM层自动梯度 ====================

// LSTM层（支持自动微分）
typedef struct AutogradLSTM {
    AutogradLayer base;              // 基础层
    tensor_t* weight_ih;       // 输入到隐藏权重
    tensor_t* weight_hh;       // 隐藏到隐藏权重
    tensor_t* bias_ih;         // 输入到隐藏偏置
    tensor_t* bias_hh;         // 隐藏到隐藏偏置
    int input_size;                 // 输入大小
    int hidden_size;                // 隐藏层大小
    int num_layers;                 // 层数
    float dropout;                  // dropout概率
    int bidirectional;             // 是否双向
    int batch_first;               // batch维度是否在前
} AutogradLSTM;

// LSTM层函数声明
AutogradLSTM* autograd_lstm_create(int input_size, int hidden_size, int num_layers, float dropout, int bidirectional, int batch_first);
void autograd_lstm_destroy(AutogradLSTM* layer);
void autograd_lstm_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_lstm_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_lstm_reset_parameters(AutogradLSTM* layer);
void autograd_lstm_to_device(AutogradLSTM* layer, DeviceType device);

// ==================== 注意力层自动梯度 ====================

// 注意力层（支持自动微分）
typedef struct AutogradMultiheadAttention {
    AutogradLayer base;              // 基础层
    tensor_t* in_proj_weight;  // 输入投影权重
    tensor_t* in_proj_bias;    // 输入投影偏置
    tensor_t* out_proj_weight; // 输出投影权重
    tensor_t* out_proj_bias;   // 输出投影偏置
    int embed_dim;                  // 嵌入维度
    int num_heads;                  // 注意力头数
    float dropout;                  // dropout概率
    int bias;                      // 是否使用偏置
} AutogradMultiheadAttention;

// 注意力层函数声明
AutogradMultiheadAttention* autograd_multihead_attention_create(int embed_dim, int num_heads, float dropout, int bias);
void autograd_multihead_attention_destroy(AutogradMultiheadAttention* layer);
void autograd_multihead_attention_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_multihead_attention_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_multihead_attention_reset_parameters(AutogradMultiheadAttention* layer);
void autograd_multihead_attention_to_device(AutogradMultiheadAttention* layer, DeviceType device);

// ==================== 池化层自动梯度 ====================

// 最大池化层（支持自动微分）
typedef struct AutogradMaxPool2d {
    AutogradLayer base;              // 基础层
    int kernel_size;                // 池化核大小
    int stride;                     // 步长
    int padding;                    // 填充
    int dilation;                   // 膨胀
    int return_indices;            // 是否返回索引
} AutogradMaxPool2d;

// 最大池化层函数声明
AutogradMaxPool2d* autograd_max_pool2d_create(int kernel_size, int stride, int padding, int dilation, int return_indices);
void autograd_max_pool2d_destroy(AutogradMaxPool2d* layer);
void autograd_max_pool2d_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_max_pool2d_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_max_pool2d_reset_parameters(AutogradMaxPool2d* layer);
void autograd_max_pool2d_to_device(AutogradMaxPool2d* layer, DeviceType device);

// 平均池化层（支持自动微分）
typedef struct AutogradAvgPool2d {
    AutogradLayer base;              // 基础层
    int kernel_size;                // 池化核大小
    int stride;                     // 步长
    int padding;                    // 填充
    int count_include_pad;         // 是否包含填充
    int divisor_override;           // 除数覆盖
} AutogradAvgPool2d;

// 平均池化层函数声明
AutogradAvgPool2d* autograd_avg_pool2d_create(int kernel_size, int stride, int padding, int count_include_pad, int divisor_override);
void autograd_avg_pool2d_destroy(AutogradAvgPool2d* layer);
void autograd_avg_pool2d_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_avg_pool2d_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_avg_pool2d_reset_parameters(AutogradAvgPool2d* layer);
void autograd_avg_pool2d_to_device(AutogradAvgPool2d* layer, DeviceType device);

// ==================== Dropout层自动梯度 ====================

// Dropout层（支持自动微分）
typedef struct AutogradDropout {
    AutogradLayer base;              // 基础层
    float p;                        // dropout概率
    int inplace;                   // 是否原地操作
    unsigned int seed;              // 随机种子
} AutogradDropout;

// Dropout层函数声明
AutogradDropout* autograd_dropout_create(float p, int inplace, unsigned int seed);
void autograd_dropout_destroy(AutogradDropout* layer);
void autograd_dropout_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_dropout_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_dropout_reset_parameters(AutogradDropout* layer);
void autograd_dropout_to_device(AutogradDropout* layer, DeviceType device);

// ==================== 层归一化自动梯度 ====================

// 层归一化（支持自动微分）
typedef struct AutogradLayerNorm {
    AutogradLayer base;              // 基础层
    tensor_t* weight;          // 缩放参数
    tensor_t* bias;            // 偏移参数
    int normalized_shape[8];        // 归一化形状
    int normalized_ndim;            // 归一化维度
    float eps;                      // 数值稳定性
    int elementwise_affine;        // 是否元素级仿射
} AutogradLayerNorm;

// 层归一化函数声明
AutogradLayerNorm* autograd_layer_norm_create(const int* normalized_shape, int normalized_ndim, float eps, int elementwise_affine);
void autograd_layer_norm_destroy(AutogradLayerNorm* layer);
void autograd_layer_norm_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);
void autograd_layer_norm_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);
void autograd_layer_norm_reset_parameters(AutogradLayerNorm* layer);
void autograd_layer_norm_to_device(AutogradLayerNorm* layer, DeviceType device);

// ==================== 辅助函数 ====================

// 设置训练模式
void autograd_layer_set_training(AutogradLayer* layer, int training);

// 设置内核优化
void autograd_layer_set_kernel_optimization(AutogradLayer* layer, int use_kernels);

// 重置参数
void autograd_layer_reset_parameters(AutogradLayer* layer);

// 移动到设备
void autograd_layer_to_device(AutogradLayer* layer, DeviceType device);

// 释放层资源
void autograd_layer_free(AutogradLayer* layer);

// ==================== 参数管理 ====================

// 获取层参数
tensor_t** autograd_layer_parameters(AutogradLayer* layer, int* num_params);

// 获取命名参数
tensor_t** autograd_layer_named_parameters(AutogradLayer* layer, const char*** names, int* num_params);

// 打印参数信息
void autograd_layer_print_parameters(AutogradLayer* layer);

// ==================== 层组合和序列 ====================

// 创建序列层
AutogradLayer* autograd_sequential_create(AutogradLayer** layers, int num_layers);

// 销毁序列层
void autograd_sequential_destroy(AutogradLayer* sequential);

// 序列层前向传播
void autograd_sequential_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);

// 序列层反向传播
void autograd_sequential_backward(AutogradLayer* base, tensor_t* grad_output, tensor_t* input, tensor_t** grad_input);

// ==================== 容器层 ====================

// 模块列表
typedef struct AutogradModuleList {
    AutogradLayer base;
    AutogradLayer** layers;
    int num_layers;
    int capacity;
} AutogradModuleList;

// 创建模块列表
AutogradModuleList* autograd_module_list_create(void);

// 销毁模块列表
void autograd_module_list_destroy(AutogradModuleList* module_list);

// 添加层到模块列表
void autograd_module_list_add(AutogradModuleList* module_list, AutogradLayer* layer);

// 模块列表前向传播
void autograd_module_list_forward(AutogradLayer* base, tensor_t* input, tensor_t** output);

// ==================== 初始化函数 ====================

// 初始化自动梯度层
void autograd_layers_init(void);

// 清理自动梯度层
void autograd_layers_cleanup(void);

// ==================== 权重初始化函数 ====================

// 均匀分布初始化
void autograd_init_uniform(tensor_t* tensor, float a, float b);

// 正态分布初始化
void autograd_init_normal(tensor_t* tensor, float mean, float std);

// Xavier均匀初始化
void autograd_init_xavier_uniform(tensor_t* tensor);

// Xavier正态初始化
void autograd_init_xavier_normal(tensor_t* tensor);

// Kaiming均匀初始化
void autograd_init_kaiming_uniform(tensor_t* tensor, float a);

// Kaiming正态初始化
void autograd_init_kaiming_normal(tensor_t* tensor, float a);

// 正交初始化
void autograd_init_orthogonal(tensor_t* tensor, float gain);

// 常数初始化
void autograd_init_constant(tensor_t* tensor, float val);

// 单位矩阵初始化
void autograd_init_eye(tensor_t* tensor);

// Dirac初始化
void autograd_init_dirac(tensor_t* tensor);

// 稀疏初始化
void autograd_init_sparse(tensor_t* tensor, float sparsity, float std);

// ==================== 实用函数 ====================

// 打印层信息
void autograd_layer_info(AutogradLayer* layer);

// 打印层摘要
void autograd_layer_summary(AutogradLayer* layer);

// 获取参数数量
size_t autograd_layer_num_parameters(AutogradLayer* layer);

// 获取内存使用量
size_t autograd_layer_memory_usage(AutogradLayer* layer);

// ==================== 层注册和工厂 ====================

// 注册层类型
void autograd_register_layer(const char* name, AutogradLayer* (*create_func)(void));

// 创建层
AutogradLayer* autograd_create_layer(const char* name);

// 注销层类型
void autograd_unregister_layer(const char* name);

#ifdef __cplusplus
}
#endif

#endif // NN_LAYERS_AUTOGRAD_H