#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include "nn.h"
#include "tensor.h"
#include "tensor_autograd.h"
#include "nn_layers_autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 自动梯度模型结构 ====================

// 自动梯度模型（支持自动微分）
typedef struct AutogradModel {
    AutogradLayer** layers;           // 模型层
    int num_layers;                   // 层数
    int training;                     // 训练模式标志
    char* name;                       // 模型名称
    void* optimizer;                  // 优化器
    void* loss_function;              // 损失函数
    float learning_rate;              // 学习率
    int use_kernel_optimization;      // 是否使用内核优化
} AutogradModel;

// ==================== 模型创建和销毁 ====================

// 创建自动梯度模型
AutogradModel* autograd_model_create(const char* name);

// 销毁自动梯度模型
void autograd_model_destroy(AutogradModel* model);

// ==================== 层管理 ====================

// 添加层到模型
void autograd_model_add_layer(AutogradModel* model, AutogradLayer* layer);

// 获取模型层
AutogradLayer* autograd_model_get_layer(AutogradModel* model, int index);

// 获取层数
int autograd_model_num_layers(AutogradModel* model);

// 移除层
void autograd_model_remove_layer(AutogradModel* model, int index);

// 插入层
void autograd_model_insert_layer(AutogradModel* model, int index, AutogradLayer* layer);

// ==================== 前向传播 ====================

// 模型前向传播
AutogradTensor* autograd_model_forward(AutogradModel* model, AutogradTensor* input);

// 模型反向传播
AutogradTensor* autograd_model_backward(AutogradModel* model, AutogradTensor* loss);

// ==================== 训练和评估 ====================

// 模型预测
AutogradTensor* autograd_model_predict(AutogradModel* model, AutogradTensor* input);

// 模型评估
float autograd_model_evaluate(AutogradModel* model, AutogradTensor* input, AutogradTensor* target);

// 训练步骤
float autograd_train_step(AutogradModel* model, AutogradTensor* input, AutogradTensor* target);

// 验证步骤
float autograd_validate_step(AutogradModel* model, AutogradTensor* input, AutogradTensor* target);

// 训练周期
void autograd_train_epoch(AutogradModel* model, AutogradTensor* train_data, AutogradTensor* train_targets,
                          int batch_size, int shuffle);

// 验证周期
void autograd_validate_epoch(AutogradModel* model, AutogradTensor* val_data, AutogradTensor* val_targets,
                            int batch_size);

// 训练模型
void autograd_train_model(AutogradModel* model, AutogradTensor* train_data, AutogradTensor* train_targets,
                          AutogradTensor* val_data, AutogradTensor* val_targets,
                          int epochs, int batch_size, float learning_rate, int verbose);

// ==================== 参数管理 ====================

// 获取模型参数
AutogradTensor** autograd_model_parameters(AutogradModel* model, int* num_params);

// 获取命名参数
AutogradTensor** autograd_model_named_parameters(AutogradModel* model, const char*** names, int* num_params);

// 设置模型参数
void autograd_model_set_parameters(AutogradModel* model, AutogradTensor** parameters, int num_params);

// 加载模型参数
void autograd_model_load_parameters(AutogradModel* model, const char* filename);

// 保存模型参数
void autograd_model_save_parameters(AutogradModel* model, const char* filename);

// ==================== 优化器管理 ====================

// 设置优化器
void autograd_model_set_optimizer(AutogradModel* model, void* optimizer);

// 获取优化器
void* autograd_model_get_optimizer(AutogradModel* model);

// 设置学习率
void autograd_model_set_learning_rate(AutogradModel* model, float learning_rate);

// 获取学习率
float autograd_model_get_learning_rate(AutogradModel* model);

// ==================== 损失函数管理 ====================

// 设置损失函数
void autograd_model_set_loss_function(AutogradModel* model, void* loss_function);

// 获取损失函数
void* autograd_model_get_loss_function(AutogradModel* model);

// ==================== 训练模式管理 ====================

// 设置训练模式
void autograd_model_set_training(AutogradModel* model, int training);

// 获取训练模式
int autograd_model_get_training(AutogradModel* model);

// ==================== 内核优化管理 ====================

// 设置内核优化
void autograd_model_set_kernel_optimization(AutogradModel* model, int use_kernels);

// 获取内核优化状态
int autograd_model_get_kernel_optimization(AutogradModel* model);

// ==================== 层输出管理 ====================

// 获取层输出
void autograd_model_get_layer_outputs(AutogradModel* model, AutogradTensor* input, 
                                      AutogradTensor*** outputs, int* num_outputs);

// 释放层输出
void autograd_model_free_layer_outputs(AutogradTensor** outputs, int num_outputs);

// ==================== 权重管理 ====================

// 设置层权重
void autograd_model_set_layer_weights(AutogradModel* model, int layer_idx, AutogradTensor* weights);

// 获取层权重
AutogradTensor* autograd_model_get_layer_weights(AutogradModel* model, int layer_idx);

// ==================== 梯度管理 ====================

// 获取层梯度
AutogradTensor* autograd_model_get_layer_gradients(AutogradModel* model, int layer_idx);

// 释放层梯度
void autograd_model_free_layer_gradients(AutogradTensor* gradients, int num_gradients);

// ==================== 模型分析 ====================

// 模型分析
void autograd_model_profile(AutogradModel* model, AutogradTensor* input, const char* output_path);

// 获取模型参数数量
size_t autograd_model_num_parameters(AutogradModel* model);

// 获取模型内存使用量
size_t autograd_model_memory_usage(AutogradModel* model);

// ==================== 模型序列化 ====================

// 序列化模型
void autograd_model_serialize(AutogradModel* model, const char* filename);

// 反序列化模型
AutogradModel* autograd_model_deserialize(const char* filename);

// ==================== 模型组合 ====================

// 创建序列模型
AutogradModel* autograd_sequential_create(AutogradLayer** layers, int num_layers);

// 创建并行模型
AutogradModel* autograd_parallel_create(AutogradModel** models, int num_models);

// ==================== 预定义模型 ====================

// 创建多层感知机
AutogradModel* autograd_mlp_create(int* layer_sizes, int num_layers, int activation_type);

// 创建卷积神经网络
AutogradModel* autograd_cnn_create(int* conv_layers, int* fc_layers, int num_conv_layers, int num_fc_layers);

// 创建循环神经网络
AutogradModel* autograd_rnn_create(int input_size, int hidden_size, int num_layers, int rnn_type);

// 创建Transformer模型
AutogradModel* autograd_transformer_create(int d_model, int nhead, int num_layers, int d_ff, float dropout);

// ==================== 模型工具函数 ====================

// 打印模型信息
void autograd_model_print_info(AutogradModel* model);

// 打印模型摘要
void autograd_model_print_summary(AutogradModel* model);

// 检查模型有效性
int autograd_model_is_valid(AutogradModel* model);

// 重置模型状态
void autograd_model_reset(AutogradModel* model);

// 复制模型
AutogradModel* autograd_model_copy(AutogradModel* model);

// 深度复制模型
AutogradModel* autograd_model_deep_copy(AutogradModel* model);

#ifdef __cplusplus
}
#endif

#endif // MODEL_BUILDER_H