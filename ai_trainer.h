#ifndef AI_TRAINER_H
#define AI_TRAINER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "unified_tensor.h"

// ==================== 基础类型定义 ====================
typedef enum {
    TRAINER_SUCCESS = 0,
    TRAINER_ERROR_INVALID_PARAM = -1,
    TRAINER_ERROR_OUT_OF_MEMORY = -2,
    TRAINER_ERROR_NOT_INITIALIZED = -3,
    TRAINER_ERROR_FILE_OPERATION = -4
} trainer_error_t;

// 模型配置常量
#define MAX_MODEL_LAYERS 100
#define EPSILON 1e-8f

// ==================== 张量类型 ====================
// 使用统一的tensor_t类型定义（来自unified_tensor.h）

// ==================== 神经网络层 ====================
typedef struct linear_layer_t {
    tensor_t* weight;
    tensor_t* bias;
    int input_size;
    int output_size;
    tensor_t* last_input; // 用于反向传播
} linear_layer_t;

typedef struct relu_layer_t {
    tensor_t* last_input; // 保存前向传播的输入，用于反向传播
} relu_layer_t;

typedef struct softmax_layer_t {
    tensor_t* last_output; // 保存前向传播的输出，用于反向传播
} softmax_layer_t;

// ==================== 优化器 ====================
typedef struct sgd_optimizer_t {
    float learning_rate;
    float momentum;
    float* velocity;
    int num_params;
} sgd_optimizer_t;

typedef struct adam_optimizer_t {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int step;
    float* m;
    float* v;
    int num_params;
} adam_optimizer_t;

// ==================== 损失函数 ====================
typedef struct mse_loss_t {
    // MSE损失没有参数
} mse_loss_t;

typedef struct cross_entropy_loss_t {
    // 交叉熵损失没有参数
} cross_entropy_loss_t;

// ==================== 模型定义 ====================
typedef struct sequential_model_t {
    void** layers;
    int num_layers;
    int* layer_types; // 0: linear, 1: relu, 2: softmax
} sequential_model_t;

// ==================== 训练器 ====================
typedef struct trainer_t {
    sequential_model_t* model;
    void* optimizer;
    int optimizer_type; // 0: sgd, 1: adam
    void* loss_function;
    int loss_type; // 0: mse, 1: cross_entropy
    int batch_size;
    int epochs;
    float* training_loss;
    float* validation_loss;
} trainer_t;

// ==================== API函数声明 ====================

// 张量操作
tensor_t* tensor_create(const float* data, const size_t* shape, size_t ndim, bool requires_grad);
void tensor_destroy(tensor_t* tensor);
tensor_t* tensor_zeros(const size_t* shape, size_t ndim, bool requires_grad);
tensor_t* tensor_ones(const size_t* shape, size_t ndim, bool requires_grad);
tensor_t* tensor_randn(const size_t* shape, size_t ndim, bool requires_grad);
void tensor_copy(tensor_t* dest, tensor_t* src);

// 神经网络层
linear_layer_t* linear_layer_create(int input_size, int output_size);
void linear_layer_destroy(linear_layer_t* layer);
tensor_t* linear_layer_forward(linear_layer_t* layer, tensor_t* input);
tensor_t* linear_layer_backward(linear_layer_t* layer, tensor_t* grad_output);

relu_layer_t* relu_layer_create();
void relu_layer_destroy(relu_layer_t* layer);
tensor_t* relu_layer_forward(relu_layer_t* layer, tensor_t* input);
tensor_t* relu_layer_backward(relu_layer_t* layer, tensor_t* grad_output);

softmax_layer_t* softmax_layer_create();
void softmax_layer_destroy(softmax_layer_t* layer);
tensor_t* softmax_layer_forward(softmax_layer_t* layer, tensor_t* input);
tensor_t* softmax_layer_backward(softmax_layer_t* layer, tensor_t* grad_output);

// 优化器
sgd_optimizer_t* sgd_optimizer_create(float learning_rate, float momentum);
void sgd_optimizer_destroy(sgd_optimizer_t* optimizer);
void sgd_optimizer_step(sgd_optimizer_t* optimizer, tensor_t** params, int num_params);

adam_optimizer_t* adam_optimizer_create(float learning_rate, float beta1, float beta2, float epsilon);
void adam_optimizer_destroy(adam_optimizer_t* optimizer);
void adam_optimizer_step(adam_optimizer_t* optimizer, tensor_t** params, int num_params);

// 损失函数
mse_loss_t* mse_loss_create();
void mse_loss_destroy(mse_loss_t* loss);
float mse_loss_forward(mse_loss_t* loss, tensor_t* predictions, tensor_t* targets);
tensor_t* mse_loss_backward(mse_loss_t* loss, tensor_t* predictions, tensor_t* targets);

cross_entropy_loss_t* cross_entropy_loss_create();
void cross_entropy_loss_destroy(cross_entropy_loss_t* loss);
float cross_entropy_loss_forward(cross_entropy_loss_t* loss, tensor_t* predictions, tensor_t* targets);
tensor_t* cross_entropy_loss_backward(cross_entropy_loss_t* loss, tensor_t* predictions, tensor_t* targets);

// 模型
sequential_model_t* sequential_model_create();
void sequential_model_destroy(sequential_model_t* model);
void sequential_model_add_layer(sequential_model_t* model, void* layer, int layer_type);
tensor_t* sequential_model_forward(sequential_model_t* model, tensor_t* input);
void sequential_model_backward(sequential_model_t* model, tensor_t* grad_output);

// 模型保存/加载
int sequential_model_save(sequential_model_t* model, const char* filename);
sequential_model_t* sequential_model_load(const char* filename);

// 训练器
trainer_t* trainer_create(sequential_model_t* model, int optimizer_type, int loss_type);
void trainer_destroy(trainer_t* trainer);
void trainer_set_hyperparameters(trainer_t* trainer, int batch_size, int epochs);
void trainer_train(trainer_t* trainer, tensor_t** inputs, tensor_t** targets, int num_samples);

// 梯度检查
float gradient_check(sequential_model_t* model, tensor_t* input, tensor_t* target, int loss_type, float epsilon);

#endif // AI_TRAINER_H