#ifndef SIMPLE_FRAMEWORK_H
#define SIMPLE_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 基本类型定义
typedef enum {
    MODEL_TYPE_FEEDFORWARD = 0,
    MODEL_TYPE_CONVOLUTIONAL = 1,
    MODEL_TYPE_RECURRENT = 2
} model_type_t;

typedef enum {
    OPTIMIZER_SGD = 0,
    OPTIMIZER_ADAM = 1,
    OPTIMIZER_RMSPROP = 2
} optimizer_type_t;

typedef enum {
    TRAINING_MODE_STANDARD = 0,
    TRAINING_MODE_HYPERPARAM_OPTIMIZATION = 1
} training_mode_t;

// 张量结构体
typedef struct {
    float* data;
    size_t* shape;
    size_t ndim;
    size_t size;
    float* grad;  // 梯度数据
} tensor_t;

// 训练配置
typedef struct {
    training_mode_t training_mode;
    model_type_t model_type;
    optimizer_type_t optimizer_type;
    float learning_rate;
    size_t batch_size;
    size_t epochs;
    int use_early_stopping;
    size_t patience;
    float validation_split;
} training_config_t;

// 训练数据
typedef struct {
    float* input_data;
    float* target_data;
    size_t data_size;
    size_t input_dim;
    size_t output_dim;
} training_data_t;

// 训练结果
typedef struct {
    int success;
    float final_loss;
    float final_accuracy;
    size_t training_time_ms;
    char error_message[256];
} training_result_t;

// 基本函数声明
tensor_t* tensor_create(const float* data, const size_t* shape, size_t ndim, int requires_grad);
void tensor_destroy(tensor_t* tensor);

#endif // SIMPLE_FRAMEWORK_H