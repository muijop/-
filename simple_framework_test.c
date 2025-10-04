#include "simple_framework.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 简单的张量创建函数
tensor_t* tensor_create(const float* data, const size_t* shape, size_t ndim, int requires_grad) {
    tensor_t* tensor = (tensor_t*)malloc(sizeof(tensor_t));
    if (!tensor) return NULL;
    
    tensor->ndim = ndim;
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    
    // 计算总大小
    tensor->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }
    
    // 分配数据内存
    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    
    // 复制数据或初始化为0
    if (data) {
        memcpy(tensor->data, data, tensor->size * sizeof(float));
    } else {
        memset(tensor->data, 0, tensor->size * sizeof(float));
    }
    
    // 分配梯度内存（如果需要）
    if (requires_grad) {
        tensor->grad = (float*)calloc(tensor->size, sizeof(float));
        if (!tensor->grad) {
            free(tensor->data);
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
    } else {
        tensor->grad = NULL;
    }
    
    return tensor;
}

// 张量销毁函数
void tensor_destroy(tensor_t* tensor) {
    if (!tensor) return;
    
    if (tensor->data) free(tensor->data);
    if (tensor->shape) free(tensor->shape);
    if (tensor->grad) free(tensor->grad);
    free(tensor);
}

// 生成示例数据
float* generate_sample_data(size_t num_samples, size_t input_dim) {
    float* data = (float*)malloc(num_samples * input_dim * sizeof(float));
    if (!data) return NULL;
    
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < input_dim; j++) {
            data[i * input_dim + j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    
    return data;
}

// 生成示例标签
float* generate_sample_labels(const float* input_data, size_t num_samples, size_t input_dim, size_t output_dim) {
    float* labels = (float*)malloc(num_samples * output_dim * sizeof(float));
    if (!labels) return NULL;
    
    for (size_t i = 0; i < num_samples; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < input_dim; j++) {
            sum += input_data[i * input_dim + j];
        }
        
        // 二分类问题
        labels[i * output_dim] = sum > 0 ? 1.0f : 0.0f;
        if (output_dim > 1) {
            labels[i * output_dim + 1] = sum <= 0 ? 1.0f : 0.0f;
        }
    }
    
    return labels;
}

// 简单的训练函数
training_result_t* simple_train(const training_config_t* config, const training_data_t* data) {
    training_result_t* result = (training_result_t*)malloc(sizeof(training_result_t));
    if (!result) return NULL;
    
    // 初始化结果
    result->success = 1;
    result->final_loss = 0.5f;  // 模拟损失
    result->final_accuracy = 0.85f;  // 模拟准确率
    result->training_time_ms = 1000;  // 模拟训练时间
    strcpy(result->error_message, "");
    
    printf("训练配置：\n");
    printf("  训练模式: %d\n", config->training_mode);
    printf("  模型类型: %d\n", config->model_type);
    printf("  优化器: %d\n", config->optimizer_type);
    printf("  学习率: %.4f\n", config->learning_rate);
    printf("  批次大小: %zu\n", config->batch_size);
    printf("  训练轮数: %zu\n", config->epochs);
    printf("  数据大小: %zu\n", data->data_size);
    printf("  输入维度: %zu\n", data->input_dim);
    printf("  输出维度: %zu\n", data->output_dim);
    
    // 模拟训练过程
    printf("\n开始模拟训练...\n");
    for (size_t epoch = 0; epoch < config->epochs; epoch++) {
        float loss = 0.8f - (float)epoch * 0.05f;  // 模拟损失下降
        float accuracy = 0.5f + (float)epoch * 0.05f;  // 模拟准确率上升
        
        printf("Epoch %zu/%zu - 损失: %.4f, 准确率: %.2f%%\n", 
               epoch + 1, config->epochs, loss, accuracy * 100);
    }
    
    printf("训练完成！\n");
    
    return result;
}

// 销毁训练结果
void destroy_training_result(training_result_t* result) {
    if (result) free(result);
}

int main() {
    printf("=== 简化AI框架测试程序 ===\n");
    
    // 设置随机种子
    srand(42);
    
    // 生成示例数据
    size_t num_samples = 100;
    size_t input_dim = 10;
    size_t output_dim = 2;
    
    float* input_data = generate_sample_data(num_samples, input_dim);
    float* target_data = generate_sample_labels(input_data, num_samples, input_dim, output_dim);
    
    if (!input_data || !target_data) {
        printf("错误：无法生成示例数据\n");
        return -1;
    }
    
    // 创建张量测试
    printf("\n1. 张量创建测试...\n");
    size_t shape[] = {2, 3, 4};
    tensor_t* tensor = tensor_create(NULL, shape, 3, 1);
    
    if (tensor) {
        printf("张量创建成功！\n");
        printf("  维度: %zu\n", tensor->ndim);
        printf("  形状: [%zu, %zu, %zu]\n", tensor->shape[0], tensor->shape[1], tensor->shape[2]);
        printf("  大小: %zu\n", tensor->size);
        printf("  梯度分配: %s\n", tensor->grad ? "是" : "否");
        
        tensor_destroy(tensor);
        printf("张量销毁成功！\n");
    } else {
        printf("张量创建失败！\n");
    }
    
    // 训练配置测试
    printf("\n2. 训练配置测试...\n");
    training_config_t config;
    config.training_mode = TRAINING_MODE_STANDARD;
    config.model_type = MODEL_TYPE_FEEDFORWARD;
    config.optimizer_type = OPTIMIZER_ADAM;
    config.learning_rate = 0.001f;
    config.batch_size = 32;
    config.epochs = 5;
    config.use_early_stopping = 1;
    config.patience = 3;
    config.validation_split = 0.2f;
    
    training_data_t data;
    data.input_data = input_data;
    data.target_data = target_data;
    data.data_size = num_samples;
    data.input_dim = input_dim;
    data.output_dim = output_dim;
    
    // 执行训练
    training_result_t* result = simple_train(&config, &data);
    
    if (result && result->success) {
        printf("\n训练结果：\n");
        printf("  最终损失: %.4f\n", result->final_loss);
        printf("  最终准确率: %.2f%%\n", result->final_accuracy * 100);
        printf("  训练时间: %zu ms\n", result->training_time_ms);
    } else {
        printf("训练失败: %s\n", result ? result->error_message : "未知错误");
    }
    
    // 清理资源
    if (result) destroy_training_result(result);
    free(input_data);
    free(target_data);
    
    printf("\n测试程序完成！\n");
    
    return 0;
}