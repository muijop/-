#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 简单的神经网络层结构
typedef struct {
    float* weights;
    float* bias;
    float* output;
    size_t input_size;
    size_t output_size;
} Layer;

// 创建层
Layer* create_layer(size_t input_size, size_t output_size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) return NULL;
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // 初始化权重和偏置
    layer->weights = (float*)malloc(input_size * output_size * sizeof(float));
    layer->bias = (float*)malloc(output_size * sizeof(float));
    layer->output = (float*)malloc(output_size * sizeof(float));
    
    if (!layer->weights || !layer->bias || !layer->output) {
        free(layer->weights);
        free(layer->bias);
        free(layer->output);
        free(layer);
        return NULL;
    }
    
    // 随机初始化权重和偏置
    for (size_t i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    for (size_t i = 0; i < output_size; i++) {
        layer->bias[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    return layer;
}

// 前向传播
void forward(Layer* layer, const float* input) {
    // 矩阵乘法：output = input * weights + bias
    for (size_t i = 0; i < layer->output_size; i++) {
        layer->output[i] = layer->bias[i];
        for (size_t j = 0; j < layer->input_size; j++) {
            layer->output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }
    }
}

// ReLU激活函数
void relu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0.0f;
    }
}

// 生成训练数据
void generate_data(float* inputs, float* targets, size_t num_samples, size_t input_size, size_t output_size) {
    for (size_t i = 0; i < num_samples; i++) {
        // 生成随机输入
        for (size_t j = 0; j < input_size; j++) {
            inputs[i * input_size + j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        
        // 生成简单目标：如果输入和为正则为1，否则为0
        float sum = 0.0f;
        for (size_t j = 0; j < input_size; j++) {
            sum += inputs[i * input_size + j];
        }
        
        targets[i * output_size] = sum > 0 ? 1.0f : 0.0f;
        if (output_size > 1) {
            targets[i * output_size + 1] = sum <= 0 ? 1.0f : 0.0f;
        }
    }
}

// 计算准确率
float calculate_accuracy(const float* predictions, const float* targets, size_t num_samples, size_t output_size) {
    size_t correct = 0;
    
    for (size_t i = 0; i < num_samples; i++) {
        size_t pred_class = 0;
        size_t target_class = 0;
        
        // 找到预测的最大值索引
        float max_pred = predictions[i * output_size];
        for (size_t j = 1; j < output_size; j++) {
            if (predictions[i * output_size + j] > max_pred) {
                max_pred = predictions[i * output_size + j];
                pred_class = j;
            }
        }
        
        // 找到目标的最大值索引
        float max_target = targets[i * output_size];
        for (size_t j = 1; j < output_size; j++) {
            if (targets[i * output_size + j] > max_target) {
                max_target = targets[i * output_size + j];
                target_class = j;
            }
        }
        
        if (pred_class == target_class) {
            correct++;
        }
    }
    
    return (float)correct / num_samples;
}

int main() {
    printf("=== 简单神经网络训练测试 ===\n");
    
    // 设置随机种子
    srand(42);
    
    // 参数设置
    size_t input_size = 10;
    size_t hidden_size = 5;
    size_t output_size = 2;
    size_t num_samples = 1000;
    size_t epochs = 10;
    float learning_rate = 0.01f;
    
    // 创建层
    Layer* hidden_layer = create_layer(input_size, hidden_size);
    Layer* output_layer = create_layer(hidden_size, output_size);
    
    if (!hidden_layer || !output_layer) {
        printf("错误：无法创建神经网络层\n");
        return -1;
    }
    
    // 生成训练数据
    float* inputs = (float*)malloc(num_samples * input_size * sizeof(float));
    float* targets = (float*)malloc(num_samples * output_size * sizeof(float));
    float* predictions = (float*)malloc(num_samples * output_size * sizeof(float));
    
    if (!inputs || !targets || !predictions) {
        printf("错误：内存分配失败\n");
        return -1;
    }
    
    generate_data(inputs, targets, num_samples, input_size, output_size);
    
    printf("开始训练...\n");
    
    // 简单的训练循环
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < num_samples; i++) {
            const float* input = inputs + i * input_size;
            const float* target = targets + i * output_size;
            
            // 前向传播
            forward(hidden_layer, input);
            relu(hidden_layer->output, hidden_size);
            
            forward(output_layer, hidden_layer->output);
            
            // 简单的softmax（用于多分类）
            float max_val = output_layer->output[0];
            for (size_t j = 1; j < output_size; j++) {
                if (output_layer->output[j] > max_val) {
                    max_val = output_layer->output[j];
                }
            }
            
            float sum = 0.0f;
            for (size_t j = 0; j < output_size; j++) {
                output_layer->output[j] = expf(output_layer->output[j] - max_val);
                sum += output_layer->output[j];
            }
            
            for (size_t j = 0; j < output_size; j++) {
                output_layer->output[j] /= sum;
            }
            
            // 计算损失（交叉熵）
            float loss = 0.0f;
            for (size_t j = 0; j < output_size; j++) {
                loss += -target[j] * logf(output_layer->output[j] + 1e-8f);
            }
            total_loss += loss;
            
            // 保存预测结果
            for (size_t j = 0; j < output_size; j++) {
                predictions[i * output_size + j] = output_layer->output[j];
            }
        }
        
        float avg_loss = total_loss / num_samples;
        float accuracy = calculate_accuracy(predictions, targets, num_samples, output_size);
        
        printf("Epoch %zu/%zu - 损失: %.4f, 准确率: %.2f%%\n", 
               epoch + 1, epochs, avg_loss, accuracy * 100);
    }
    
    printf("训练完成！\n");
    
    // 清理内存
    free(hidden_layer->weights);
    free(hidden_layer->bias);
    free(hidden_layer->output);
    free(hidden_layer);
    
    free(output_layer->weights);
    free(output_layer->bias);
    free(output_layer->output);
    free(output_layer);
    
    free(inputs);
    free(targets);
    free(predictions);
    
    printf("测试程序完成！\n");
    
    return 0;
}