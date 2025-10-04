#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 简单的神经网络测试程序

// 生成随机数据
float* generate_random_data(size_t size) {
    float* data = (float*)malloc(size * sizeof(float));
    if (!data) return NULL;
    
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    return data;
}

// 简单的矩阵乘法（用于测试）
void matrix_multiply(const float* A, const float* B, float* C, 
                    size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// 简单的ReLU激活函数
void relu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0.0f;
    }
}

// 简单的Softmax激活函数
void softmax(float* data, size_t size) {
    float max_val = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

int main() {
    printf("=== 简单神经网络测试程序 ===\n");
    
    // 设置随机种子
    srand(42);
    
    // 生成测试数据
    size_t input_size = 10;
    size_t hidden_size = 5;
    size_t output_size = 2;
    size_t batch_size = 32;
    
    float* input_data = generate_random_data(batch_size * input_size);
    float* weights1 = generate_random_data(input_size * hidden_size);
    float* weights2 = generate_random_data(hidden_size * output_size);
    float* hidden_layer = (float*)malloc(batch_size * hidden_size * sizeof(float));
    float* output_layer = (float*)malloc(batch_size * output_size * sizeof(float));
    
    if (!input_data || !weights1 || !weights2 || !hidden_layer || !output_layer) {
        printf("错误：内存分配失败\n");
        return -1;
    }
    
    // 前向传播
    printf("执行前向传播...\n");
    
    // 第一层：输入 -> 隐藏层
    matrix_multiply(input_data, weights1, hidden_layer, batch_size, input_size, hidden_size);
    relu(hidden_layer, batch_size * hidden_size);
    
    // 第二层：隐藏层 -> 输出层
    matrix_multiply(hidden_layer, weights2, output_layer, batch_size, hidden_size, output_size);
    
    // 对每个样本应用softmax
    for (size_t i = 0; i < batch_size; i++) {
        softmax(output_layer + i * output_size, output_size);
    }
    
    printf("前向传播完成！\n");
    
    // 显示一些结果
    printf("前5个样本的输出：\n");
    for (size_t i = 0; i < 5 && i < batch_size; i++) {
        printf("样本 %zu: ", i);
        for (size_t j = 0; j < output_size; j++) {
            printf("%.4f ", output_layer[i * output_size + j]);
        }
        printf("\n");
    }
    
    // 清理内存
    free(input_data);
    free(weights1);
    free(weights2);
    free(hidden_layer);
    free(output_layer);
    
    printf("测试程序完成！\n");
    
    return 0;
}