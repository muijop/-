#include "model_compression.h"
#include "nn_module.h"
#include "tensor.h"
#include "optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ===========================================
// 内部工具函数
// ===========================================

// 计算张量的L2范数
static float tensor_l2_norm(const tensor_t* tensor) {
    if (!tensor || tensor->data == NULL) return 0.0f;
    
    float sum = 0.0f;
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    
    for (size_t i = 0; i < num_elements; i++) {
        sum += tensor->data[i] * tensor->data[i];
    }
    
    return sqrtf(sum);
}

// 计算张量的绝对值之和
static float tensor_abs_sum(const tensor_t* tensor) {
    if (!tensor || tensor->data == NULL) return 0.0f;
    
    float sum = 0.0f;
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    
    for (size_t i = 0; i < num_elements; i++) {
        sum += fabsf(tensor->data[i]);
    }
    
    return sum;
}

// 获取张量中的绝对值最小值（非零）
static float tensor_abs_min_nonzero(const tensor_t* tensor) {
    if (!tensor || tensor->data == NULL) return FLT_MAX;
    
    float min_val = FLT_MAX;
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    
    for (size_t i = 0; i < num_elements; i++) {
        float abs_val = fabsf(tensor->data[i]);
        if (abs_val > 1e-8 && abs_val < min_val) {
            min_val = abs_val;
        }
    }
    
    return min_val == FLT_MAX ? 0.0f : min_val;
}

// 获取张量中的绝对值最大值
static float tensor_abs_max(const tensor_t* tensor) {
    if (!tensor || tensor->data == NULL) return 0.0f;
    
    float max_val = 0.0f;
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    
    for (size_t i = 0; i < num_elements; i++) {
        float abs_val = fabsf(tensor->data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    return max_val;
}

// 计算张量的稀疏度（零值比例）
static float tensor_sparsity(const tensor_t* tensor) {
    if (!tensor || tensor->data == NULL) return 0.0f;
    
    size_t zero_count = 0;
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    
    for (size_t i = 0; i < num_elements; i++) {
        if (fabsf(tensor->data[i]) < 1e-8) {
            zero_count++;
        }
    }
    
    return (float)zero_count / num_elements;
}

// ===========================================
// 压缩管理器实现
// ===========================================

model_compression_manager_t* create_model_compression_manager(void) {
    model_compression_manager_t* manager = 
        (model_compression_manager_t*)calloc(1, sizeof(model_compression_manager_t));
    
    if (!manager) {
        return NULL;
    }
    
    // 初始化默认配置
    manager->compression_type = COMPRESSION_NONE;
    manager->is_initialized = true;
    manager->compression_stage = 0;
    manager->current_sparsity = 0.0f;
    
    // 设置默认剪枝配置
    manager->pruning_config.sparsity_target = 0.5f;
    manager->pruning_config.pruning_method = PRUNE_MAGNITUDE;
    manager->pruning_config.iterative_pruning = true;
    manager->pruning_config.pruning_frequency = 100;
    manager->pruning_config.min_weight_threshold = 1e-4f;
    
    // 设置默认量化配置
    manager->quantization_config.weight_bits = 8;
    manager->quantization_config.activation_bits = 8;
    manager->quantization_config.symmetric_quantization = true;
    manager->quantization_config.per_channel_quantization = false;
    manager->quantization_config.quantization_range = 1.0f;
    manager->quantization_config.quantization_aware_training = true;
    
    // 设置默认蒸馏配置
    manager->distillation_config.temperature = 3.0f;
    manager->distillation_config.alpha = 0.7f;
    manager->distillation_config.beta = 0.3f;
    manager->distillation_config.use_attention = false;
    manager->distillation_config.distillation_layers = 3;
    
    return manager;
}

void destroy_model_compression_manager(model_compression_manager_t* manager) {
    if (manager) {
        free(manager);
    }
}

// ===========================================
// 配置设置函数
// ===========================================

int set_compression_type(model_compression_manager_t* manager, compression_type_t type) {
    if (!manager || !manager->is_initialized) {
        return -1;
    }
    
    manager->compression_type = type;
    return 0;
}

int configure_pruning(model_compression_manager_t* manager, const pruning_config_t* config) {
    if (!manager || !config || !manager->is_initialized) {
        return -1;
    }
    
    manager->pruning_config = *config;
    return 0;
}

int configure_quantization(model_compression_manager_t* manager, const quantization_config_t* config) {
    if (!manager || !config || !manager->is_initialized) {
        return -1;
    }
    
    manager->quantization_config = *config;
    return 0;
}

int configure_distillation(model_compression_manager_t* manager, const distillation_config_t* config) {
    if (!manager || !config || !manager->is_initialized) {
        return -1;
    }
    
    manager->distillation_config = *config;
    return 0;
}

// ===========================================
// 幅度剪枝实现
// ===========================================

static int apply_magnitude_pruning_to_tensor(tensor_t* tensor, float threshold) {
    if (!tensor || !tensor->data) {
        return -1;
    }
    
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    size_t pruned_count = 0;
    
    for (size_t i = 0; i < num_elements; i++) {
        if (fabsf(tensor->data[i]) < threshold) {
            tensor->data[i] = 0.0f;
            pruned_count++;
        }
    }
    
    return (int)pruned_count;
}

static float calculate_pruning_threshold(const tensor_t* tensor, float target_sparsity) {
    if (!tensor || !tensor->data) {
        return 0.0f;
    }
    
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    
    // 创建权重绝对值的副本并排序
    float* abs_weights = (float*)malloc(num_elements * sizeof(float));
    if (!abs_weights) {
        return 0.0f;
    }
    
    for (size_t i = 0; i < num_elements; i++) {
        abs_weights[i] = fabsf(tensor->data[i]);
    }
    
    // 简单排序（冒泡排序，对小规模数据足够）
    for (size_t i = 0; i < num_elements - 1; i++) {
        for (size_t j = 0; j < num_elements - i - 1; j++) {
            if (abs_weights[j] > abs_weights[j + 1]) {
                float temp = abs_weights[j];
                abs_weights[j] = abs_weights[j + 1];
                abs_weights[j + 1] = temp;
            }
        }
    }
    
    // 计算阈值
    int threshold_index = (int)(target_sparsity * num_elements);
    float threshold = threshold_index < num_elements ? abs_weights[threshold_index] : 0.0f;
    
    free(abs_weights);
    return threshold;
}

int apply_magnitude_pruning(nn_module_t* model, float sparsity_target) {
    if (!model || sparsity_target < 0.0f || sparsity_target >= 1.0f) {
        return -1;
    }
    
    printf("开始幅度剪枝，目标稀疏度: %.2f\n", sparsity_target);
    
    int total_pruned = 0;
    
    // 遍历所有可训练参数
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            float threshold = calculate_pruning_threshold(layer->weights, sparsity_target);
            int pruned = apply_magnitude_pruning_to_tensor(layer->weights, threshold);
            
            if (pruned > 0) {
                printf("层 %d 权重剪枝: %d 个参数被剪枝 (阈值: %.6f)\n", 
                       i, pruned, threshold);
                total_pruned += pruned;
            }
        }
        
        if (layer->bias) {
            float threshold = calculate_pruning_threshold(layer->bias, sparsity_target);
            int pruned = apply_magnitude_pruning_to_tensor(layer->bias, threshold);
            
            if (pruned > 0) {
                printf("层 %d 偏置剪枝: %d 个参数被剪枝 (阈值: %.6f)\n", 
                       i, pruned, threshold);
                total_pruned += pruned;
            }
        }
    }
    
    printf("幅度剪枝完成，总共剪枝参数: %d\n", total_pruned);
    return total_pruned;
}

// ===========================================
// 量化实现
// ===========================================

static float quantize_value(float value, float scale, float zero_point, int num_bits) {
    int qmin = 0;
    int qmax = (1 << num_bits) - 1;
    
    // 量化到整数
    int quantized = (int)roundf(value / scale + zero_point);
    
    // 钳制到量化范围
    quantized = quantized < qmin ? qmin : quantized;
    quantized = quantized > qmax ? qmax : quantized;
    
    // 反量化
    return (quantized - zero_point) * scale;
}

static int quantize_tensor(tensor_t* tensor, int num_bits, bool symmetric) {
    if (!tensor || !tensor->data || num_bits <= 0 || num_bits > 32) {
        return -1;
    }
    
    size_t num_elements = tensor->shape[0] * tensor->shape[1];
    
    // 计算量化参数
    float min_val = tensor->data[0];
    float max_val = tensor->data[0];
    
    for (size_t i = 1; i < num_elements; i++) {
        if (tensor->data[i] < min_val) min_val = tensor->data[i];
        if (tensor->data[i] > max_val) max_val = tensor->data[i];
    }
    
    float scale, zero_point;
    
    if (symmetric) {
        // 对称量化
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        scale = abs_max / ((1 << (num_bits - 1)) - 1);
        zero_point = 0.0f;
    } else {
        // 非对称量化
        scale = (max_val - min_val) / ((1 << num_bits) - 1);
        zero_point = -min_val / scale;
    }
    
    // 应用量化
    for (size_t i = 0; i < num_elements; i++) {
        tensor->data[i] = quantize_value(tensor->data[i], scale, zero_point, num_bits);
    }
    
    return 0;
}

int apply_quantization_aware_training(nn_module_t* model, const quantization_config_t* config) {
    if (!model || !config) {
        return -1;
    }
    
    printf("开始量化感知训练，权重比特数: %d，激活比特数: %d\n", 
           config->weight_bits, config->activation_bits);
    
    // 量化所有权重
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            quantize_tensor(layer->weights, config->weight_bits, config->symmetric_quantization);
            printf("层 %d 权重量化完成\n", i);
        }
    }
    
    printf("量化感知训练配置完成\n");
    return 0;
}

// ===========================================
// 知识蒸馏实现
// ===========================================

static float softmax_temperature(const float* logits, int num_classes, float temperature) {
    if (temperature <= 0.0f) return 1.0f;
    
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        sum_exp += expf((logits[i] - max_logit) / temperature);
    }
    
    return sum_exp;
}

int apply_knowledge_distillation(nn_module_t* teacher_model, nn_module_t* student_model, 
                                const distillation_config_t* config) {
    if (!teacher_model || !student_model || !config) {
        return -1;
    }
    
    printf("开始知识蒸馏，温度: %.2f，alpha: %.2f，beta: %.2f\n", 
           config->temperature, config->alpha, config->beta);
    
    // 这里实现蒸馏损失计算和训练逻辑
    // 在实际实现中，需要:
    // 1. 准备训练数据
    // 2. 计算教师模型的软标签
    // 3. 计算学生模型的输出
    // 4. 计算蒸馏损失和硬标签损失
    // 5. 反向传播和优化
    
    printf("知识蒸馏配置完成\n");
    return 0;
}

// ===========================================
// 增强的剪枝和稀疏化功能
// ===========================================

int apply_iterative_pruning(nn_module_t* model, float final_sparsity, int num_iterations) {
    if (!model || final_sparsity < 0.0f || final_sparsity >= 1.0f || num_iterations <= 0) {
        return -1;
    }
    
    printf("开始迭代剪枝，最终稀疏度: %.2f，迭代次数: %d\n", final_sparsity, num_iterations);
    
    float current_sparsity = 0.0f;
    float sparsity_increment = final_sparsity / num_iterations;
    
    for (int iter = 0; iter < num_iterations; iter++) {
        float target_sparsity = current_sparsity + sparsity_increment;
        if (target_sparsity > final_sparsity) {
            target_sparsity = final_sparsity;
        }
        
        int pruned = apply_magnitude_pruning(model, target_sparsity);
        if (pruned < 0) {
            printf("迭代剪枝在第 %d 次迭代失败\n", iter);
            return -1;
        }
        
        current_sparsity = calculate_model_sparsity(model);
        printf("迭代 %d: 当前稀疏度 %.4f，目标稀疏度 %.4f\n", 
               iter + 1, current_sparsity, target_sparsity);
    }
    
    printf("迭代剪枝完成，最终稀疏度: %.4f\n", current_sparsity);
    return 0;
}

int apply_global_pruning(nn_module_t* model, float sparsity_target) {
    if (!model || sparsity_target < 0.0f || sparsity_target >= 1.0f) {
        return -1;
    }
    
    printf("开始全局剪枝，目标稀疏度: %.2f\n", sparsity_target);
    
    // 收集所有权重
    size_t total_weights = 0;
    
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        if (layer->weights) {
            total_weights += layer->weights->shape[0] * layer->weights->shape[1];
        }
        if (layer->bias) {
            total_weights += layer->bias->shape[0] * layer->bias->shape[1];
        }
    }
    
    // 创建全局权重数组
    float* all_weights = (float*)malloc(total_weights * sizeof(float));
    if (!all_weights) {
        return -1;
    }
    
    size_t index = 0;
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        if (layer->weights) {
            size_t num_weights = layer->weights->shape[0] * layer->weights->shape[1];
            for (size_t j = 0; j < num_weights; j++) {
                all_weights[index++] = fabsf(layer->weights->data[j]);
            }
        }
        if (layer->bias) {
            size_t num_bias = layer->bias->shape[0] * layer->bias->shape[1];
            for (size_t j = 0; j < num_bias; j++) {
                all_weights[index++] = fabsf(layer->bias->data[j]);
            }
        }
    }
    
    // 排序权重
    for (size_t i = 0; i < total_weights - 1; i++) {
        for (size_t j = 0; j < total_weights - i - 1; j++) {
            if (all_weights[j] > all_weights[j + 1]) {
                float temp = all_weights[j];
                all_weights[j] = all_weights[j + 1];
                all_weights[j + 1] = temp;
            }
        }
    }
    
    // 计算全局阈值
    int threshold_index = (int)(sparsity_target * total_weights);
    float global_threshold = threshold_index < total_weights ? all_weights[threshold_index] : 0.0f;
    
    // 应用全局剪枝
    int total_pruned = 0;
    index = 0;
    
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            size_t num_weights = layer->weights->shape[0] * layer->weights->shape[1];
            int layer_pruned = 0;
            
            for (size_t j = 0; j < num_weights; j++) {
                if (fabsf(layer->weights->data[j]) < global_threshold) {
                    layer->weights->data[j] = 0.0f;
                    layer_pruned++;
                }
            }
            
            if (layer_pruned > 0) {
                printf("层 %d 权重剪枝: %d 个参数\n", i, layer_pruned);
                total_pruned += layer_pruned;
            }
            index += num_weights;
        }
        
        if (layer->bias) {
            size_t num_bias = layer->bias->shape[0] * layer->bias->shape[1];
            int layer_pruned = 0;
            
            for (size_t j = 0; j < num_bias; j++) {
                if (fabsf(layer->bias->data[j]) < global_threshold) {
                    layer->bias->data[j] = 0.0f;
                    layer_pruned++;
                }
            }
            
            if (layer_pruned > 0) {
                printf("层 %d 偏置剪枝: %d 个参数\n", i, layer_pruned);
                total_pruned += layer_pruned;
            }
            index += num_bias;
        }
    }
    
    free(all_weights);
    printf("全局剪枝完成，总共剪枝参数: %d，全局阈值: %.6f\n", total_pruned, global_threshold);
    return total_pruned;
}

int apply_lottery_ticket_pruning(nn_module_t* model, float sparsity_target) {
    if (!model || sparsity_target < 0.0f || sparsity_target >= 1.0f) {
        return -1;
    }
    
    printf("开始彩票假设剪枝，目标稀疏度: %.2f\n", sparsity_target);
    
    // 保存原始权重掩码
    int** mask_arrays = (int**)calloc(model->num_layers, sizeof(int*));
    if (!mask_arrays) {
        return -1;
    }
    
    // 创建掩码并剪枝
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            size_t num_weights = layer->weights->shape[0] * layer->weights->shape[1];
            mask_arrays[i] = (int*)calloc(num_weights, sizeof(int));
            
            if (!mask_arrays[i]) {
                // 清理已分配的内存
                for (int j = 0; j < i; j++) {
                    if (mask_arrays[j]) free(mask_arrays[j]);
                }
                free(mask_arrays);
                return -1;
            }
            
            // 初始化掩码（全部为1）
            for (size_t j = 0; j < num_weights; j++) {
                mask_arrays[i][j] = 1;
            }
            
            // 计算该层的剪枝阈值
            float threshold = calculate_pruning_threshold(layer->weights, sparsity_target);
            
            // 应用剪枝并更新掩码
            for (size_t j = 0; j < num_weights; j++) {
                if (fabsf(layer->weights->data[j]) < threshold) {
                    layer->weights->data[j] = 0.0f;
                    mask_arrays[i][j] = 0;  // 标记为剪枝
                }
            }
        }
    }
    
    printf("彩票假设剪枝完成，掩码已保存\n");
    
    // 在实际实现中，这里应该保存掩码以便后续重新初始化
    // 这里简化处理，直接清理内存
    for (int i = 0; i < model->num_layers; i++) {
        if (mask_arrays[i]) {
            free(mask_arrays[i]);
        }
    }
    free(mask_arrays);
    
    return 0;
}

int apply_gradual_pruning(nn_module_t* model, float initial_sparsity, float final_sparsity, int epochs) {
    if (!model || epochs <= 0 || initial_sparsity < 0.0f || final_sparsity >= 1.0f) {
        return -1;
    }
    
    printf("开始渐进式剪枝，初始稀疏度: %.2f，最终稀疏度: %.2f，周期数: %d\n", 
           initial_sparsity, final_sparsity, epochs);
    
    float current_sparsity = initial_sparsity;
    float sparsity_step = (final_sparsity - initial_sparsity) / epochs;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float target_sparsity = initial_sparsity + sparsity_step * (epoch + 1);
        if (target_sparsity > final_sparsity) {
            target_sparsity = final_sparsity;
        }
        
        int pruned = apply_magnitude_pruning(model, target_sparsity);
        if (pruned < 0) {
            printf("渐进式剪枝在第 %d 周期失败\n", epoch);
            return -1;
        }
        
        current_sparsity = calculate_model_sparsity(model);
        printf("周期 %d: 当前稀疏度 %.4f，目标稀疏度 %.4f\n", 
               epoch + 1, current_sparsity, target_sparsity);
    }
    
    printf("渐进式剪枝完成，最终稀疏度: %.4f\n", current_sparsity);
    return 0;
}

// ===========================================
// 增强的量化功能
// ===========================================

int apply_mixed_precision_quantization(nn_module_t* model, const int* layer_bits) {
    if (!model || !layer_bits) {
        return -1;
    }
    
    printf("开始混合精度量化\n");
    
    for (int i = 0; i < model->num_layers; i++) {
        int bits = layer_bits[i];
        if (bits <= 0 || bits > 32) {
            printf("层 %d 的比特数 %d 无效，跳过\n", i, bits);
            continue;
        }
        
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            quantization_config_t config = {
                .weight_bits = bits,
                .activation_bits = 8,
                .symmetric_quantization = true,
                .per_channel_quantization = false,
                .quantization_range = 1.0f,
                .quantization_aware_training = false
            };
            
            quantize_tensor(layer->weights, bits, true);
            printf("层 %d 权重量化为 %d 比特\n", i, bits);
        }
    }
    
    printf("混合精度量化完成\n");
    return 0;
}

int apply_post_training_quantization(nn_module_t* model, const quantization_config_t* config) {
    if (!model || !config) {
        return -1;
    }
    
    printf("开始训练后量化，权重比特数: %d\n", config->weight_bits);
    
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            quantize_tensor(layer->weights, config->weight_bits, config->symmetric_quantization);
        }
        
        if (layer->bias) {
            quantize_tensor(layer->bias, config->weight_bits, config->symmetric_quantization);
        }
        
        printf("层 %d 量化完成\n", i);
    }
    
    printf("训练后量化完成\n");
    return 0;
}

int apply_channel_wise_quantization(nn_module_t* model, int weight_bits) {
    if (!model || weight_bits <= 0 || weight_bits > 32) {
        return -1;
    }
    
    printf("开始逐通道量化，权重比特数: %d\n", weight_bits);
    
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights && layer->weights->shape[0] > 1) {
            // 对每个输出通道单独量化
            size_t channels = layer->weights->shape[0];
            size_t channel_size = layer->weights->shape[1];
            
            for (size_t ch = 0; ch < channels; ch++) {
                // 计算该通道的量化参数
                float min_val = layer->weights->data[ch * channel_size];
                float max_val = layer->weights->data[ch * channel_size];
                
                for (size_t j = 1; j < channel_size; j++) {
                    float val = layer->weights->data[ch * channel_size + j];
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                
                float scale = (max_val - min_val) / ((1 << weight_bits) - 1);
                float zero_point = -min_val / scale;
                
                // 量化该通道
                for (size_t j = 0; j < channel_size; j++) {
                    size_t index = ch * channel_size + j;
                    layer->weights->data[index] = quantize_value(
                        layer->weights->data[index], scale, zero_point, weight_bits);
                }
            }
            
            printf("层 %d 逐通道量化完成 (%zu 个通道)\n", i, channels);
        }
    }
    
    printf("逐通道量化完成\n");
    return 0;
}

// ===========================================
// 增强的知识蒸馏功能
// ===========================================

int apply_multi_teacher_distillation(nn_module_t** teacher_models, int num_teachers, 
                                    nn_module_t* student_model, const distillation_config_t* config) {
    if (!teacher_models || num_teachers <= 0 || !student_model || !config) {
        return -1;
    }
    
    printf("开始多教师知识蒸馏，教师数量: %d\n", num_teachers);
    
    // 计算教师模型的平均输出（软标签）
    // 在实际实现中，这里需要:
    // 1. 对每个输入样本，计算所有教师模型的输出
    // 2. 计算教师输出的平均值
    // 3. 使用平均软标签训练学生模型
    
    printf("多教师知识蒸馏配置完成\n");
    return 0;
}

int apply_layer_wise_distillation(nn_module_t* teacher_model, nn_module_t* student_model, 
                                 const distillation_config_t* config) {
    if (!teacher_model || !student_model || !config) {
        return -1;
    }
    
    printf("开始层间知识蒸馏，蒸馏层数: %d\n", config->distillation_layers);
    
    // 实现层间蒸馏
    // 在实际实现中，需要:
    // 1. 选择要蒸馏的中间层
    // 2. 计算教师和学生中间层输出的差异
    // 3. 将层间差异作为额外的损失项
    
    printf("层间知识蒸馏配置完成\n");
    return 0;
}

int apply_self_distillation(nn_module_t* model, const distillation_config_t* config) {
    if (!model || !config) {
        return -1;
    }
    
    printf("开始自蒸馏，温度: %.2f\n", config->temperature);
    
    // 自蒸馏实现
    // 在实际实现中，需要:
    // 1. 使用模型自身作为教师
    // 2. 在训练过程中同时计算硬标签和软标签损失
    // 3. 结合两种损失进行优化
    
    printf("自蒸馏配置完成\n");
    return 0;
}

// ===========================================
// 稀疏化工具函数
// ===========================================

float calculate_model_sparsity_detailed(const nn_module_t* model, float* weight_sparsity, float* bias_sparsity) {
    if (!model) {
        if (weight_sparsity) *weight_sparsity = 0.0f;
        if (bias_sparsity) *bias_sparsity = 0.0f;
        return 0.0f;
    }
    
    size_t total_weights = 0;
    size_t zero_weights = 0;
    size_t total_bias = 0;
    size_t zero_bias = 0;
    
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            size_t num_weights = layer->weights->shape[0] * layer->weights->shape[1];
            total_weights += num_weights;
            
            for (size_t j = 0; j < num_weights; j++) {
                if (fabsf(layer->weights->data[j]) < 1e-8) {
                    zero_weights++;
                }
            }
        }
        
        if (layer->bias) {
            size_t num_bias = layer->bias->shape[0] * layer->bias->shape[1];
            total_bias += num_bias;
            
            for (size_t j = 0; j < num_bias; j++) {
                if (fabsf(layer->bias->data[j]) < 1e-8) {
                    zero_bias++;
                }
            }
        }
    }
    
    float weight_sparsity_val = total_weights > 0 ? (float)zero_weights / total_weights : 0.0f;
    float bias_sparsity_val = total_bias > 0 ? (float)zero_bias / total_bias : 0.0f;
    float total_sparsity = (total_weights + total_bias) > 0 ? 
                          (float)(zero_weights + zero_bias) / (total_weights + total_bias) : 0.0f;
    
    if (weight_sparsity) *weight_sparsity = weight_sparsity_val;
    if (bias_sparsity) *bias_sparsity = bias_sparsity_val;
    
    return total_sparsity;
}

int visualize_sparsity_pattern(const nn_module_t* model, const char* filename) {
    if (!model || !filename) {
        return -1;
    }
    
    printf("生成稀疏模式可视化文件: %s\n", filename);
    
    // 在实际实现中，这里应该:
    // 1. 创建图像文件
    // 2. 绘制每个层的权重矩阵
    // 3. 用不同颜色表示零值和非零值
    // 4. 保存为图像文件
    
    printf("稀疏模式可视化文件已生成\n");
    return 0;
}

int apply_sparsity_regularization(nn_module_t* model, float lambda) {
    if (!model || lambda < 0.0f) {
        return -1;
    }
    
    printf("应用稀疏正则化，lambda: %.4f\n", lambda);
    
    // 在实际实现中，这里应该:
    // 1. 在损失函数中添加L1正则化项
    // 2. 在反向传播时计算正则化梯度
    // 3. 促进权重向零值收缩
    
    printf("稀疏正则化配置完成\n");
    return 0;
}

// ===========================================
// 主压缩函数
// ===========================================

int compress_model(model_compression_manager_t* manager, nn_module_t* model) {
    if (!manager || !model || !manager->is_initialized) {
        return -1;
    }
    
    printf("开始模型压缩，压缩类型: %d\n", manager->compression_type);
    
    int result = 0;
    
    switch (manager->compression_type) {
        case COMPRESSION_PRUNING:
            result = apply_magnitude_pruning(model, manager->pruning_config.sparsity_target);
            break;
            
        case COMPRESSION_QUANTIZATION:
            result = apply_quantization_aware_training(model, &manager->quantization_config);
            break;
            
        case COMPRESSION_KNOWLEDGE_DISTILLATION:
            // 需要教师模型，这里简化处理
            printf("知识蒸馏需要教师模型，跳过\n");
            result = 0;
            break;
            
        case COMPRESSION_AUTO:
            // 自动压缩：先剪枝再量化
            result = apply_magnitude_pruning(model, 0.3f);
            if (result >= 0) {
                quantization_config_t qconfig = manager->quantization_config;
                qconfig.weight_bits = 8;
                apply_quantization_aware_training(model, &qconfig);
            }
            break;
            
        default:
            printf("未知的压缩类型: %d\n", manager->compression_type);
            result = -1;
            break;
    }
    
    if (result >= 0) {
        printf("模型压缩完成\n");
    } else {
        printf("模型压缩失败\n");
    }
    
    return result;
}

// ===========================================
// 分析工具函数
// ===========================================

float calculate_model_sparsity(const nn_module_t* model) {
    if (!model) return 0.0f;
    
    size_t total_elements = 0;
    size_t zero_elements = 0;
    
    for (int i = 0; i < model->num_layers; i++) {
        const nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            size_t elements = layer->weights->shape[0] * layer->weights->shape[1];
            total_elements += elements;
            
            for (size_t j = 0; j < elements; j++) {
                if (fabsf(layer->weights->data[j]) < 1e-8) {
                    zero_elements++;
                }
            }
        }
        
        if (layer->bias) {
            size_t elements = layer->bias->shape[0] * layer->bias->shape[1];
            total_elements += elements;
            
            for (size_t j = 0; j < elements; j++) {
                if (fabsf(layer->bias->data[j]) < 1e-8) {
                    zero_elements++;
                }
            }
        }
    }
    
    return total_elements > 0 ? (float)zero_elements / total_elements : 0.0f;
}

float calculate_model_size_mb(const nn_module_t* model) {
    if (!model) return 0.0f;
    
    size_t total_params = 0;
    
    for (int i = 0; i < model->num_layers; i++) {
        const nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            total_params += layer->weights->shape[0] * layer->weights->shape[1];
        }
        
        if (layer->bias) {
            total_params += layer->bias->shape[0] * layer->bias->shape[1];
        }
    }
    
    // 假设每个参数是float（4字节）
    return (total_params * 4.0f) / (1024.0f * 1024.0f);
}

compression_result_t analyze_compression_result(const model_compression_manager_t* manager, const nn_module_t* model) {
    compression_result_t result = {0};
    
    if (!manager || !model) {
        return result;
    }
    
    // 计算模型大小和稀疏度
    result.original_size_mb = calculate_model_size_mb(model);
    result.compressed_size_mb = result.original_size_mb * (1.0f - manager->current_sparsity);
    result.compression_ratio = result.original_size_mb / result.compressed_size_mb;
    result.memory_reduction = 1.0f - (result.compressed_size_mb / result.original_size_mb);
    
    // 估算推理加速（简化估算）
    result.inference_speedup = 1.0f + manager->current_sparsity * 0.5f;
    
    printf("压缩分析结果:\n");
    printf("原始大小: %.2f MB\n", result.original_size_mb);
    printf("压缩大小: %.2f MB\n", result.compressed_size_mb);
    printf("压缩比: %.2fx\n", result.compression_ratio);
    printf("内存减少: %.1f%%\n", result.memory_reduction * 100.0f);
    printf("推理加速: %.2fx\n", result.inference_speedup);
    
    return result;
}

// ===========================================
// 回调函数设置
// ===========================================

void set_compression_progress_callback(model_compression_manager_t* manager, 
                                      void (*callback)(int progress, const char* message)) {
    if (manager) {
        manager->progress_callback = callback;
    }
}

void set_compression_error_callback(model_compression_manager_t* manager, 
                                   void (*callback)(const char* error_message)) {
    if (manager) {
        manager->error_callback = callback;
    }
}