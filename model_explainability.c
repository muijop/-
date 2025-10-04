#include "model_explainability.h"
#include "nn_module.h"
#include "tensor.h"
#include "autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// ===========================================
// 内部工具函数
// ===========================================

// 生成随机数（0-1）
static float random_float(void) {
    return (float)rand() / RAND_MAX;
}

// 生成高斯随机数
static float gaussian_random(float mean, float stddev) {
    float u1 = random_float();
    float u2 = random_float();
    
    while (u1 <= 0.0f) u1 = random_float();
    
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + stddev * z0;
}

// 计算向量的均值
static float vector_mean(const float* data, int size) {
    if (!data || size <= 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    
    return sum / size;
}

// 计算向量的标准差
static float vector_stddev(const float* data, int size) {
    if (!data || size <= 1) return 0.0f;
    
    float mean = vector_mean(data, size);
    float sum_sq = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    
    return sqrtf(sum_sq / (size - 1));
}

// 归一化向量（0-1）
static void normalize_vector(float* data, int size) {
    if (!data || size <= 0) return;
    
    float min_val = data[0];
    float max_val = data[0];
    
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    float range = max_val - min_val;
    if (range <= 0.0f) return;
    
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - min_val) / range;
    }
}

// ===========================================
// 解释管理器实现
// ===========================================

model_explainability_manager_t* create_model_explainability_manager(void) {
    model_explainability_manager_t* manager = 
        (model_explainability_manager_t*)calloc(1, sizeof(model_explainability_manager_t));
    
    if (!manager) {
        return NULL;
    }
    
    // 初始化默认配置
    manager->method = EXPLAIN_GRADIENT;
    manager->is_initialized = true;
    manager->explanation_id = 0;
    
    // 设置默认SHAP配置
    manager->shap_config.num_samples = 1000;
    manager->shap_config.max_features = 100;
    manager->shap_config.use_kernel_shap = true;
    manager->shap_config.use_tree_shap = false;
    manager->shap_config.background_data_ratio = 0.1f;
    
    // 设置默认LIME配置
    manager->lime_config.num_samples = 5000;
    manager->lime_config.num_features = 10;
    manager->lime_config.kernel_width = 0.75f;
    manager->lime_config.use_superpixels = true;
    manager->lime_config.superpixel_segments = 50;
    
    // 设置默认梯度配置
    manager->gradient_config.use_guided_backprop = true;
    manager->gradient_config.use_deconvnet = false;
    manager->gradient_config.use_smoothgrad = true;
    manager->gradient_config.smoothgrad_samples = 50;
    manager->gradient_config.smoothgrad_noise = 0.1f;
    
    // 设置默认注意力配置
    manager->attention_config.use_multi_head_attention = true;
    manager->attention_config.attention_heads = 8;
    manager->attention_config.normalize_attention = true;
    manager->attention_config.attention_threshold = 0.1f;
    
    // 设置默认公平性配置
    manager->fairness_config.sensitive_features = NULL;
    manager->fairness_config.num_sensitive_features = 0;
    manager->fairness_config.fairness_threshold = 0.8f;
    manager->fairness_config.check_disparate_impact = true;
    manager->fairness_config.check_equal_opportunity = true;
    manager->fairness_config.check_demographic_parity = true;
    
    // 初始化随机种子
    srand((unsigned int)time(NULL));
    
    return manager;
}

void destroy_model_explainability_manager(model_explainability_manager_t* manager) {
    if (manager) {
        // 释放敏感特征名称
        if (manager->fairness_config.sensitive_features) {
            for (int i = 0; i < manager->fairness_config.num_sensitive_features; i++) {
                free(manager->fairness_config.sensitive_features[i]);
            }
            free(manager->fairness_config.sensitive_features);
        }
        free(manager);
    }
}

// ===========================================
// 配置设置函数
// ===========================================

int set_explanation_method(model_explainability_manager_t* manager, explainability_method_t method) {
    if (!manager || !manager->is_initialized) {
        return -1;
    }
    
    manager->method = method;
    return 0;
}

int configure_shap(model_explainability_manager_t* manager, const shap_config_t* config) {
    if (!manager || !config || !manager->is_initialized) {
        return -1;
    }
    
    manager->shap_config = *config;
    return 0;
}

int configure_lime(model_explainability_manager_t* manager, const lime_config_t* config) {
    if (!manager || !config || !manager->is_initialized) {
        return -1;
    }
    
    manager->lime_config = *config;
    return 0;
}

int configure_gradient_explanation(model_explainability_manager_t* manager, const gradient_config_t* config) {
    if (!manager || !config || !manager->is_initialized) {
        return -1;
    }
    
    manager->gradient_config = *config;
    return 0;
}

int configure_attention_explanation(model_explainability_manager_t* manager, const attention_config_t* config) {
    if (!manager || !config || !manager->is_initialized) {
        return -1;
    }
    
    manager->attention_config = *config;
    return 0;
}

// ===========================================
// 梯度解释实现
// ===========================================

// 计算输入相对于输出的梯度
static tensor_t* compute_input_gradient(const nn_module_t* model, const tensor_t* input, int target_class) {
    if (!model || !input) {
        return NULL;
    }
    
    // 前向传播
    tensor_t* output = nn_module_forward(model, input);
    if (!output) {
        return NULL;
    }
    
    // 创建目标梯度（one-hot编码）
    tensor_t* target_grad = tensor_create(output->shape[0], output->shape[1]);
    if (!target_grad) {
        tensor_free(output);
        return NULL;
    }
    
    // 设置目标类的梯度为1，其他为0
    for (int i = 0; i < target_grad->shape[0] * target_grad->shape[1]; i++) {
        target_grad->data[i] = 0.0f;
    }
    
    if (target_class >= 0 && target_class < output->shape[1]) {
        target_grad->data[target_class] = 1.0f;
    }
    
    // 反向传播计算输入梯度
    tensor_t* input_grad = autograd_backward(model, input, target_grad);
    
    // 清理临时张量
    tensor_free(output);
    tensor_free(target_grad);
    
    return input_grad;
}

// 平滑梯度实现
static tensor_t* compute_smooth_gradient(const nn_module_t* model, const tensor_t* input, 
                                       int target_class, int num_samples, float noise_std) {
    if (!model || !input || num_samples <= 0) {
        return NULL;
    }
    
    tensor_t* avg_gradient = tensor_create(input->shape[0], input->shape[1]);
    if (!avg_gradient) {
        return NULL;
    }
    
    // 初始化平均梯度
    for (int i = 0; i < avg_gradient->shape[0] * avg_gradient->shape[1]; i++) {
        avg_gradient->data[i] = 0.0f;
    }
    
    // 多次采样计算平均梯度
    for (int sample = 0; sample < num_samples; sample++) {
        // 添加高斯噪声
        tensor_t* noisy_input = tensor_copy(input);
        if (!noisy_input) {
            tensor_free(avg_gradient);
            return NULL;
        }
        
        for (int i = 0; i < noisy_input->shape[0] * noisy_input->shape[1]; i++) {
            noisy_input->data[i] += gaussian_random(0.0f, noise_std);
        }
        
        // 计算梯度
        tensor_t* gradient = compute_input_gradient(model, noisy_input, target_class);
        
        if (gradient) {
            // 累加梯度
            for (int i = 0; i < gradient->shape[0] * gradient->shape[1]; i++) {
                avg_gradient->data[i] += gradient->data[i];
            }
            tensor_free(gradient);
        }
        
        tensor_free(noisy_input);
    }
    
    // 计算平均梯度
    for (int i = 0; i < avg_gradient->shape[0] * avg_gradient->shape[1]; i++) {
        avg_gradient->data[i] /= num_samples;
    }
    
    return avg_gradient;
}

// ===========================================
// SHAP值近似计算
// ===========================================

// 计算SHAP值的简化实现（核SHAP近似）
static float* compute_shap_values_approximate(const nn_module_t* model, const tensor_t* input, 
                                            int target_class, int num_samples) {
    if (!model || !input || num_samples <= 0) {
        return NULL;
    }
    
    int num_features = input->shape[0] * input->shape[1];
    float* shap_values = (float*)calloc(num_features, sizeof(float));
    if (!shap_values) {
        return NULL;
    }
    
    // 创建基线输入（均值）
    tensor_t* baseline = tensor_create(input->shape[0], input->shape[1]);
    if (!baseline) {
        free(shap_values);
        return NULL;
    }
    
    // 简单基线：零向量
    for (int i = 0; i < num_features; i++) {
        baseline->data[i] = 0.0f;
    }
    
    // 简化SHAP计算（实际实现需要更复杂的采样和加权）
    for (int sample = 0; sample < num_samples; sample++) {
        // 随机选择特征子集
        tensor_t* masked_input = tensor_copy(input);
        if (!masked_input) {
            free(shap_values);
            tensor_free(baseline);
            return NULL;
        }
        
        // 随机掩码特征
        for (int i = 0; i < num_features; i++) {
            if (random_float() < 0.5f) {
                masked_input->data[i] = baseline->data[i];
            }
        }
        
        // 计算预测
        tensor_t* output_with = nn_module_forward(model, masked_input);
        tensor_t* output_without = nn_module_forward(model, baseline);
        
        if (output_with && output_without) {
            float pred_with = target_class >= 0 ? output_with->data[target_class] : 0.0f;
            float pred_without = target_class >= 0 ? output_without->data[target_class] : 0.0f;
            
            // 简化SHAP值更新
            for (int i = 0; i < num_features; i++) {
                if (masked_input->data[i] != baseline->data[i]) {
                    shap_values[i] += (pred_with - pred_without) / num_samples;
                }
            }
        }
        
        if (output_with) tensor_free(output_with);
        if (output_without) tensor_free(output_without);
        tensor_free(masked_input);
    }
    
    tensor_free(baseline);
    return shap_values;
}

// ===========================================
// 主解释函数
// ===========================================

explanation_result_t* explain_model_prediction(model_explainability_manager_t* manager, 
                                             const nn_module_t* model, 
                                             const tensor_t* input, 
                                             int target_class) {
    if (!manager || !model || !input || !manager->is_initialized) {
        return NULL;
    }
    
    printf("开始模型解释，方法: %d，目标类: %d\n", manager->method, target_class);
    
    explanation_result_t* result = (explanation_result_t*)calloc(1, sizeof(explanation_result_t));
    if (!result) {
        return NULL;
    }
    
    int num_features = input->shape[0] * input->shape[1];
    result->num_features = num_features;
    result->feature_importance = (float*)calloc(num_features, sizeof(float));
    
    if (!result->feature_importance) {
        free(result);
        return NULL;
    }
    
    // 根据选择的方法计算解释
    switch (manager->method) {
        case EXPLAIN_GRADIENT:
            {
                tensor_t* gradient = NULL;
                
                if (manager->gradient_config.use_smoothgrad) {
                    gradient = compute_smooth_gradient(model, input, target_class,
                                                       manager->gradient_config.smoothgrad_samples,
                                                       manager->gradient_config.smoothgrad_noise);
                } else {
                    gradient = compute_input_gradient(model, input, target_class);
                }
                
                if (gradient) {
                    for (int i = 0; i < num_features; i++) {
                        result->feature_importance[i] = fabsf(gradient->data[i]);
                    }
                    tensor_free(gradient);
                }
            }
            break;
            
        case EXPLAIN_SHAP:
            {
                float* shap_values = compute_shap_values_approximate(model, input, target_class,
                                                                    manager->shap_config.num_samples);
                if (shap_values) {
                    for (int i = 0; i < num_features; i++) {
                        result->feature_importance[i] = fabsf(shap_values[i]);
                    }
                    free(shap_values);
                }
            }
            break;
            
        case EXPLAIN_SALIENCY:
            {
                // 显著性图：梯度的绝对值
                tensor_t* gradient = compute_input_gradient(model, input, target_class);
                if (gradient) {
                    for (int i = 0; i < num_features; i++) {
                        result->feature_importance[i] = fabsf(gradient->data[i]);
                    }
                    tensor_free(gradient);
                }
            }
            break;
            
        default:
            printf("暂不支持的解释方法: %d\n", manager->method);
            // 默认使用梯度方法
            tensor_t* gradient = compute_input_gradient(model, input, target_class);
            if (gradient) {
                for (int i = 0; i < num_features; i++) {
                    result->feature_importance[i] = fabsf(gradient->data[i]);
                }
                tensor_free(gradient);
            }
            break;
    }
    
    // 归一化重要性分数
    normalize_vector(result->feature_importance, num_features);
    
    // 计算置信度和保真度
    result->confidence_score = calculate_prediction_confidence(model, input);
    result->explanation_fidelity = calculate_explanation_fidelity(result, model, input);
    
    printf("模型解释完成，置信度: %.3f，保真度: %.3f\n", 
           result->confidence_score, result->explanation_fidelity);
    
    return result;
}

// ===========================================
// 公平性评估实现
// ===========================================

fairness_result_t* evaluate_model_fairness(model_explainability_manager_t* manager,
                                         const nn_module_t* model,
                                         const tensor_t* features,
                                         const tensor_t* labels,
                                         const tensor_t* sensitive_attributes) {
    if (!manager || !model || !features || !labels || !sensitive_attributes) {
        return NULL;
    }
    
    printf("开始模型公平性评估\n");
    
    fairness_result_t* result = (fairness_result_t*)calloc(1, sizeof(fairness_result_t));
    if (!result) {
        return NULL;
    }
    
    int num_samples = features->shape[0];
    
    // 简化公平性评估（实际实现需要更复杂的统计计算）
    
    // 计算各组准确率（简化：假设只有2组）
    result->num_groups = 2;
    result->group_accuracy = (float*)calloc(2, sizeof(float));
    
    if (result->group_accuracy) {
        int group1_correct = 0, group1_total = 0;
        int group2_correct = 0, group2_total = 0;
        
        for (int i = 0; i < num_samples; i++) {
            // 提取样本特征
            tensor_t* sample = tensor_create(1, features->shape[1]);
            if (sample) {
                for (int j = 0; j < features->shape[1]; j++) {
                    sample->data[j] = features->data[i * features->shape[1] + j];
                }
                
                // 预测
                tensor_t* prediction = nn_module_forward(model, sample);
                if (prediction) {
                    int pred_class = 0;
                    float max_prob = prediction->data[0];
                    
                    for (int j = 1; j < prediction->shape[1]; j++) {
                        if (prediction->data[j] > max_prob) {
                            max_prob = prediction->data[j];
                            pred_class = j;
                        }
                    }
                    
                    int true_class = (int)labels->data[i];
                    
                    // 根据敏感属性分组
                    int group = (int)sensitive_attributes->data[i];
                    
                    if (group == 0) {
                        group1_total++;
                        if (pred_class == true_class) group1_correct++;
                    } else {
                        group2_total++;
                        if (pred_class == true_class) group2_correct++;
                    }
                    
                    tensor_free(prediction);
                }
                tensor_free(sample);
            }
        }
        
        result->group_accuracy[0] = group1_total > 0 ? (float)group1_correct / group1_total : 0.0f;
        result->group_accuracy[1] = group2_total > 0 ? (float)group2_correct / group2_total : 0.0f;
        
        // 计算公平性指标
        if (group1_total > 0 && group2_total > 0) {
            // 差异影响比
            result->disparate_impact_ratio = result->group_accuracy[1] / result->group_accuracy[0];
            
            // 平等机会差异
            result->equal_opportunity_difference = fabsf(result->group_accuracy[0] - result->group_accuracy[1]);
            
            // 人口统计均等差异
            result->demographic_parity_difference = result->equal_opportunity_difference;
            
            // 判断是否公平
            result->is_fair = (result->disparate_impact_ratio >= manager->fairness_config.fairness_threshold) &&
                            (result->equal_opportunity_difference <= (1.0f - manager->fairness_config.fairness_threshold));
        }
    }
    
    printf("公平性评估完成，组1准确率: %.3f，组2准确率: %.3f，是否公平: %s\n",
           result->group_accuracy[0], result->group_accuracy[1], result->is_fair ? "是" : "否");
    
    return result;
}

// ===========================================
// 工具函数实现
// ===========================================

float calculate_prediction_confidence(const nn_module_t* model, const tensor_t* input) {
    if (!model || !input) {
        return 0.0f;
    }
    
    tensor_t* output = nn_module_forward(model, input);
    if (!output) {
        return 0.0f;
    }
    
    // 使用最大概率作为置信度
    float max_prob = output->data[0];
    for (int i = 1; i < output->shape[0] * output->shape[1]; i++) {
        if (output->data[i] > max_prob) {
            max_prob = output->data[i];
        }
    }
    
    tensor_free(output);
    return max_prob;
}

float calculate_explanation_fidelity(const explanation_result_t* explanation,
                                   const nn_module_t* model,
                                   const tensor_t* input) {
    if (!explanation || !model || !input) {
        return 0.0f;
    }
    
    // 简化保真度计算：使用最重要的特征进行预测
    // 实际实现需要更复杂的评估
    
    // 找到最重要的特征
    int max_importance_idx = 0;
    float max_importance = explanation->feature_importance[0];
    
    for (int i = 1; i < explanation->num_features; i++) {
        if (explanation->feature_importance[i] > max_importance) {
            max_importance = explanation->feature_importance[i];
            max_importance_idx = i;
        }
    }
    
    // 简化保真度：使用最重要特征的重要性分数
    return max_importance;
}

// ===========================================
// 内存管理函数
// ===========================================

void free_explanation_result(explanation_result_t* result) {
    if (result) {
        if (result->feature_importance) {
            free(result->feature_importance);
        }
        if (result->feature_names) {
            for (int i = 0; i < result->num_features; i++) {
                free(result->feature_names[i]);
            }
            free(result->feature_names);
        }
        if (result->baseline_values) {
            free(result->baseline_values);
        }
        free(result);
    }
}

void free_fairness_result(fairness_result_t* result) {
    if (result) {
        if (result->group_accuracy) {
            free(result->group_accuracy);
        }
        free(result);
    }
}

// ===========================================
// 回调函数设置
// ===========================================

void set_explanation_progress_callback(model_explainability_manager_t* manager,
                                      void (*callback)(int progress, const char* message)) {
    if (manager) {
        manager->progress_callback = callback;
    }
}

void set_explanation_ready_callback(model_explainability_manager_t* manager,
                                   void (*callback)(const explanation_result_t* result)) {
    if (manager) {
        manager->explanation_ready_callback = callback;
    }
}

void set_fairness_ready_callback(model_explainability_manager_t* manager,
                                void (*callback)(const fairness_result_t* result)) {
    if (manager) {
        manager->fairness_ready_callback = callback;
    }
}