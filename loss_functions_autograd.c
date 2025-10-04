#include "loss_functions_autograd.h"
#include "kernels.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// 损失函数注册表
typedef struct LossRegistry {
    const char* name;
    AutogradLossFunction* (*create_func)(void);
    struct LossRegistry* next;
} LossRegistry;

static LossRegistry* g_loss_registry = NULL;

// 内部辅助函数
static float safe_log(float x);
static float safe_exp(float x);
static float calculate_mean(const float* data, int size);
static float calculate_sum(const float* data, int size);
static int argmax(const float* data, int size);

// 通用损失函数操作
AutogradTensor* autograd_apply_reduction(AutogradTensor* loss, bool reduction_mean, bool reduction_sum) {
    if (!loss) return NULL;
    
    if (reduction_mean) {
        return autograd_tensor_mean(loss, -1, true);
    } else if (reduction_sum) {
        return autograd_tensor_sum(loss, -1, true);
    } else {
        // 无降维，返回原始损失
        return autograd_tensor_clone(loss);
    }
}

AutogradTensor* autograd_apply_class_weighting(AutogradTensor* loss, AutogradTensor* weight, AutogradTensor* target) {
    if (!loss || !weight) return autograd_tensor_clone(loss);
    
    // 根据目标类别应用权重
    // 这里简化实现，实际应该根据target的类别索引来应用权重
    return autograd_tensor_multiply(loss, weight);
}

// MSE损失函数实现
AutogradMSELoss* autograd_mse_loss_create(bool reduction_mean) {
    AutogradMSELoss* loss = (AutogradMSELoss*)malloc(sizeof(AutogradMSELoss));
    if (!loss) return NULL;
    
    loss->base.type = LOSS_MSE;
    loss->base.config.reduction_mean = reduction_mean;
    loss->base.config.reduction_sum = !reduction_mean;
    loss->base.forward = autograd_mse_loss_forward;
    loss->base.backward = autograd_mse_loss_backward;
    loss->base.destroy = autograd_mse_loss_destroy;
    
    return loss;
}

void autograd_mse_loss_destroy(AutogradLossFunction* base) {
    AutogradMSELoss* loss = (AutogradMSELoss*)base;
    if (loss) {
        free(loss);
    }
}

AutogradTensor* autograd_mse_loss_forward(AutogradLossFunction* base, AutogradTensor* input, AutogradTensor* target) {
    if (!base || !input || !target) return NULL;
    
    // 计算平方差: (input - target)^2
    AutogradTensor* diff = autograd_tensor_subtract(input, target);
    AutogradTensor* squared = autograd_tensor_power(diff, 2.0f);
    
    // 应用降维
    AutogradTensor* result = autograd_apply_reduction(squared, base->config.reduction_mean, base->config.reduction_sum);
    
    // 清理中间结果
    autograd_tensor_destroy(diff);
    autograd_tensor_destroy(squared);
    
    return result;
}

void autograd_mse_loss_backward(AutogradLossFunction* base, AutogradTensor* grad_output,
                               AutogradTensor* input, AutogradTensor* target, AutogradTensor** grad_input) {
    if (!base || !input || !target || !grad_output || !grad_input) return;
    
    // MSE损失的梯度: 2 * (input - target) / n
    AutogradTensor* diff = autograd_tensor_subtract(input, target);
    AutogradTensor* grad_scale = autograd_tensor_multiply_scalar(diff, 2.0f);
    
    if (base->config.reduction_mean) {
        int input_size = autograd_tensor_size(input);
        *grad_input = autograd_tensor_divide_scalar(grad_scale, (float)input_size);
    } else {
        *grad_input = autograd_tensor_clone(grad_scale);
    }
    
    // 应用grad_output缩放
    if (grad_output && autograd_tensor_size(grad_output) == 1) {
        float scale = autograd_tensor_data_float(grad_output)[0];
        AutogradTensor* scaled = autograd_tensor_multiply_scalar(*grad_input, scale);
        autograd_tensor_destroy(*grad_input);
        *grad_input = scaled;
    }
    
    // 清理中间结果
    autograd_tensor_destroy(diff);
    autograd_tensor_destroy(grad_scale);
}

// L1损失函数实现
AutogradL1Loss* autograd_l1_loss_create(bool reduction_mean) {
    AutogradL1Loss* loss = (AutogradL1Loss*)malloc(sizeof(AutogradL1Loss));
    if (!loss) return NULL;
    
    loss->base.type = LOSS_L1;
    loss->base.config.reduction_mean = reduction_mean;
    loss->base.config.reduction_sum = !reduction_mean;
    loss->base.forward = autograd_l1_loss_forward;
    loss->base.backward = autograd_l1_loss_backward;
    loss->base.destroy = autograd_l1_loss_destroy;
    
    return loss;
}

void autograd_l1_loss_destroy(AutogradLossFunction* base) {
    AutogradL1Loss* loss = (AutogradL1Loss*)base;
    if (loss) {
        free(loss);
    }
}

AutogradTensor* autograd_l1_loss_forward(AutogradLossFunction* base, AutogradTensor* input, AutogradTensor* target) {
    if (!base || !input || !target) return NULL;
    
    // 计算绝对差: |input - target|
    AutogradTensor* diff = autograd_tensor_subtract(input, target);
    AutogradTensor* abs_diff = autograd_tensor_abs(diff);
    
    // 应用降维
    AutogradTensor* result = autograd_apply_reduction(abs_diff, base->config.reduction_mean, base->config.reduction_sum);
    
    // 清理中间结果
    autograd_tensor_destroy(diff);
    autograd_tensor_destroy(abs_diff);
    
    return result;
}

void autograd_l1_loss_backward(AutogradLossFunction* base, AutogradTensor* grad_output,
                              AutogradTensor* input, AutogradTensor* target, AutogradTensor** grad_input) {
    if (!base || !input || !target || !grad_output || !grad_input) return;
    
    // L1损失的梯度: sign(input - target)
    AutogradTensor* diff = autograd_tensor_subtract(input, target);
    AutogradTensor* sign = autograd_tensor_sign(diff);
    
    if (base->config.reduction_mean) {
        int input_size = autograd_tensor_size(input);
        *grad_input = autograd_tensor_divide_scalar(sign, (float)input_size);
    } else {
        *grad_input = autograd_tensor_clone(sign);
    }
    
    // 应用grad_output缩放
    if (grad_output && autograd_tensor_size(grad_output) == 1) {
        float scale = autograd_tensor_data_float(grad_output)[0];
        AutogradTensor* scaled = autograd_tensor_multiply_scalar(*grad_input, scale);
        autograd_tensor_destroy(*grad_input);
        *grad_input = scaled;
    }
    
    // 清理中间结果
    autograd_tensor_destroy(diff);
    autograd_tensor_destroy(sign);
}

// 交叉熵损失函数实现
AutogradCrossEntropyLoss* autograd_cross_entropy_loss_create(AutogradTensor* weight, int ignore_index, bool weight_by_class) {
    AutogradCrossEntropyLoss* loss = (AutogradCrossEntropyLoss*)malloc(sizeof(AutogradCrossEntropyLoss));
    if (!loss) return NULL;
    
    loss->base.type = LOSS_CROSS_ENTROPY;
    loss->base.config.reduction_mean = true;
    loss->base.config.reduction_sum = false;
    loss->base.config.weight_by_class = weight_by_class;
    loss->base.config.ignore_index = (ignore_index >= 0);
    loss->base.config.ignore_index_value = ignore_index;
    loss->base.forward = autograd_cross_entropy_loss_forward;
    loss->base.backward = autograd_cross_entropy_loss_backward;
    loss->base.destroy = autograd_cross_entropy_loss_destroy;
    
    loss->weight = weight ? autograd_tensor_clone(weight) : NULL;
    loss->ignore_index = ignore_index;
    
    return loss;
}

void autograd_cross_entropy_loss_destroy(AutogradLossFunction* base) {
    AutogradCrossEntropyLoss* loss = (AutogradCrossEntropyLoss*)base;
    if (loss) {
        if (loss->weight) {
            autograd_tensor_destroy(loss->weight);
        }
        free(loss);
    }
}

AutogradTensor* autograd_cross_entropy_loss_forward(AutogradLossFunction* base, AutogradTensor* input, AutogradTensor* target) {
    if (!base || !input || !target) return NULL;
    
    AutogradCrossEntropyLoss* loss = (AutogradCrossEntropyLoss*)base;
    
    // 获取输入形状
    const int* input_shape = autograd_tensor_shape(input);
    int num_classes = input_shape[1]; // 假设input是[batch_size, num_classes]
    int batch_size = input_shape[0];
    
    // 创建损失张量
    int loss_shape[] = {batch_size};
    AutogradTensor* loss_tensor = autograd_tensor_create(loss_shape, 1, DTYPE_FLOAT32, true);
    
    float* input_data = autograd_tensor_data_float(input);
    float* target_data = autograd_tensor_data_float(target);
    float* loss_data = autograd_tensor_data_float(loss_tensor);
    float* weight_data = loss->weight ? autograd_tensor_data_float(loss->weight) : NULL;
    
    // 计算交叉熵损失
    for (int i = 0; i < batch_size; i++) {
        int target_class = (int)target_data[i];
        
        // 检查是否忽略该索引
        if (base->config.ignore_index && target_class == loss->ignore_index) {
            loss_data[i] = 0.0f;
            continue;
        }
        
        // 获取对应类别的logits
        float logit = input_data[i * num_classes + target_class];
        
        // 计算log-sum-exp用于数值稳定性
        float max_logit = -FLT_MAX;
        for (int j = 0; j < num_classes; j++) {
            if (input_data[i * num_classes + j] > max_logit) {
                max_logit = input_data[i * num_classes + j];
            }
        }
        
        // 计算softmax的对数
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += safe_exp(input_data[i * num_classes + j] - max_logit);
        }
        
        float log_softmax = logit - max_logit - safe_log(sum_exp);
        
        // 应用类别权重
        float class_weight = 1.0f;
        if (weight_data && target_class < num_classes) {
            class_weight = weight_data[target_class];
        }
        
        loss_data[i] = -log_softmax * class_weight;
    }
    
    // 应用降维
    AutogradTensor* result = autograd_apply_reduction(loss_tensor, base->config.reduction_mean, base->config.reduction_sum);
    autograd_tensor_destroy(loss_tensor);
    
    return result;
}

void autograd_cross_entropy_loss_backward(AutogradLossFunction* base, AutogradTensor* grad_output,
                                        AutogradTensor* input, AutogradTensor* target, AutogradTensor** grad_input) {
    if (!base || !input || !target || !grad_output || !grad_input) return;
    
    AutogradCrossEntropyLoss* loss = (AutogradCrossEntropyLoss*)base;
    
    // 获取输入形状
    const int* input_shape = autograd_tensor_shape(input);
    int num_classes = input_shape[1];
    int batch_size = input_shape[0];
    
    // 创建梯度张量
    *grad_input = autograd_tensor_create_like(input);
    
    float* input_data = autograd_tensor_data_float(input);
    float* target_data = autograd_tensor_data_float(target);
    float* grad_input_data = autograd_tensor_data_float(*grad_input);
    float* weight_data = loss->weight ? autograd_tensor_data_float(loss->weight) : NULL;
    
    // 获取输出梯度
    float grad_scale = 1.0f;
    if (grad_output && autograd_tensor_size(grad_output) == 1) {
        grad_scale = autograd_tensor_data_float(grad_output)[0];
    }
    
    if (base->config.reduction_mean) {
        grad_scale /= (float)batch_size;
    }
    
    // 计算交叉熵梯度: softmax(input) - one_hot(target)
    for (int i = 0; i < batch_size; i++) {
        int target_class = (int)target_data[i];
        
        // 检查是否忽略该索引
        if (base->config.ignore_index && target_class == loss->ignore_index) {
            for (int j = 0; j < num_classes; j++) {
                grad_input_data[i * num_classes + j] = 0.0f;
            }
            continue;
        }
        
        // 计算softmax
        float max_logit = -FLT_MAX;
        for (int j = 0; j < num_classes; j++) {
            if (input_data[i * num_classes + j] > max_logit) {
                max_logit = input_data[i * num_classes + j];
            }
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += safe_exp(input_data[i * num_classes + j] - max_logit);
        }
        
        // 计算梯度
        float class_weight = 1.0f;
        if (weight_data && target_class < num_classes) {
            class_weight = weight_data[target_class];
        }
        
        for (int j = 0; j < num_classes; j++) {
            float softmax = safe_exp(input_data[i * num_classes + j] - max_logit) / sum_exp;
            float target_val = (j == target_class) ? 1.0f : 0.0f;
            grad_input_data[i * num_classes + j] = (softmax - target_val) * grad_scale * class_weight;
        }
    }
}

// BCE损失函数实现
AutogradBCELoss* autograd_bce_loss_create(AutogradTensor* weight, AutogradTensor* pos_weight) {
    AutogradBCELoss* loss = (AutogradBCELoss*)malloc(sizeof(AutogradBCELoss));
    if (!loss) return NULL;
    
    loss->base.type = LOSS_BINARY_CROSS_ENTROPY;
    loss->base.config.reduction_mean = true;
    loss->base.config.reduction_sum = false;
    loss->base.forward = autograd_bce_loss_forward;
    loss->base.backward = autograd_bce_loss_backward;
    loss->base.destroy = autograd_bce_loss_destroy;
    
    loss->weight = weight ? autograd_tensor_clone(weight) : NULL;
    loss->pos_weight = pos_weight ? autograd_tensor_clone(pos_weight) : NULL;
    
    return loss;
}

void autograd_bce_loss_destroy(AutogradLossFunction* base) {
    AutogradBCELoss* loss = (AutogradBCELoss*)base;
    if (loss) {
        if (loss->weight) {
            autograd_tensor_destroy(loss->weight);
        }
        if (loss->pos_weight) {
            autograd_tensor_destroy(loss->pos_weight);
        }
        free(loss);
    }
}

AutogradTensor* autograd_bce_loss_forward(AutogradLossFunction* base, AutogradTensor* input, AutogradTensor* target) {
    if (!base || !input || !target) return NULL;
    
    // 确保输入在[0, 1]范围内
    AutogradTensor* clamped_input = autograd_tensor_clamp(input, 1e-7f, 1.0f - 1e-7f);
    
    // 计算BCE损失: -[target * log(input) + (1-target) * log(1-input)]
    AutogradTensor* log_input = autograd_tensor_log(clamped_input);
    AutogradTensor* log_1_minus_input = autograd_tensor_log(autograd_tensor_subtract_scalar(clamped_input, 1.0f));
    
    AutogradTensor* target_log_input = autograd_tensor_multiply(target, log_input);
    AutogradTensor* one_minus_target = autograd_tensor_subtract_scalar(target, 1.0f);
    AutogradTensor* target_log_1_minus_input = autograd_tensor_multiply(one_minus_target, log_1_minus_input);
    
    AutogradTensor* loss_sum = autograd_tensor_subtract(target_log_input, target_log_1_minus_input);
    AutogradTensor* loss_neg = autograd_tensor_multiply_scalar(loss_sum, -1.0f);
    
    // 应用样本权重
    AutogradBCELoss* bce_loss = (AutogradBCELoss*)base;
    if (bce_loss->weight) {
        AutogradTensor* weighted_loss = autograd_tensor_multiply(loss_neg, bce_loss->weight);
        autograd_tensor_destroy(loss_neg);
        loss_neg = weighted_loss;
    }
    
    // 应用正样本权重
    if (bce_loss->pos_weight) {
        AutogradTensor* pos_weight_term = autograd_tensor_multiply(target, bce_loss->pos_weight);
        AutogradTensor* weighted_loss = autograd_tensor_multiply(loss_neg, pos_weight_term);
        autograd_tensor_destroy(loss_neg);
        autograd_tensor_destroy(pos_weight_term);
        loss_neg = weighted_loss;
    }
    
    // 应用降维
    AutogradTensor* result = autograd_apply_reduction(loss_neg, base->config.reduction_mean, base->config.reduction_sum);
    
    // 清理中间结果
    autograd_tensor_destroy(clamped_input);
    autograd_tensor_destroy(log_input);
    autograd_tensor_destroy(log_1_minus_input);
    autograd_tensor_destroy(target_log_input);
    autograd_tensor_destroy(one_minus_target);
    autograd_tensor_destroy(target_log_1_minus_input);
    autograd_tensor_destroy(loss_sum);
    autograd_tensor_destroy(loss_neg);
    
    return result;
}

void autograd_bce_loss_backward(AutogradLossFunction* base, AutogradTensor* grad_output,
                              AutogradTensor* input, AutogradTensor* target, AutogradTensor** grad_input) {
    if (!base || !input || !target || !grad_output || !grad_input) return;
    
    // 确保输入在[0, 1]范围内
    AutogradTensor* clamped_input = autograd_tensor_clamp(input, 1e-7f, 1.0f - 1e-7f);
    
    // BCE损失的梯度: (input - target) / (input * (1 - input))
    AutogradTensor* one_minus_input = autograd_tensor_subtract_scalar(clamped_input, 1.0f);
    AutogradTensor* input_times_one_minus_input = autograd_tensor_multiply(clamped_input, one_minus_input);
    AutogradTensor* diff = autograd_tensor_subtract(clamped_input, target);
    
    *grad_input = autograd_tensor_divide(diff, input_times_one_minus_input);
    
    // 应用grad_output缩放
    if (grad_output && autograd_tensor_size(grad_output) == 1) {
        float scale = autograd_tensor_data_float(grad_output)[0];
        if (base->config.reduction_mean) {
            int input_size = autograd_tensor_size(input);
            scale /= (float)input_size;
        }
        
        AutogradTensor* scaled = autograd_tensor_multiply_scalar(*grad_input, scale);
        autograd_tensor_destroy(*grad_input);
        *grad_input = scaled;
    }
    
    // 清理中间结果
    autograd_tensor_destroy(clamped_input);
    autograd_tensor_destroy(one_minus_input);
    autograd_tensor_destroy(input_times_one_minus_input);
    autograd_tensor_destroy(diff);
}

// 内部辅助函数实现
static float safe_log(float x) {
    if (x <= 0.0f) return -FLT_MAX;
    return logf(x);
}

static float safe_exp(float x) {
    if (x > 88.0f) return FLT_MAX;
    if (x < -88.0f) return 0.0f;
    return expf(x);
}

static float calculate_mean(const float* data, int size) {
    if (size <= 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / (float)size;
}

static float calculate_sum(const float* data, int size) {
    if (size <= 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

static int argmax(const float* data, int size) {
    if (size <= 0) return -1;
    
    int max_idx = 0;
    float max_val = data[0];
    
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// 损失函数计算辅助函数实现
float autograd_calculate_cross_entropy(float input, int target) {
    return -safe_log(input);
}

float autograd_calculate_binary_cross_entropy(float input, float target) {
    input = fmaxf(1e-7f, fminf(1.0f - 1e-7f, input));
    return -(target * safe_log(input) + (1.0f - target) * safe_log(1.0f - input));
}

float autograd_calculate_kl_divergence(float input, float target) {
    if (target <= 0.0f || input <= 0.0f) return 0.0f;
    return target * (safe_log(target) - safe_log(input));
}

float autograd_calculate_dice_coefficient(float input, float target) {
    return 2.0f * input * target / (input * input + target * target + 1e-7f);
}

float autograd_calculate_tversky_index(float input, float target, float alpha, float beta) {
    float tp = input * target;
    float fp = input * (1.0f - target);
    float fn = (1.0f - input) * target;
    return tp / (tp + alpha * fp + beta * fn + 1e-7f);
}

float autograd_calculate_focal_weight(float input, float target, float alpha, float gamma) {
    float ce_loss = autograd_calculate_binary_cross_entropy(input, target);
    float p_t = target * input + (1.0f - target) * (1.0f - input);
    return alpha * powf(1.0f - p_t, gamma) * ce_loss;
}

float autograd_calculate_smooth_l1(float input, float target) {
    float diff = fabsf(input - target);
    if (diff < 1.0f) {
        return 0.5f * diff * diff;
    } else {
        return diff - 0.5f;
    }
}

float autograd_calculate_huber(float input, float target, float delta) {
    float diff = fabsf(input - target);
    if (diff < delta) {
        return 0.5f * diff * diff;
    } else {
        return delta * (diff - 0.5f * delta);
    }
}

float autograd_calculate_log_cosh(float input, float target) {
    float diff = input - target;
    return logf(coshf(diff));
}

float autograd_calculate_quantile(float input, float target, float quantile) {
    float diff = target - input;
    if (diff >= 0.0f) {
        return quantile * diff;
    } else {
        return (quantile - 1.0f) * diff;
    }
}

float autograd_calculate_poisson(float input, float target) {
    return input - target * safe_log(input);
}

float autograd_calculate_gaussian_nll(float input, float target, float var) {
    if (var <= 0.0f) return 0.0f;
    return 0.5f * (safe_log(var) + (input - target) * (input - target) / var);
}

float autograd_calculate_hinge(float input, float target, float margin) {
    return fmaxf(0.0f, margin - input * target);
}

float autograd_calculate_cosine_embedding(float input1, float input2, float target, float margin) {
    float similarity = input1 * input2;
    if (target > 0.0f) {
        return 1.0f - similarity;
    } else {
        return fmaxf(0.0f, similarity - margin);
    }
}

float autograd_calculate_margin_ranking(float input1, float input2, float target, float margin) {
    return fmaxf(0.0f, margin - target * (input1 - input2));
}

float autograd_calculate_triplet_margin(float anchor, float positive, float negative, float margin) {
    return fmaxf(0.0f, anchor - positive + margin);
}

// 损失函数注册函数
void autograd_loss_register(const char* name, AutogradLossFunction* (*create_func)(void)) {
    if (!name || !create_func) return;
    
    LossRegistry* entry = (LossRegistry*)malloc(sizeof(LossRegistry));
    if (!entry) return;
    
    entry->name = strdup(name);
    entry->create_func = create_func;
    entry->next = g_loss_registry;
    g_loss_registry = entry;
}

AutogradLossFunction* autograd_loss_create(const char* name) {
    if (!name) return NULL;
    
    LossRegistry* current = g_loss_registry;
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current->create_func();
        }
        current = current->next;
    }
    
    return NULL;
}

void autograd_loss_unregister(const char* name) {
    if (!name) return;
    
    LossRegistry** current = &g_loss_registry;
    while (*current) {
        if (strcmp((*current)->name, name) == 0) {
            LossRegistry* to_remove = *current;
            *current = (*current)->next;
            free((void*)to_remove->name);
            free(to_remove);
            return;
        }
        current = &(*current)->next;
    }
}

void autograd_loss_functions_init(void) {
    // 注册标准损失函数
    autograd_loss_register("MSELoss", (AutogradLossFunction* (*)(void))autograd_mse_loss_create);
    autograd_loss_register("L1Loss", (AutogradLossFunction* (*)(void))autograd_l1_loss_create);
    autograd_loss_register("CrossEntropyLoss", (AutogradLossFunction* (*)(void))autograd_cross_entropy_loss_create);
    autograd_loss_register("BCELoss", (AutogradLossFunction* (*)(void))autograd_bce_loss_create);
}

void autograd_loss_functions_cleanup(void) {
    // 清理注册表
    while (g_loss_registry) {
        LossRegistry* to_remove = g_loss_registry;
        g_loss_registry = g_loss_registry->next;
        free((void*)to_remove->name);
        free(to_remove);
    }
}