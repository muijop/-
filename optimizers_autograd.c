#include "optimizers_autograd.h"
#include "kernels.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

// 优化器注册表
typedef struct OptimizerRegistry {
    const char* name;
    AutogradOptimizer* (*create_func)(float learning_rate, ...);
    struct OptimizerRegistry* next;
} OptimizerRegistry;

static OptimizerRegistry* g_optimizer_registry = NULL;

// 内部辅助函数
static void optimizer_state_init(OptimizerState* state, int num_parameters);
static void optimizer_state_free(OptimizerState* state);
static void apply_weight_decay(AutogradTensor* param, float weight_decay, float learning_rate);

// SGD优化器实现
AutogradSGD* autograd_sgd_create(float learning_rate, float momentum, float weight_decay, 
                                float dampening, bool nesterov) {
    AutogradSGD* optimizer = (AutogradSGD*)malloc(sizeof(AutogradSGD));
    if (!optimizer) return NULL;
    
    // 初始化基础优化器
    optimizer->base.type = OPTIMIZER_SGD;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.weight_decay = weight_decay;
    optimizer->base.use_kernel_optimization = true;
    optimizer->base.step = autograd_sgd_step;
    optimizer->base.zero_grad = autograd_optimizer_zero_grad;
    optimizer->base.state_dict = autograd_optimizer_state_dict;
    optimizer->base.load_state_dict = autograd_optimizer_load_state_dict;
    optimizer->base.destroy = autograd_sgd_destroy;
    
    // 初始化SGD参数
    optimizer->params.momentum = momentum;
    optimizer->params.dampening = dampening;
    optimizer->params.nesterov = nesterov;
    
    // 初始化状态
    optimizer_state_init(&optimizer->state, 0);
    
    return optimizer;
}

void autograd_sgd_destroy(AutogradOptimizer* base) {
    AutogradSGD* optimizer = (AutogradSGD*)base;
    if (optimizer) {
        optimizer_state_free(&optimizer->state);
        free(optimizer);
    }
}

void autograd_sgd_step(AutogradOptimizer* base, AutogradTensor** parameters, int num_parameters) {
    AutogradSGD* optimizer = (AutogradSGD*)base;
    if (!optimizer || !parameters || num_parameters <= 0) return;
    
    // 确保状态缓冲区足够大
    if (optimizer->state.num_parameters != num_parameters) {
        optimizer_state_free(&optimizer->state);
        optimizer_state_init(&optimizer->state, num_parameters);
    }
    
    float lr = optimizer->base.learning_rate;
    float momentum = optimizer->params.momentum;
    float dampening = optimizer->params.dampening;
    float weight_decay = optimizer->base.weight_decay;
    bool nesterov = optimizer->params.nesterov;
    
    for (int i = 0; i < num_parameters; i++) {
        AutogradTensor* param = parameters[i];
        if (!param || !param->grad_node || !param->grad_node->grad) continue;
        
        float* param_data = autograd_tensor_data_float(param);
        float* grad_data = autograd_tensor_data_float(param->grad_node->grad);
        int param_size = autograd_tensor_size(param);
        
        // 应用权重衰减
        if (weight_decay != 0.0f) {
            apply_weight_decay(param, weight_decay, lr);
        }
        
        // 应用动量
        if (momentum != 0.0f) {
            AutogradTensor* momentum_buf = optimizer->state.momentum_buffers[i];
            float* momentum_data = autograd_tensor_data_float(momentum_buf);
            
            if (optimizer->state.step_count == 0) {
                // 初始化动量缓冲区
                for (int j = 0; j < param_size; j++) {
                    momentum_data[j] = grad_data[j];
                }
            } else {
                // 更新动量
                for (int j = 0; j < param_size; j++) {
                    momentum_data[j] = momentum * momentum_data[j] + (1 - dampening) * grad_data[j];
                }
            }
            
            if (nesterov) {
                // Nesterov动量
                for (int j = 0; j < param_size; j++) {
                    param_data[j] -= lr * (grad_data[j] + momentum * momentum_data[j]);
                }
            } else {
                // 标准动量
                for (int j = 0; j < param_size; j++) {
                    param_data[j] -= lr * momentum_data[j];
                }
            }
        } else {
            // 无动量的SGD
            for (int j = 0; j < param_size; j++) {
                param_data[j] -= lr * grad_data[j];
            }
        }
    }
    
    optimizer->state.step_count++;
}

// Momentum优化器实现
AutogradMomentum* autograd_momentum_create(float learning_rate, float momentum, float weight_decay,
                                          float dampening, bool nesterov) {
    AutogradMomentum* optimizer = (AutogradMomentum*)malloc(sizeof(AutogradMomentum));
    if (!optimizer) return NULL;
    
    // 初始化基础优化器
    optimizer->base.type = OPTIMIZER_MOMENTUM;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.weight_decay = weight_decay;
    optimizer->base.use_kernel_optimization = true;
    optimizer->base.step = autograd_momentum_step;
    optimizer->base.zero_grad = autograd_optimizer_zero_grad;
    optimizer->base.state_dict = autograd_optimizer_state_dict;
    optimizer->base.load_state_dict = autograd_optimizer_load_state_dict;
    optimizer->base.destroy = autograd_momentum_destroy;
    
    // 初始化Momentum参数
    optimizer->params.momentum = momentum;
    optimizer->params.dampening = dampening;
    optimizer->params.nesterov = nesterov;
    
    // 初始化状态
    optimizer_state_init(&optimizer->state, 0);
    
    return optimizer;
}

void autograd_momentum_destroy(AutogradOptimizer* base) {
    AutogradMomentum* optimizer = (AutogradMomentum*)base;
    if (optimizer) {
        optimizer_state_free(&optimizer->state);
        free(optimizer);
    }
}

void autograd_momentum_step(AutogradOptimizer* base, AutogradTensor** parameters, int num_parameters) {
    // Momentum优化器与SGD类似，但始终使用动量
    autograd_sgd_step(base, parameters, num_parameters);
}

// Adam优化器实现
AutogradAdam* autograd_adam_create(float learning_rate, float beta1, float beta2, float epsilon,
                                   float weight_decay, bool amsgrad) {
    AutogradAdam* optimizer = (AutogradAdam*)malloc(sizeof(AutogradAdam));
    if (!optimizer) return NULL;
    
    // 初始化基础优化器
    optimizer->base.type = OPTIMIZER_ADAM;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.weight_decay = weight_decay;
    optimizer->base.use_kernel_optimization = true;
    optimizer->base.step = autograd_adam_step;
    optimizer->base.zero_grad = autograd_optimizer_zero_grad;
    optimizer->base.state_dict = autograd_optimizer_state_dict;
    optimizer->base.load_state_dict = autograd_optimizer_load_state_dict;
    optimizer->base.destroy = autograd_adam_destroy;
    
    // 初始化Adam参数
    optimizer->params.beta1 = beta1;
    optimizer->params.beta2 = beta2;
    optimizer->params.epsilon = epsilon;
    optimizer->params.amsgrad = amsgrad;
    
    // 初始化状态
    optimizer_state_init(&optimizer->state, 0);
    optimizer->bias_correction1 = 1.0f;
    optimizer->bias_correction2 = 1.0f;
    
    return optimizer;
}

void autograd_adam_destroy(AutogradOptimizer* base) {
    AutogradAdam* optimizer = (AutogradAdam*)base;
    if (optimizer) {
        optimizer_state_free(&optimizer->state);
        free(optimizer);
    }
}

void autograd_adam_step(AutogradOptimizer* base, AutogradTensor** parameters, int num_parameters) {
    AutogradAdam* optimizer = (AutogradAdam*)base;
    if (!optimizer || !parameters || num_parameters <= 0) return;
    
    // 确保状态缓冲区足够大
    if (optimizer->state.num_parameters != num_parameters) {
        optimizer_state_free(&optimizer->state);
        optimizer_state_init(&optimizer->state, num_parameters);
    }
    
    float lr = optimizer->base.learning_rate;
    float beta1 = optimizer->params.beta1;
    float beta2 = optimizer->params.beta2;
    float epsilon = optimizer->params.epsilon;
    float weight_decay = optimizer->base.weight_decay;
    
    // 更新偏差修正
    optimizer->bias_correction1 *= beta1;
    optimizer->bias_correction2 *= beta2;
    
    float bias_correction1_corrected = 1.0f - optimizer->bias_correction1;
    float bias_correction2_corrected = 1.0f - optimizer->bias_correction2;
    
    float step_size = lr * sqrtf(bias_correction2_corrected) / bias_correction1_corrected;
    
    for (int i = 0; i < num_parameters; i++) {
        AutogradTensor* param = parameters[i];
        if (!param || !param->grad_node || !param->grad_node->grad) continue;
        
        float* param_data = autograd_tensor_data_float(param);
        float* grad_data = autograd_tensor_data_float(param->grad_node->grad);
        int param_size = autograd_tensor_size(param);
        
        // 应用权重衰减
        if (weight_decay != 0.0f) {
            for (int j = 0; j < param_size; j++) {
                grad_data[j] += weight_decay * param_data[j];
            }
        }
        
        // 获取动量和方差缓冲区
        AutogradTensor* momentum_buf = optimizer->state.momentum_buffers[i];
        AutogradTensor* variance_buf = optimizer->state.variance_buffers[i];
        float* momentum_data = autograd_tensor_data_float(momentum_buf);
        float* variance_data = autograd_tensor_data_float(variance_buf);
        
        // 更新指数移动平均
        for (int j = 0; j < param_size; j++) {
            momentum_data[j] = beta1 * momentum_data[j] + (1 - beta1) * grad_data[j];
            variance_data[j] = beta2 * variance_data[j] + (1 - beta2) * grad_data[j] * grad_data[j];
        }
        
        // 更新参数
        for (int j = 0; j < param_size; j++) {
            float denom = sqrtf(variance_data[j]) + epsilon;
            param_data[j] -= step_size * momentum_data[j] / denom;
        }
    }
}

// RMSprop优化器实现
AutogradRMSprop* autograd_rmsprop_create(float learning_rate, float alpha, float momentum,
                                        float epsilon, float weight_decay, bool centered) {
    AutogradRMSprop* optimizer = (AutogradRMSprop*)malloc(sizeof(AutogradRMSprop));
    if (!optimizer) return NULL;
    
    // 初始化基础优化器
    optimizer->base.type = OPTIMIZER_RMSPROP;
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.weight_decay = weight_decay;
    optimizer->base.use_kernel_optimization = true;
    optimizer->base.step = autograd_rmsprop_step;
    optimizer->base.zero_grad = autograd_optimizer_zero_grad;
    optimizer->base.state_dict = autograd_optimizer_state_dict;
    optimizer->base.load_state_dict = autograd_optimizer_load_state_dict;
    optimizer->base.destroy = autograd_rmsprop_destroy;
    
    // 初始化RMSprop参数
    optimizer->params.alpha = alpha;
    optimizer->params.momentum = momentum;
    optimizer->params.epsilon = epsilon;
    optimizer->params.centered = centered;
    
    // 初始化状态
    optimizer_state_init(&optimizer->state, 0);
    
    return optimizer;
}

void autograd_rmsprop_destroy(AutogradOptimizer* base) {
    AutogradRMSprop* optimizer = (AutogradRMSprop*)base;
    if (optimizer) {
        optimizer_state_free(&optimizer->state);
        free(optimizer);
    }
}

void autograd_rmsprop_step(AutogradOptimizer* base, AutogradTensor** parameters, int num_parameters) {
    AutogradRMSprop* optimizer = (AutogradRMSprop*)base;
    if (!optimizer || !parameters || num_parameters <= 0) return;
    
    // 确保状态缓冲区足够大
    if (optimizer->state.num_parameters != num_parameters) {
        optimizer_state_free(&optimizer->state);
        optimizer_state_init(&optimizer->state, num_parameters);
    }
    
    float lr = optimizer->base.learning_rate;
    float alpha = optimizer->params.alpha;
    float momentum = optimizer->params.momentum;
    float epsilon = optimizer->params.epsilon;
    float weight_decay = optimizer->base.weight_decay;
    bool centered = optimizer->params.centered;
    
    for (int i = 0; i < num_parameters; i++) {
        AutogradTensor* param = parameters[i];
        if (!param || !param->grad_node || !param->grad_node->grad) continue;
        
        float* param_data = autograd_tensor_data_float(param);
        float* grad_data = autograd_tensor_data_float(param->grad_node->grad);
        int param_size = autograd_tensor_size(param);
        
        // 应用权重衰减
        if (weight_decay != 0.0f) {
            for (int j = 0; j < param_size; j++) {
                grad_data[j] += weight_decay * param_data[j];
            }
        }
        
        // 获取方差缓冲区
        AutogradTensor* variance_buf = optimizer->state.variance_buffers[i];
        float* variance_data = autograd_tensor_data_float(variance_buf);
        
        // 更新平方梯度平均值
        for (int j = 0; j < param_size; j++) {
            variance_data[j] = alpha * variance_data[j] + (1 - alpha) * grad_data[j] * grad_data[j];
        }
        
        // 计算梯度缩放
        if (centered) {
            AutogradTensor* grad_avg_buf = optimizer->state.momentum_buffers[i];
            float* grad_avg_data = autograd_tensor_data_float(grad_avg_buf);
            
            // 更新梯度平均值
            for (int j = 0; j < param_size; j++) {
                grad_avg_data[j] = alpha * grad_avg_data[j] + (1 - alpha) * grad_data[j];
            }
            
            // 中心化RMSprop
            for (int j = 0; j < param_size; j++) {
                float avg = grad_avg_data[j];
                float var = variance_data[j];
                grad_data[j] = grad_data[j] / sqrtf(var - avg * avg + epsilon);
            }
        } else {
            // 标准RMSprop
            for (int j = 0; j < param_size; j++) {
                grad_data[j] = grad_data[j] / sqrtf(variance_data[j] + epsilon);
            }
        }
        
        // 应用动量
        if (momentum != 0.0f) {
            AutogradTensor* momentum_buf = optimizer->state.momentum_buffers[i];
            float* momentum_data = autograd_tensor_data_float(momentum_buf);
            
            for (int j = 0; j < param_size; j++) {
                momentum_data[j] = momentum * momentum_data[j] + grad_data[j];
                param_data[j] -= lr * momentum_data[j];
            }
        } else {
            // 无动量
            for (int j = 0; j < param_size; j++) {
                param_data[j] -= lr * grad_data[j];
            }
        }
    }
}

// 通用优化器操作函数
void autograd_optimizer_step(AutogradOptimizer* optimizer, AutogradTensor** parameters, int num_parameters) {
    if (optimizer && optimizer->step) {
        optimizer->step(optimizer, parameters, num_parameters);
    }
}

void autograd_optimizer_zero_grad(AutogradOptimizer* optimizer, AutogradTensor** parameters, int num_parameters) {
    if (!optimizer || !parameters || num_parameters <= 0) return;
    
    for (int i = 0; i < num_parameters; i++) {
        AutogradTensor* param = parameters[i];
        if (param && param->grad_node && param->grad_node->grad) {
            autograd_tensor_zero(param->grad_node->grad);
        }
    }
}

void autograd_optimizer_state_dict(AutogradOptimizer* optimizer, const char* filename) {
    if (!optimizer || !filename) return;
    
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    
    // 保存优化器类型和参数
    fwrite(&optimizer->type, sizeof(OptimizerType), 1, file);
    fwrite(&optimizer->learning_rate, sizeof(float), 1, file);
    fwrite(&optimizer->weight_decay, sizeof(float), 1, file);
    
    // 根据优化器类型保存特定参数
    switch (optimizer->type) {
        case OPTIMIZER_SGD:
        case OPTIMIZER_MOMENTUM: {
            SGDParams* params = (SGDParams*)((char*)optimizer + sizeof(AutogradOptimizer));
            fwrite(params, sizeof(SGDParams), 1, file);
            break;
        }
        case OPTIMIZER_ADAM: {
            AdamParams* params = (AdamParams*)((char*)optimizer + sizeof(AutogradOptimizer));
            fwrite(params, sizeof(AdamParams), 1, file);
            break;
        }
        case OPTIMIZER_RMSPROP: {
            RMSpropParams* params = (RMSpropParams*)((char*)optimizer + sizeof(AutogradOptimizer));
            fwrite(params, sizeof(RMSpropParams), 1, file);
            break;
        }
        default:
            break;
    }
    
    fclose(file);
}

void autograd_optimizer_load_state_dict(AutogradOptimizer* optimizer, const char* filename) {
    if (!optimizer || !filename) return;
    
    FILE* file = fopen(filename, "rb");
    if (!file) return;
    
    // 读取优化器状态
    OptimizerType saved_type;
    float saved_lr, saved_weight_decay;
    
    fread(&saved_type, sizeof(OptimizerType), 1, file);
    fread(&saved_lr, sizeof(float), 1, file);
    fread(&saved_weight_decay, sizeof(float), 1, file);
    
    // 验证优化器类型匹配
    if (saved_type != optimizer->type) {
        fclose(file);
        return;
    }
    
    // 更新参数
    optimizer->learning_rate = saved_lr;
    optimizer->weight_decay = saved_weight_decay;
    
    fclose(file);
}

void autograd_optimizer_destroy(AutogradOptimizer* optimizer) {
    if (optimizer && optimizer->destroy) {
        optimizer->destroy(optimizer);
    }
}

// 内部辅助函数实现
static void optimizer_state_init(OptimizerState* state, int num_parameters) {
    if (!state) return;
    
    state->num_parameters = num_parameters;
    state->step_count = 0;
    
    if (num_parameters <= 0) return;
    
    // 分配缓冲区数组
    state->momentum_buffers = (AutogradTensor**)malloc(num_parameters * sizeof(AutogradTensor*));
    state->variance_buffers = (AutogradTensor**)malloc(num_parameters * sizeof(AutogradTensor*));
    state->grad_buffers = (AutogradTensor**)malloc(num_parameters * sizeof(AutogradTensor*));
    state->exp_avg_factors = (float*)malloc(num_parameters * sizeof(float));
    
    // 初始化缓冲区指针
    for (int i = 0; i < num_parameters; i++) {
        state->momentum_buffers[i] = NULL;
        state->variance_buffers[i] = NULL;
        state->grad_buffers[i] = NULL;
        state->exp_avg_factors[i] = 0.0f;
    }
}

static void optimizer_state_free(OptimizerState* state) {
    if (!state) return;
    
    // 释放缓冲区
    if (state->momentum_buffers) {
        for (int i = 0; i < state->num_parameters; i++) {
            if (state->momentum_buffers[i]) {
                autograd_tensor_destroy(state->momentum_buffers[i]);
            }
        }
        free(state->momentum_buffers);
    }
    
    if (state->variance_buffers) {
        for (int i = 0; i < state->num_parameters; i++) {
            if (state->variance_buffers[i]) {
                autograd_tensor_destroy(state->variance_buffers[i]);
            }
        }
        free(state->variance_buffers);
    }
    
    if (state->grad_buffers) {
        for (int i = 0; i < state->num_parameters; i++) {
            if (state->grad_buffers[i]) {
                autograd_tensor_destroy(state->grad_buffers[i]);
            }
        }
        free(state->grad_buffers);
    }
    
    if (state->exp_avg_factors) {
        free(state->exp_avg_factors);
    }
    
    state->num_parameters = 0;
    state->step_count = 0;
}

static void apply_weight_decay(AutogradTensor* param, float weight_decay, float learning_rate) {
    if (!param || !param->grad_node || !param->grad_node->grad) return;
    
    float* param_data = autograd_tensor_data_float(param);
    float* grad_data = autograd_tensor_data_float(param->grad_node->grad);
    int param_size = autograd_tensor_size(param);
    
    // L2权重衰减: grad += weight_decay * param
    for (int i = 0; i < param_size; i++) {
        grad_data[i] += weight_decay * param_data[i];
    }
}

// 优化器注册函数
void autograd_optimizer_register(const char* name, AutogradOptimizer* (*create_func)(float learning_rate, ...)) {
    if (!name || !create_func) return;
    
    OptimizerRegistry* entry = (OptimizerRegistry*)malloc(sizeof(OptimizerRegistry));
    if (!entry) return;
    
    entry->name = strdup(name);
    entry->create_func = create_func;
    entry->next = g_optimizer_registry;
    g_optimizer_registry = entry;
}

AutogradOptimizer* autograd_optimizer_create(const char* name, float learning_rate, ...) {
    if (!name) return NULL;
    
    OptimizerRegistry* current = g_optimizer_registry;
    while (current) {
        if (strcmp(current->name, name) == 0) {
            va_list args;
            va_start(args, learning_rate);
            AutogradOptimizer* optimizer = current->create_func(learning_rate, args);
            va_end(args);
            return optimizer;
        }
        current = current->next;
    }
    
    return NULL;
}

void autograd_optimizer_unregister(const char* name) {
    if (!name) return;
    
    OptimizerRegistry** current = &g_optimizer_registry;
    while (*current) {
        if (strcmp((*current)->name, name) == 0) {
            OptimizerRegistry* to_remove = *current;
            *current = (*current)->next;
            free((void*)to_remove->name);
            free(to_remove);
            return;
        }
        current = &(*current)->next;
    }
}

void autograd_optimizers_init(void) {
    // 注册标准优化器
    autograd_optimizer_register("SGD", (AutogradOptimizer* (*)(float learning_rate, ...))autograd_sgd_create);
    autograd_optimizer_register("Momentum", (AutogradOptimizer* (*)(float learning_rate, ...))autograd_momentum_create);
    autograd_optimizer_register("Adam", (AutogradOptimizer* (*)(float learning_rate, ...))autograd_adam_create);
    autograd_optimizer_register("RMSprop", (AutogradOptimizer* (*)(float learning_rate, ...))autograd_rmsprop_create);
}

void autograd_optimizers_cleanup(void) {
    // 清理注册表
    while (g_optimizer_registry) {
        OptimizerRegistry* to_remove = g_optimizer_registry;
        g_optimizer_registry = g_optimizer_registry->next;
        free((void*)to_remove->name);
        free(to_remove);
    }
}