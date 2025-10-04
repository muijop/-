#ifndef OPTIMIZERS_AUTOGRAD_H
#define OPTIMIZERS_AUTOGRAD_H

#include "nn.h"
#include "tensor.h"
#include "tensor_autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 优化器类型定义 ====================

typedef enum {
    OPTIMIZER_SGD = 0,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_ADAGRAD,
    OPTIMIZER_ADADELTA,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_ADAMAX,
    OPTIMIZER_NADAM,
    OPTIMIZER_RADAM,
    OPTIMIZER_LAMB,
    OPTIMIZER_LARS,
    OPTIMIZER_SGDW,
    OPTIMIZER_AMSGRAD,
    OPTIMIZER_ADABELIEF
} AutogradOptimizerType;

// ==================== 优化器配置结构 ====================

typedef struct AutogradOptimizerConfig {
    AutogradOptimizerType type;      // 优化器类型
    float learning_rate;             // 学习率
    float momentum;                  // 动量参数
    float beta1;                    // Adam beta1
    float beta2;                    // Adam beta2
    float epsilon;                  // 数值稳定性参数
    float weight_decay;             // 权重衰减
    float rho;                      // Adadelta rho
    float alpha;                    // RMSprop alpha
    int amsgrad;                    // 是否使用AMSGrad
    int centered;                   // 是否使用中心化RMSprop
    int nesterov;                   // 是否使用Nesterov动量
    int maximize;                   // 是否最大化目标
    float dampening;                // 动量阻尼
    int foreach;                    // 是否使用foreach实现
    int differentiable;             // 是否可微分
    float lr_decay;                 // 学习率衰减
    float initial_accumulator_value; // 初始累加器值
    float eps;                      // 数值稳定性参数
    int max_iter;                   // 最大迭代次数
    int tolerance_change;           // 容忍变化
    int tolerance_grad;             // 梯度容忍度
    int history_size;               // 历史大小
    int line_search_fn;             // 线搜索函数
} AutogradOptimizerConfig;

// ==================== 优化器状态结构 ====================

typedef struct AutogradOptimizerState {
    AutogradTensor* momentum_buffer;    // 动量缓冲区
    AutogradTensor* velocity_buffer;    // 速度缓冲区
    AutogradTensor* exp_avg;            // 指数移动平均
    AutogradTensor* exp_avg_sq;         // 指数移动平均平方
    AutogradTensor* max_exp_avg_sq;    // 最大指数移动平均平方
    AutogradTensor* step_buffer;        // 步数缓冲区
    AutogradTensor* prev_loss;          // 前一次损失
    AutogradTensor* grad_buffer;        // 梯度缓冲区
    AutogradTensor* param_buffer;       // 参数缓冲区
    int step;                          // 当前步数
    float loss;                        // 当前损失
    float prev_loss_value;             // 前一次损失值
    int converged;                     // 是否收敛
    int num_no_improvement;            // 无改进次数
} AutogradOptimizerState;

// ==================== 优化器结构 ====================

typedef struct AutogradOptimizer {
    AutogradOptimizerType type;        // 优化器类型
    AutogradOptimizerConfig config;    // 优化器配置
    AutogradOptimizerState state;      // 优化器状态
    AutogradTensor** params;           // 参数数组
    int num_params;                    // 参数数量
    AutogradTensor** param_groups;     // 参数组
    int num_param_groups;              // 参数组数量
    char* name;                        // 优化器名称
    int initialized;                   // 是否已初始化
    void* user_data;                   // 用户数据
} AutogradOptimizer;

// ==================== 优化器创建和销毁 ====================

// 创建优化器
AutogradOptimizer* autograd_optimizer_create(AutogradOptimizerType type, float learning_rate);

// 销毁优化器
void autograd_optimizer_destroy(AutogradOptimizer* optimizer);

// ==================== 优化器配置管理 ====================

// 设置优化器配置
void autograd_optimizer_set_config(AutogradOptimizer* optimizer, const AutogradOptimizerConfig* config);

// 获取优化器配置
AutogradOptimizerConfig autograd_optimizer_get_config(const AutogradOptimizer* optimizer);

// ==================== 参数管理 ====================

// 添加参数到优化器
void autograd_optimizer_add_param(AutogradOptimizer* optimizer, AutogradTensor* param);

// 添加参数组到优化器
void autograd_optimizer_add_param_group(AutogradOptimizer* optimizer, AutogradTensor** params, int num_params);

// 获取优化器参数
AutogradTensor** autograd_optimizer_get_params(AutogradOptimizer* optimizer, int* num_params);

// 获取参数组
AutogradTensor** autograd_optimizer_get_param_group(AutogradOptimizer* optimizer, int group_idx, int* num_params);

// 获取参数组数量
int autograd_optimizer_num_param_groups(const AutogradOptimizer* optimizer);

// ==================== 优化步骤 ====================

// 优化器步骤
void autograd_optimizer_step(AutogradOptimizer* optimizer);

// 优化器步骤（带闭包）
float autograd_optimizer_step_with_closure(AutogradOptimizer* optimizer, float (*closure)(void*), void* closure_data);

// 零梯度
void autograd_optimizer_zero_grad(AutogradOptimizer* optimizer);

// ==================== 状态管理 ====================

// 获取优化器状态
AutogradOptimizerState autograd_optimizer_get_state(const AutogradOptimizer* optimizer);

// 设置优化器状态
void autograd_optimizer_set_state(AutogradOptimizer* optimizer, const AutogradOptimizerState* state);

// 重置优化器状态
void autograd_optimizer_reset_state(AutogradOptimizer* optimizer);

// ==================== 学习率管理 ====================

// 设置学习率
void autograd_optimizer_set_learning_rate(AutogradOptimizer* optimizer, float learning_rate);

// 获取学习率
float autograd_optimizer_get_learning_rate(const AutogradOptimizer* optimizer);

// 设置动量
void autograd_optimizer_set_momentum(AutogradOptimizer* optimizer, float momentum);

// 获取动量
float autograd_optimizer_get_momentum(const AutogradOptimizer* optimizer);

// ==================== 特定优化器函数 ====================

// SGD优化器步骤
void autograd_optimizer_sgd_step(AutogradOptimizer* optimizer);

// Momentum优化器步骤
void autograd_optimizer_momentum_step(AutogradOptimizer* optimizer);

// Adagrad优化器步骤
void autograd_optimizer_adagrad_step(AutogradOptimizer* optimizer);

// Adadelta优化器步骤
void autograd_optimizer_adadelta_step(AutogradOptimizer* optimizer);

// RMSprop优化器步骤
void autograd_optimizer_rmsprop_step(AutogradOptimizer* optimizer);

// Adam优化器步骤
void autograd_optimizer_adam_step(AutogradOptimizer* optimizer);

// AdamW优化器步骤
void autograd_optimizer_adamw_step(AutogradOptimizer* optimizer);

// Adamax优化器步骤
void autograd_optimizer_adamax_step(AutogradOptimizer* optimizer);

// NAdam优化器步骤
void autograd_optimizer_nadam_step(AutogradOptimizer* optimizer);

// RAdam优化器步骤
void autograd_optimizer_radam_step(AutogradOptimizer* optimizer);

// LAMB优化器步骤
void autograd_optimizer_lamb_step(AutogradOptimizer* optimizer);

// LARS优化器步骤
void autograd_optimizer_lars_step(AutogradOptimizer* optimizer);

// SGDW优化器步骤
void autograd_optimizer_sgdw_step(AutogradOptimizer* optimizer);

// AMSGrad优化器步骤
void autograd_optimizer_amsgrad_step(AutogradOptimizer* optimizer);

// AdaBelief优化器步骤
void autograd_optimizer_adabelief_step(AutogradOptimizer* optimizer);

// ==================== 优化器工厂函数 ====================

// 创建SGD优化器
AutogradOptimizer* autograd_optimizer_sgd(float learning_rate, float momentum, int nesterov, float dampening);

// 创建Momentum优化器
AutogradOptimizer* autograd_optimizer_momentum(float learning_rate, float momentum, int nesterov, float dampening);

// 创建Adagrad优化器
AutogradOptimizer* autograd_optimizer_adagrad(float learning_rate, float lr_decay, float weight_decay, float initial_accumulator_value);

// 创建Adadelta优化器
AutogradOptimizer* autograd_optimizer_adadelta(float learning_rate, float rho, float eps, float weight_decay);

// 创建RMSprop优化器
AutogradOptimizer* autograd_optimizer_rmsprop(float learning_rate, float alpha, float eps, float weight_decay, float momentum, int centered);

// 创建Adam优化器
AutogradOptimizer* autograd_optimizer_adam(float learning_rate, float beta1, float beta2, float eps, float weight_decay, int amsgrad);

// 创建AdamW优化器
AutogradOptimizer* autograd_optimizer_adamw(float learning_rate, float beta1, float beta2, float eps, float weight_decay, int amsgrad);

// 创建Adamax优化器
AutogradOptimizer* autograd_optimizer_adamax(float learning_rate, float beta1, float beta2, float eps, float weight_decay);

// 创建NAdam优化器
AutogradOptimizer* autograd_optimizer_nadam(float learning_rate, float beta1, float beta2, float eps, float weight_decay, float momentum_decay);

// 创建RAdam优化器
AutogradOptimizer* autograd_optimizer_radam(float learning_rate, float beta1, float beta2, float eps, float weight_decay);

// 创建LAMB优化器
AutogradOptimizer* autograd_optimizer_lamb(float learning_rate, float beta1, float beta2, float eps, float weight_decay);

// 创建LARS优化器
AutogradOptimizer* autograd_optimizer_lars(float learning_rate, float momentum, float weight_decay, float eta, float epsilon);

// 创建SGDW优化器
AutogradOptimizer* autograd_optimizer_sgdw(float learning_rate, float momentum, float weight_decay, int nesterov, float dampening);

// 创建AMSGrad优化器
AutogradOptimizer* autograd_optimizer_amsgrad(float learning_rate, float beta1, float beta2, float eps, float weight_decay);

// 创建AdaBelief优化器
AutogradOptimizer* autograd_optimizer_adabelief(float learning_rate, float beta1, float beta2, float eps, float weight_decay, int amsgrad);

// ==================== 优化器工具函数 ====================

// 优化器初始化
void autograd_optimizer_initialize(AutogradOptimizer* optimizer);

// 优化器验证
int autograd_optimizer_validate(const AutogradOptimizer* optimizer);

// 优化器重置
void autograd_optimizer_reset(AutogradOptimizer* optimizer);

// 优化器克隆
AutogradOptimizer* autograd_optimizer_clone(const AutogradOptimizer* optimizer);

// 优化器比较
int autograd_optimizer_compare(const AutogradOptimizer* optimizer1, const AutogradOptimizer* optimizer2);

// 优化器序列化
void autograd_optimizer_serialize(const AutogradOptimizer* optimizer, const char* filename);

// 优化器反序列化
AutogradOptimizer* autograd_optimizer_deserialize(const char* filename);

// ==================== 优化器统计 ====================

// 获取优化器统计信息
typedef struct AutogradOptimizerStats {
    int total_steps;                   // 总步数
    float total_loss;                  // 总损失
    float average_loss;                // 平均损失
    float min_loss;                    // 最小损失
    float max_loss;                    // 最大损失
    float current_loss;                // 当前损失
    int converged;                     // 是否收敛
    int num_no_improvement;            // 无改进次数
    float learning_rate;               // 当前学习率
    float momentum;                    // 当前动量
} AutogradOptimizerStats;

// 获取优化器统计
AutogradOptimizerStats autograd_optimizer_get_stats(const AutogradOptimizer* optimizer);

// 重置优化器统计
void autograd_optimizer_reset_stats(AutogradOptimizer* optimizer);

// ==================== 优化器回调 ====================

// 优化器回调函数类型
typedef void (*AutogradOptimizerCallback)(AutogradOptimizer* optimizer, int step, float loss, void* user_data);

// 设置优化器回调
void autograd_optimizer_set_callback(AutogradOptimizer* optimizer, AutogradOptimizerCallback callback, void* user_data);

// ==================== 优化器调试 ====================

// 打印优化器信息
void autograd_optimizer_print_info(const AutogradOptimizer* optimizer);

// 打印优化器状态
void autograd_optimizer_print_state(const AutogradOptimizer* optimizer);

// 打印优化器统计
void autograd_optimizer_print_stats(const AutogradOptimizer* optimizer);

// 优化器调试模式
void autograd_optimizer_set_debug_mode(AutogradOptimizer* optimizer, int debug);

// ==================== 优化器配置预设 ====================

// 获取默认优化器配置
AutogradOptimizerConfig autograd_optimizer_config_default(AutogradOptimizerType type);

// 获取SGD默认配置
AutogradOptimizerConfig autograd_optimizer_config_sgd(void);

// 获取Adam默认配置
AutogradOptimizerConfig autograd_optimizer_config_adam(void);

// 获取RMSprop默认配置
AutogradOptimizerConfig autograd_optimizer_config_rmsprop(void);

#ifdef __cplusplus
}
#endif

#endif // OPTIMIZERS_AUTOGRAD_H