#ifndef HYPERPARAMETER_OPTIMIZATION_H
#define HYPERPARAMETER_OPTIMIZATION_H

#include "nn_module.h"
#include "ai_trainer.h"
#include "tensor.h"
#include <stdbool.h>

// ===========================================
// 超参数优化类型枚举
// ===========================================

typedef enum {
    OPTIMIZATION_GRID_SEARCH = 0,      // 网格搜索
    OPTIMIZATION_RANDOM_SEARCH = 1,    // 随机搜索
    OPTIMIZATION_BAYESIAN = 2,         // 贝叶斯优化
    OPTIMIZATION_GRADIENT_BASED = 3,   // 基于梯度的优化
    OPTIMIZATION_EVOLUTIONARY = 4,     // 进化算法
    OPTIMIZATION_BANDIT = 5,           // 多臂老虎机算法
    OPTIMIZATION_HYPERBAND = 6,        // HyperBand算法
    OPTIMIZATION_BOHB = 7              // BOHB（贝叶斯优化+HyperBand）
} hyperparameter_optimization_type_t;

// ===========================================
// 超参数空间定义
// ===========================================

typedef enum {
    PARAM_TYPE_FLOAT = 0,      // 浮点数参数
    PARAM_TYPE_INT = 1,        // 整数参数
    PARAM_TYPE_CATEGORICAL = 2,// 分类参数
    PARAM_TYPE_LOG = 3         // 对数尺度参数
} hyperparameter_type_t;

typedef struct {
    char* name;                // 参数名称
    hyperparameter_type_t type;// 参数类型
    union {
        struct {
            float min_value;    // 最小值
            float max_value;    // 最大值
            float step;         // 步长（可选）
        } float_range;
        struct {
            int min_value;      // 最小值
            int max_value;      // 最大值
            int step;           // 步长（可选）
        } int_range;
        struct {
            char** categories;  // 分类值数组
            int num_categories; // 分类数量
        } categorical;
    } range;
    float default_value;       // 默认值
    bool is_log_scale;         // 是否对数尺度
} hyperparameter_definition_t;

typedef struct {
    hyperparameter_definition_t* parameters; // 参数定义数组
    int num_parameters;        // 参数数量
    int max_parameters;        // 最大参数数量
} hyperparameter_space_t;

// ===========================================
// 优化配置结构体
// ===========================================

typedef struct {
    hyperparameter_optimization_type_t method; // 优化方法
    int max_trials;             // 最大试验次数
    int max_epochs_per_trial;   // 每次试验的最大轮次
    float early_stopping_patience; // 早停耐心值
    int num_folds;              // 交叉验证折数
    float train_validation_split; // 训练验证分割比例
    
    // 贝叶斯优化配置
    int bayesian_num_initial_points; // 初始点数量
    float bayesian_acquisition_weight; // 获取函数权重
    
    // 进化算法配置
    int evolutionary_population_size; // 种群大小
    float evolutionary_mutation_rate; // 变异率
    float evolutionary_crossover_rate; // 交叉率
    
    // HyperBand配置
    int hyperband_max_iter;     // 最大迭代次数
    float hyperband_eta;        // 淘汰比例
    
    // 通用配置
    bool use_parallel;          // 是否使用并行
    int max_parallel_jobs;      // 最大并行任务数
    char* metric;               // 优化指标
    bool maximize_metric;       // 是否最大化指标
    
    // 资源限制
    int max_time_seconds;       // 最大时间（秒）
    int max_memory_mb;          // 最大内存（MB）
} hyperparameter_optimization_config_t;

// ===========================================
// 试验结果结构体
// ===========================================

typedef struct {
    int trial_id;               // 试验ID
    float* parameter_values;    // 参数值数组
    float score;               // 得分
    float training_time;       // 训练时间（秒）
    int epochs_trained;        // 训练的轮次
    float* validation_metrics; // 验证指标数组
    int num_metrics;           // 指标数量
    bool is_completed;         // 是否完成
    char* status_message;      // 状态消息
    
    // 模型相关信息
    nn_module_t* best_model;   // 最佳模型
    ai_trainer_t* trainer;     // 训练器
} hyperparameter_trial_result_t;

// ===========================================
// 优化结果结构体
// ===========================================

typedef struct {
    hyperparameter_trial_result_t* trials; // 试验结果数组
    int num_trials;            // 试验数量
    int best_trial_index;      // 最佳试验索引
    float best_score;         // 最佳得分
    float total_optimization_time; // 总优化时间
    
    // 统计信息
    float mean_score;          // 平均得分
    float std_score;           // 得分标准差
    float median_score;        // 得分中位数
    
    // 收敛分析
    float convergence_rate;    // 收敛率
    bool is_converged;        // 是否收敛
    
    // 超参数重要性
    float* parameter_importance; // 参数重要性分数
    
    // 配置信息
    hyperparameter_optimization_config_t config; // 优化配置
    hyperparameter_space_t space; // 参数空间
} hyperparameter_optimization_result_t;

// ===========================================
// 超参数优化管理器结构体
// ===========================================

typedef struct {
    bool is_initialized;       // 是否已初始化
    hyperparameter_space_t space; // 参数空间
    hyperparameter_optimization_config_t config; // 优化配置
    
    // 状态信息
    int current_trial;         // 当前试验
    bool is_running;          // 是否正在运行
    float start_time;         // 开始时间
    
    // 结果存储
    hyperparameter_optimization_result_t* result; // 优化结果
    
    // 回调函数
    void (*progress_callback)(int progress, const char* message); // 进度回调
    void (*trial_complete_callback)(const hyperparameter_trial_result_t* trial); // 试验完成回调
    void (*optimization_complete_callback)(const hyperparameter_optimization_result_t* result); // 优化完成回调
} hyperparameter_optimization_manager_t;

// ===========================================
// API接口函数声明
// ===========================================

// 管理器创建和销毁
hyperparameter_optimization_manager_t* create_hyperparameter_optimization_manager(void);
void destroy_hyperparameter_optimization_manager(hyperparameter_optimization_manager_t* manager);

// 参数空间管理
int add_hyperparameter_float(hyperparameter_optimization_manager_t* manager, 
                           const char* name, float min_value, float max_value, 
                           float default_value, bool is_log_scale);
int add_hyperparameter_int(hyperparameter_optimization_manager_t* manager,
                         const char* name, int min_value, int max_value,
                         int default_value, bool is_log_scale);
int add_hyperparameter_categorical(hyperparameter_optimization_manager_t* manager,
                                 const char* name, const char** categories, 
                                 int num_categories, const char* default_category);

// 优化配置
int set_optimization_method(hyperparameter_optimization_manager_t* manager, 
                          hyperparameter_optimization_type_t method);
int configure_optimization(hyperparameter_optimization_manager_t* manager,
                         const hyperparameter_optimization_config_t* config);

// 优化执行
int start_hyperparameter_optimization(hyperparameter_optimization_manager_t* manager,
                                     nn_module_t* (*model_creator)(const float* params),
                                     tensor_t* train_data, tensor_t* train_labels,
                                     tensor_t* validation_data, tensor_t* validation_labels);

int stop_hyperparameter_optimization(hyperparameter_optimization_manager_t* manager);

// 结果获取和分析
hyperparameter_optimization_result_t* get_optimization_result(
    const hyperparameter_optimization_manager_t* manager);

float* get_best_hyperparameters(const hyperparameter_optimization_manager_t* manager);
nn_module_t* get_best_model(const hyperparameter_optimization_manager_t* manager);

// 分析和可视化支持
float* get_parameter_importance(const hyperparameter_optimization_manager_t* manager);
int save_optimization_history(const hyperparameter_optimization_manager_t* manager, 
                             const char* filename);
int load_optimization_history(hyperparameter_optimization_manager_t* manager, 
                            const char* filename);

// 回调函数设置
void set_optimization_progress_callback(hyperparameter_optimization_manager_t* manager,
                                      void (*callback)(int progress, const char* message));
void set_trial_complete_callback(hyperparameter_optimization_manager_t* manager,
                               void (*callback)(const hyperparameter_trial_result_t* trial));
void set_optimization_complete_callback(hyperparameter_optimization_manager_t* manager,
                                      void (*callback)(const hyperparameter_optimization_result_t* result));

// 工具函数
hyperparameter_space_t* create_hyperparameter_space(void);
void destroy_hyperparameter_space(hyperparameter_space_t* space);

hyperparameter_optimization_config_t create_default_optimization_config(void);

// 高级功能：多目标优化
int set_multi_objective_metrics(hyperparameter_optimization_manager_t* manager,
                              const char** metrics, int num_metrics,
                              const bool* maximize_flags);

// 高级功能：约束优化
int add_optimization_constraint(hyperparameter_optimization_manager_t* manager,
                              bool (*constraint_function)(const float* params));

// 高级功能：热启动
int add_warm_start_point(hyperparameter_optimization_manager_t* manager,
                       const float* params, float score);

#endif // HYPERPARAMETER_OPTIMIZATION_H