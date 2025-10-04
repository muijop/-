#ifndef META_LEARNING_H
#define META_LEARNING_H

#include "nn_module.h"
#include "ai_trainer.h"
#include "tensor.h"
#include <stdbool.h>

// ===========================================
// 元学习类型枚举
// ===========================================

typedef enum {
    META_LEARNING_MAML = 0,            // 模型无关元学习（MAML）
    META_LEARNING_REPTILE = 1,         // Reptile算法
    META_LEARNING_PROTONET = 2,         // 原型网络
    META_LEARNING_MATCHINGNET = 3,      // 匹配网络
    META_LEARNING_RELATIONNET = 4,      // 关系网络
    META_LEARNING_LEARN2LEARN = 5,     // 学习如何学习
    META_LEARNING_METACURVATURE = 6,   // 元曲率
    META_LEARNING_ANIL = 7             // 几乎内循环学习（ANIL）
} meta_learning_type_t;

// ===========================================
// 元学习任务定义
// ===========================================

typedef struct {
    char* task_name;           // 任务名称
    tensor_t* support_set;     // 支持集（训练数据）
    tensor_t* support_labels;  // 支持集标签
    tensor_t* query_set;       // 查询集（测试数据）
    tensor_t* query_labels;    // 查询集标签
    int num_classes;          // 类别数量
    int k_shot;              // k-shot（每个类别的支持样本数）
    int n_way;                // n-way（类别数量）
    int adaptation_steps;     // 适应步数
} meta_learning_task_t;

// ===========================================
// 元学习配置结构体
// ===========================================

typedef struct {
    meta_learning_type_t method;       // 元学习方法
    int num_tasks;                     // 任务数量
    int inner_loop_steps;              // 内循环步数
    float inner_learning_rate;         // 内循环学习率
    float outer_learning_rate;         // 外循环学习率
    int batch_size;                    // 批次大小
    int adaptation_steps;              // 适应步数
    
    // MAML特定配置
    float maml_first_order;           // 是否使用一阶近似
    bool maml_use_batch_norm;         // 是否使用批归一化
    
    // Reptile特定配置
    float reptile_step_size;           // Reptile步长
    int reptile_inner_batches;         // 内循环批次
    
    // 原型网络配置
    bool protonet_use_euclidean;       // 是否使用欧氏距离
    float protonet_distance_scale;     // 距离缩放因子
    
    // 通用配置
    bool use_second_order;             // 是否使用二阶导数
    bool normalize_gradients;          // 是否归一化梯度
    float gradient_clip_value;          // 梯度裁剪值
    
    // 训练配置
    int max_epochs;                    // 最大训练轮次
    float early_stopping_patience;    // 早停耐心值
    bool use_validation;              // 是否使用验证集
    
    // 资源限制
    int max_time_seconds;             // 最大时间（秒）
    int max_memory_mb;                 // 最大内存（MB）
} meta_learning_config_t;

// ===========================================
// 元学习结果结构体
// ===========================================

typedef struct {
    float meta_train_loss;             // 元训练损失
    float meta_validation_loss;        // 元验证损失
    float meta_test_loss;              // 元测试损失
    float meta_train_accuracy;         // 元训练准确率
    float meta_validation_accuracy;    // 元验证准确率
    float meta_test_accuracy;          // 元测试准确率
    
    // 快速适应性能
    float* adaptation_accuracy;        // 适应过程中的准确率
    int num_adaptation_steps;          // 适应步数
    
    // 任务特定性能
    float* task_performance;           // 各任务性能
    int num_tasks;                     // 任务数量
    
    // 收敛分析
    float convergence_rate;            // 收敛率
    bool is_converged;                 // 是否收敛
    
    // 时间统计
    float total_training_time;         // 总训练时间
    float average_adaptation_time;     // 平均适应时间
    
    // 模型状态
    nn_module_t* meta_model;          // 元学习后的模型
    bool is_model_ready;              // 模型是否就绪
} meta_learning_result_t;

// ===========================================
// 元学习管理器结构体
// ===========================================

typedef struct {
    bool is_initialized;               // 是否已初始化
    meta_learning_config_t config;     // 元学习配置
    
    // 任务数据
    meta_learning_task_t* tasks;       // 任务数组
    int num_tasks;                     // 任务数量
    int max_tasks;                     // 最大任务数量
    
    // 训练状态
    bool is_training;                  // 是否正在训练
    bool is_adapted;                   // 是否已适应
    int current_epoch;                 // 当前轮次
    float start_time;                  // 开始时间
    
    // 结果存储
    meta_learning_result_t* result;    // 元学习结果
    
    // 回调函数
    void (*progress_callback)(int progress, const char* message); // 进度回调
    void (*task_complete_callback)(int task_id, float performance); // 任务完成回调
    void (*meta_learning_complete_callback)(const meta_learning_result_t* result); // 元学习完成回调
} meta_learning_manager_t;

// ===========================================
// API接口函数声明
// ===========================================

// 管理器创建和销毁
meta_learning_manager_t* create_meta_learning_manager(void);
void destroy_meta_learning_manager(meta_learning_manager_t* manager);

// 配置设置
int set_meta_learning_method(meta_learning_manager_t* manager, meta_learning_type_t method);
int configure_meta_learning(meta_learning_manager_t* manager, const meta_learning_config_t* config);

// 任务管理
int add_meta_learning_task(meta_learning_manager_t* manager, const meta_learning_task_t* task);
int load_meta_learning_tasks_from_directory(meta_learning_manager_t* manager, const char* directory_path);

// 元学习执行
int start_meta_learning(meta_learning_manager_t* manager, nn_module_t* base_model);
int stop_meta_learning(meta_learning_manager_t* manager);

// 快速适应
int adapt_to_new_task(meta_learning_manager_t* manager, const meta_learning_task_t* new_task);

// 结果获取
meta_learning_result_t* get_meta_learning_result(const meta_learning_manager_t* manager);
nn_module_t* get_meta_learned_model(const meta_learning_manager_t* manager);

// 性能评估
float evaluate_meta_learning_performance(meta_learning_manager_t* manager, 
                                       const meta_learning_task_t* test_task);

// 分析和可视化
int save_meta_learning_history(const meta_learning_manager_t* manager, const char* filename);
int load_meta_learning_history(meta_learning_manager_t* manager, const char* filename);

// 回调函数设置
void set_meta_learning_progress_callback(meta_learning_manager_t* manager,
                                       void (*callback)(int progress, const char* message));
void set_task_complete_callback(meta_learning_manager_t* manager,
                              void (*callback)(int task_id, float performance));
void set_meta_learning_complete_callback(meta_learning_manager_t* manager,
                                       void (*callback)(const meta_learning_result_t* result));

// 工具函数
meta_learning_config_t create_default_meta_learning_config(void);
meta_learning_task_t* create_meta_learning_task(const char* name, int n_way, int k_shot);
void destroy_meta_learning_task(meta_learning_task_t* task);

// 高级功能：多任务元学习
int enable_multi_task_meta_learning(meta_learning_manager_t* manager, bool enable);

// 高级功能：跨域元学习
int set_cross_domain_meta_learning(meta_learning_manager_t* manager, const char* source_domain, 
                                 const char* target_domain);

// 高级功能：在线元学习
int enable_online_meta_learning(meta_learning_manager_t* manager, bool enable);

#endif // META_LEARNING_H