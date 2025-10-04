#ifndef AI_FRAMEWORK_MAIN_H
#define AI_FRAMEWORK_MAIN_H

#include <stddef.h>
#include <stdbool.h>

// ===========================================
// 基础类型定义
// ===========================================

typedef enum {
    FRAMEWORK_MODE_UNIFIED = 0,      // 统一模式
    FRAMEWORK_MODE_MODULAR = 1,      // 模块化模式
    FRAMEWORK_MODE_HIGH_PERFORMANCE = 2, // 高性能模式
    FRAMEWORK_MODE_LIGHTWEIGHT = 3   // 轻量级模式
} ai_framework_mode_t;

typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_CRITICAL = 4
} log_level_t;

// ===========================================
// 训练相关类型定义
// ===========================================

typedef enum {
    TRAINING_MODE_STANDARD = 0,              // 标准训练
    TRAINING_MODE_HYPERPARAM_OPTIMIZATION = 1, // 超参数优化训练
    TRAINING_MODE_META_LEARNING = 2,         // 元学习训练
    TRAINING_MODE_REINFORCEMENT_LEARNING = 3, // 强化学习训练
    TRAINING_MODE_DISTRIBUTED = 4,           // 分布式训练
    TRAINING_MODE_FEDERATED = 5              // 联邦学习训练
} training_mode_t;

typedef enum {
    MODEL_TYPE_FEEDFORWARD = 0,      // 前馈神经网络
    MODEL_TYPE_CONVOLUTIONAL = 1,     // 卷积神经网络
    MODEL_TYPE_RECURRENT = 2,         // 循环神经网络
    MODEL_TYPE_TRANSFORMER = 3,      // Transformer模型
    MODEL_TYPE_GRAPH = 4,            // 图神经网络
    MODEL_TYPE_TIME_SERIES = 5       // 时间序列模型
} model_type_t;

typedef enum {
    OPTIMIZER_SGD = 0,               // 随机梯度下降
    OPTIMIZER_ADAM = 1,              // Adam优化器
    OPTIMIZER_RMSPROP = 2,           // RMSprop优化器
    OPTIMIZER_ADAGRAD = 3            // Adagrad优化器
} optimizer_type_t;

// 训练配置结构体
typedef struct {
    training_mode_t training_mode;    // 训练模式
    model_type_t model_type;          // 模型类型
    optimizer_type_t optimizer_type;  // 优化器类型
    float learning_rate;               // 学习率
    size_t batch_size;                // 批量大小
    size_t epochs;                    // 训练轮数
    bool use_early_stopping;          // 是否使用早停
    size_t patience;                  // 早停耐心值
    float validation_split;           // 验证集比例
} training_config_t;

// 训练数据结构体
typedef struct {
    float* input_data;                // 输入数据
    float* target_data;               // 目标数据
    size_t data_size;                 // 数据大小
    size_t input_dim;                 // 输入维度
    size_t output_dim;                // 输出维度
    size_t batch_size;                // 批次大小
    size_t num_samples;               // 样本数量
    bool is_sparse;                   // 是否为稀疏数据
    char* data_format;                // 数据格式
    void* metadata;                   // 元数据指针
} training_data_t;

// 训练结果结构体
typedef struct {
    bool success;                     // 训练是否成功
    float final_loss;                 // 最终损失值
    float final_accuracy;            // 最终准确率
    size_t training_time_ms;         // 训练时间(毫秒)
    char* error_message;             // 错误信息
    char* training_log;              // 训练日志
    
    // 子模块结果指针
    void* hyperparameter_optimization_info;  // 超参数优化信息
    void* meta_learning_info;                // 元学习信息
    void* reinforcement_learning_info;        // 强化学习信息
    void* distributed_training_info;         // 分布式训练信息
    void* federated_learning_info;           // 联邦学习信息
} training_result_t;

// ===========================================
// 推理相关类型定义
// ===========================================

// 推理请求结构体
typedef struct {
    size_t request_id;                // 请求ID
    float* input_data;                // 输入数据
    size_t input_size;                // 输入大小
    void* output_buffer;              // 输出缓冲区
    size_t output_size;               // 输出大小
    size_t timeout_ms;                // 超时时间(毫秒)
    double submission_time;          // 提交时间
} inference_request_t;

// 推理结果结构体
typedef struct {
    size_t request_id;                // 请求ID
    bool success;                     // 推理是否成功
    float inference_time_ms;          // 推理时间(毫秒)
    float confidence;                 // 置信度
    void* output_data;                // 输出数据
    size_t output_size;               // 输出大小
    float preprocessing_time_ms;      // 预处理时间
    float postprocessing_time_ms;     // 后处理时间
    size_t memory_used_bytes;         // 内存使用量
    char* error_message;              // 错误信息
    char* model_name;                 // 模型名称
    size_t model_version;             // 模型版本
} inference_result_t;

// ===========================================
// 框架配置结构体
// ===========================================

typedef struct {
    ai_framework_mode_t framework_mode;      // 框架模式
    bool use_gpu_acceleration;               // 是否使用GPU加速
    bool use_distributed_training;           // 是否使用分布式训练
    bool use_model_compression;              // 是否使用模型压缩
    bool use_model_explainability;           // 是否使用模型解释性
    bool use_visualization;                  // 是否使用可视化
    size_t max_memory_usage_mb;              // 最大内存使用量(MB)
    size_t max_computation_time_ms;          // 最大计算时间(毫秒)
    log_level_t log_level;                   // 日志级别
} ai_framework_config_t;

// ===========================================
// AI框架管理器结构体
// ===========================================

// 前向声明各模块管理器
typedef struct hyperparameter_optimizer_t hyperparameter_optimizer_t;
typedef struct meta_learning_manager_t meta_learning_manager_t;
typedef struct reinforcement_learning_agent_t reinforcement_learning_agent_t;
typedef struct gnn_model_t gnn_model_t;
typedef struct time_series_model_t time_series_model_t;
typedef struct federated_learning_model_t federated_learning_model_t;
typedef struct deployment_model_t deployment_model_t;
typedef struct model_compression_manager_t model_compression_manager_t;
typedef struct model_explainability_manager_t model_explainability_manager_t;
typedef struct distributed_training_manager_t distributed_training_manager_t;
typedef struct visualization_manager_t visualization_manager_t;
typedef struct unified_ai_framework_t unified_ai_framework_t;

typedef struct {
    // 框架配置
    ai_framework_config_t config;
    
    // 各模块管理器
    hyperparameter_optimizer_t* hyperparam_optimizer;
    meta_learning_manager_t* meta_learner;
    reinforcement_learning_agent_t* rl_agent;
    gnn_model_t* gnn_model;
    time_series_model_t* time_series_model;
    federated_learning_model_t* federated_model;
    deployment_model_t* deployment_model;
    model_compression_manager_t* compression_model;
    model_explainability_manager_t* explainability_model;
    distributed_training_manager_t* distributed_trainer;
    visualization_manager_t* visualizer;
    
    // 统一框架
    unified_ai_framework_t* unified_framework;
    
    // 框架状态
    bool is_initialized;
    size_t initialization_time;
    size_t total_training_requests;
    size_t total_inference_requests;
    size_t successful_training_requests;
    size_t successful_inference_requests;
    
} ai_framework_manager_t;

// ===========================================
// 主要API函数声明
// ===========================================

// 框架管理器创建和销毁
ai_framework_manager_t* create_ai_framework_manager(void);
void destroy_ai_framework_manager(ai_framework_manager_t* manager);

// 框架配置管理
int configure_ai_framework(ai_framework_manager_t* manager,
                          const ai_framework_config_t* config);
int set_framework_mode(ai_framework_manager_t* manager,
                      ai_framework_mode_t mode);

// 模型训练接口
training_result_t* train_model(ai_framework_manager_t* manager,
                              const training_config_t* config,
                              const training_data_t* data);

// 模型推理接口
inference_result_t* perform_inference_with_framework(
    ai_framework_manager_t* manager,
    const inference_request_t* request);

// 模型部署接口
deployment_result_t* deploy_model_with_framework(
    ai_framework_manager_t* manager,
    deployment_environment_t environment,
    const deployment_config_t* config);

// 模型优化接口
int optimize_model_with_framework(ai_framework_manager_t* manager,
                                 const char* optimization_strategy);

// 模型解释性接口
model_explainability_result_t* explain_model_with_framework(
    ai_framework_manager_t* manager,
    const model_explainability_request_t* request);

// 可视化接口
int visualize_training_progress(ai_framework_manager_t* manager,
                               const visualization_config_t* config);
int visualize_model_architecture(ai_framework_manager_t* manager,
                                const visualization_config_t* config);

// 框架状态监控
void print_framework_status(const ai_framework_manager_t* manager);

// 工具函数
void destroy_training_result(training_result_t* result);
void destroy_inference_result(inference_result_t* result);
void destroy_inference_request(inference_request_t* request);
ai_framework_config_t get_default_framework_config(void);

// ===========================================
// 内部函数声明（仅供框架内部使用）
// ===========================================

// 内部训练函数
training_result_t* train_standard_model(ai_framework_manager_t* manager,
                                       const training_config_t* config,
                                       const training_data_t* data);

training_result_t* train_with_hyperparameter_optimization(
    ai_framework_manager_t* manager,
    const training_config_t* config,
    const training_data_t* data);

training_result_t* train_with_meta_learning(
    ai_framework_manager_t* manager,
    const training_config_t* config,
    const training_data_t* data);

training_result_t* train_with_reinforcement_learning(
    ai_framework_manager_t* manager,
    const training_config_t* config,
    const training_data_t* data);

training_result_t* train_with_distributed_training(
    ai_framework_manager_t* manager,
    const training_config_t* config,
    const training_data_t* data);

training_result_t* train_with_federated_learning(
    ai_framework_manager_t* manager,
    const training_config_t* config,
    const training_data_t* data);

#endif // AI_FRAMEWORK_MAIN_H