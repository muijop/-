#ifndef AI_FRAMEWORK_MANAGER_H
#define AI_FRAMEWORK_MANAGER_H

#include "ai_framework_main.h"
#include "ai_framework_unified.h"
#include "ai_framework_enhanced.h"
#include "nn.h"
#include "model_builder.h"
#include "optimizer.h"
#include "distributed_training.h"
#include "federated_learning.h"
#include "multimodal_training.h"
#include "large_cnn_architectures.h"
#include "advanced_transformer.h"

#ifdef __cplusplus
extern "C" {
#endif

// 框架管理器状态
typedef enum {
    FRAMEWORK_STATE_UNINITIALIZED = 0,
    FRAMEWORK_STATE_INITIALIZED = 1,
    FRAMEWORK_STATE_TRAINING = 2,
    FRAMEWORK_STATE_EVALUATING = 3,
    FRAMEWORK_STATE_PREDICTING = 4,
    FRAMEWORK_STATE_ERROR = 5
} framework_state_t;

// 框架管理器配置
typedef struct {
    char* framework_name;
    framework_type_t framework_type;
    
    // 计算后端配置
    compute_backend_t compute_backend;
    int use_gpu;
    int gpu_device_id;
    
    // 内存管理配置
    size_t max_memory_usage;
    int enable_memory_pool;
    
    // 日志配置
    int log_level;
    char* log_file_path;
    int enable_tensorboard;
    
    // 性能配置
    int enable_profiling;
    int enable_optimizations;
    
    // 分布式配置
    int enable_distributed;
    distributed_config_t* distributed_config;
    
    // 联邦学习配置
    int enable_federated;
    federated_config_t* federated_config;
    
    // 多模态配置
    int enable_multimodal;
    multimodal_model_config_t* multimodal_config;
    
} framework_manager_config_t;

// 框架管理器统计
typedef struct {
    // 框架状态
    framework_state_t current_state;
    int total_models_created;
    int total_training_sessions;
    
    // 性能统计
    double total_training_time;
    double total_inference_time;
    size_t peak_memory_usage;
    
    // 资源使用
    int gpu_utilization;
    int cpu_utilization;
    size_t current_memory_usage;
    
    // 错误统计
    int total_errors;
    int last_error_code;
    char* last_error_message;
    
} framework_manager_stats_t;

// 框架管理器结构
typedef struct {
    char* manager_id;
    framework_manager_config_t* config;
    
    // 子框架实例
    ai_framework_t* main_framework;
    unified_framework_t* unified_framework;
    enhanced_framework_t* enhanced_framework;
    
    // 组件管理器
    model_builder_t* model_builder;
    optimizer_t* optimizer_manager;
    distributed_trainer_t* distributed_trainer;
    federated_server_t* federated_server;
    multimodal_model_t* multimodal_model;
    
    // 当前活动模型
    nn_module_t* current_model;
    training_config_t* current_training_config;
    
    // 状态管理
    framework_state_t state;
    framework_manager_stats_t* stats;
    
    // 回调函数
    void (*on_state_change)(framework_state_t new_state, void* user_data);
    void (*on_error)(int error_code, const char* error_message, void* user_data);
    
} framework_manager_t;

// ==================== 框架管理器核心API ====================

// 初始化框架管理器
int framework_manager_init(framework_manager_t* manager, 
                          framework_manager_config_t* config);

// 销毁框架管理器
int framework_manager_destroy(framework_manager_t* manager);

// 启动框架管理器
int framework_manager_start(framework_manager_t* manager);

// 停止框架管理器
int framework_manager_stop(framework_manager_t* manager);

// 重置框架管理器
int framework_manager_reset(framework_manager_t* manager);

// ==================== 模型管理API ====================

// 创建新模型
nn_module_t* framework_create_model(framework_manager_t* manager,
                                   model_config_t* config);

// 加载预训练模型
nn_module_t* framework_load_model(framework_manager_t* manager,
                                 const char* model_path);

// 保存当前模型
int framework_save_model(framework_manager_t* manager,
                        const char* model_path);

// 设置当前模型
int framework_set_current_model(framework_manager_t* manager,
                               nn_module_t* model);

// 获取当前模型
nn_module_t* framework_get_current_model(framework_manager_t* manager);

// ==================== 训练管理API ====================

// 配置训练参数
int framework_configure_training(framework_manager_t* manager,
                                training_config_t* config);

// 开始训练
int framework_start_training(framework_manager_t* manager,
                            training_data_t* train_data,
                            training_data_t* val_data);

// 暂停训练
int framework_pause_training(framework_manager_t* manager);

// 恢复训练
int framework_resume_training(framework_manager_t* manager);

// 停止训练
int framework_stop_training(framework_manager_t* manager);

// 获取训练进度
float framework_get_training_progress(framework_manager_t* manager);

// ==================== 推理管理API ====================

// 单样本推理
tensor_t* framework_predict(framework_manager_t* manager,
                          tensor_t* input);

// 批量推理
tensor_t** framework_predict_batch(framework_manager_t* manager,
                                 tensor_t** inputs,
                                 int batch_size);

// 流式推理
int framework_start_streaming_inference(framework_manager_t* manager,
                                       void (*callback)(tensor_t* result, void* user_data),
                                       void* user_data);

// 停止流式推理
int framework_stop_streaming_inference(framework_manager_t* manager);

// ==================== 分布式训练API ====================

// 初始化分布式训练
int framework_init_distributed_training(framework_manager_t* manager,
                                      distributed_config_t* config);

// 开始分布式训练
int framework_start_distributed_training(framework_manager_t* manager,
                                        training_data_t* train_data);

// 同步模型参数
int framework_sync_model_parameters(framework_manager_t* manager);

// 获取分布式训练状态
distributed_training_stats_t* framework_get_distributed_stats(framework_manager_t* manager);

// ==================== 联邦学习API ====================

// 初始化联邦学习
int framework_init_federated_learning(framework_manager_t* manager,
                                    federated_config_t* config);

// 启动联邦学习服务器
int framework_start_federated_server(framework_manager_t* manager);

// 加入联邦学习客户端
int framework_join_federated_client(framework_manager_t* manager,
                                  const char* server_address);

// 获取联邦学习状态
federated_training_stats_t* framework_get_federated_stats(framework_manager_t* manager);

// ==================== 多模态训练API ====================

// 初始化多模态训练
int framework_init_multimodal_training(framework_manager_t* manager,
                                     multimodal_model_config_t* config);

// 加载多模态数据
int framework_load_multimodal_data(framework_manager_t* manager,
                                  multimodal_dataset_t* dataset);

// 开始多模态训练
int framework_start_multimodal_training(framework_manager_t* manager);

// 多模态推理
tensor_t* framework_multimodal_predict(framework_manager_t* manager,
                                     tensor_t** modality_inputs,
                                     int num_modalities);

// 获取多模态训练状态
multimodal_training_stats_t* framework_get_multimodal_stats(framework_manager_t* manager);

// ==================== 复杂模型支持API ====================

// 创建大型CNN模型
nn_module_t* framework_create_large_cnn(framework_manager_t* manager,
                                      large_cnn_config_t* config);

// 创建Transformer模型
nn_module_t* framework_create_transformer(framework_manager_t* manager,
                                        transformer_config_t* config);

// 创建混合模型
nn_module_t* framework_create_hybrid_model(framework_manager_t* manager,
                                          hybrid_model_config_t* config);

// ==================== 优化和压缩API ====================

// 模型优化
int framework_optimize_model(framework_manager_t* manager,
                            optimization_config_t* config);

// 模型量化
int framework_quantize_model(framework_manager_t* manager,
                            quantization_config_t* config);

// 模型剪枝
int framework_prune_model(framework_manager_t* manager,
                         pruning_config_t* config);

// 模型蒸馏
int framework_distill_model(framework_manager_t* manager,
                          distillation_config_t* config);

// ==================== 监控和调试API ====================

// 获取框架统计
framework_manager_stats_t* framework_get_stats(framework_manager_t* manager);

// 获取训练历史
training_history_t* framework_get_training_history(framework_manager_t* manager);

// 性能分析
void framework_performance_profile(framework_manager_t* manager);

// 内存分析
void framework_memory_profile(framework_manager_t* manager);

// 调试信息
void framework_debug_info(framework_manager_t* manager);

// ==================== 回调管理API ====================

// 注册状态变化回调
int framework_register_state_callback(framework_manager_t* manager,
                                    void (*callback)(framework_state_t new_state, void* user_data),
                                    void* user_data);

// 注册错误回调
int framework_register_error_callback(framework_manager_t* manager,
                                     void (*callback)(int error_code, const char* error_message, void* user_data),
                                     void* user_data);

// 注册训练进度回调
int framework_register_progress_callback(framework_manager_t* manager,
                                       void (*callback)(float progress, void* user_data),
                                       void* user_data);

// ==================== 工具函数 ====================

// 配置验证
bool framework_validate_config(framework_manager_config_t* config);

// 框架版本信息
const char* framework_get_version(void);

// 支持的框架类型
const char** framework_get_supported_types(int* num_types);

// 检查功能支持
bool framework_supports_feature(framework_manager_t* manager, const char* feature_name);

// 错误处理
int framework_set_error(framework_manager_t* manager, int error_code, const char* error_message);

// 清理错误状态
int framework_clear_error(framework_manager_t* manager);

#ifdef __cplusplus
}
#endif

#endif // AI_FRAMEWORK_MANAGER_H