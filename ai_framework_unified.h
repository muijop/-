#ifndef AI_FRAMEWORK_UNIFIED_H
#define AI_FRAMEWORK_UNIFIED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

// 统一框架版本信息
#define AI_FRAMEWORK_VERSION "1.0.0"
#define AI_FRAMEWORK_NAME "UnifiedAI"
#define AI_FRAMEWORK_BUILD_DATE __DATE__
#define AI_FRAMEWORK_BUILD_TIME __TIME__

// 包含所有子系统
#include "dynamic_graph.h"
#include "static_graph_optimizer.h"
#include "distributed_training.h"
#include "high_performance_kernels.h"
#include "chinese_nlp_support.h"
#include "full_scene_deployment.h"
#include "keras_style_api.h"

// 统一错误码
typedef enum {
    AI_SUCCESS = 0,
    AI_ERROR_INVALID_PARAM = -1,
    AI_ERROR_OUT_OF_MEMORY = -2,
    AI_ERROR_NOT_INITIALIZED = -3,
    AI_ERROR_INVALID_OPERATION = -4,
    AI_ERROR_DISTRIBUTED_FAILURE = -5,
    AI_ERROR_DEPLOYMENT_FAILURE = -6,
    AI_ERROR_CHINESE_NLP_FAILURE = -7,
    AI_ERROR_KERNEL_FAILURE = -8,
    AI_ERROR_OPTIMIZER_FAILURE = -9,
    AI_ERROR_GRAPH_FAILURE = -10
} ai_error_t;

// 框架配置结构体
typedef struct {
    // 计算图配置
    bool use_dynamic_graph;           // PyTorch风格动态图
    bool use_static_optimization;     // TensorFlow风格静态优化
    bool enable_jit;                  // JAX风格JIT编译
    
    // 分布式配置
    bool enable_distributed;          // 分布式训练
    char* distributed_backend;        // NCCL/MPI/Gloo/RPC
    int world_size;                   // 总进程数
    int rank;                         // 当前进程排名
    
    // 部署配置
    bool enable_deployment;           // 全场景部署
    char* deployment_target;          // cloud/edge/mobile/embedded/iot/web
    char* hardware_platform;          // CPU/GPU/NPU/TPU/ASCEND
    
    // 中文优化
    bool enable_chinese_optimization; // 中文NLP优化
    char* chinese_encoding;          // UTF-8/GBK/BIG5
    
    // 性能配置
    bool enable_mixed_precision;      // 混合精度训练
    bool enable_memory_optimization;    // 内存优化
    bool enable_parallel_optimization;  // 并行优化
    int num_threads;                  // 线程数
    
    // Keras风格配置
    bool use_keras_style_api;         // Keras风格API
    bool enable_model_checkpointing;  // 模型检查点
    bool enable_early_stopping;       // 早停
    
    // 调试配置
    bool enable_debug_mode;           // 调试模式
    bool enable_profiling;            // 性能分析
    char* log_level;                  // DEBUG/INFO/WARN/ERROR
} ai_framework_config_t;

// 统一AI框架上下文
typedef struct {
    ai_framework_config_t config;
    dynamic_graph_t* dynamic_graph;
    static_graph_optimizer_t* static_optimizer;
    distributed_trainer_t* distributed_trainer;
    high_performance_kernel_engine_t* kernel_engine;
    chinese_nlp_engine_t* chinese_nlp_engine;
    deployment_engine_t* deployment_engine;
    keras_model_t* keras_model;
    
    // 全局状态
    bool is_initialized;
    ai_error_t last_error;
    char error_message[1024];
    double initialization_time;
} ai_framework_t;

// 统一张量结构体（兼容所有框架风格）
typedef struct {
    // 基础属性
    float* data;
    int* shape;
    int ndim;
    int size;
    
    // 设备信息
    char* device;           // cpu/cuda/npu/tpu
    int device_id;          // 设备ID
    
    // 计算图信息
    bool requires_grad;     // 是否需要梯度
    tensor_t* grad;         // 梯度张量
    char* op_type;          // 操作类型
    tensor_t** parents;     // 父节点
    int num_parents;        // 父节点数量
    
    // 分布式信息
    bool is_distributed;    // 是否分布式
    int tensor_partition;   // 张量分区信息
    
    // 中文优化信息
    bool is_chinese_optimized;  // 中文优化
    char* encoding;               // 编码信息
    
    // 性能优化
    bool is_memory_optimized;     // 内存优化
    bool is_jit_compiled;         // JIT编译
    
    // 元数据
    char* name;             // 张量名称
    char* dtype;            // 数据类型
    void* metadata;           // 额外元数据
} unified_tensor_t;

// 统一模型结构体
typedef struct {
    char* name;                     // 模型名称
    char* architecture;             // 架构类型
    unified_tensor_t** parameters;  // 模型参数
    int num_parameters;              // 参数数量
    
    // 计算图
    dynamic_graph_t* dynamic_graph;
    static_graph_optimizer_t* static_optimizer;
    
    // 分布式
    distributed_trainer_t* distributed_trainer;
    
    // 部署
    deployment_engine_t* deployment_engine;
    
    // Keras风格
    keras_model_t* keras_model;
    
    // 中文优化
    chinese_nlp_engine_t* chinese_nlp_engine;
    
    // 性能
    high_performance_kernel_engine_t* kernel_engine;
    
    // 元数据
    char* description;              // 模型描述
    char* author;                   // 作者
    char* version;                  // 版本
    char* framework_version;        // 框架版本
    void* metadata;                 // 额外元数据
} unified_model_t;

// 统一优化器结构体
typedef struct {
    char* name;                     // 优化器名称
    char* type;                     // 优化器类型
    float learning_rate;            // 学习率
    float momentum;                 // 动量
    float weight_decay;             // 权重衰减
    
    // 分布式优化
    bool is_distributed_optimized;  // 分布式优化
    
    // 混合精度
    bool enable_mixed_precision;    // 混合精度
    
    // 中文优化
    bool enable_chinese_optimization; // 中文优化
    
    // 性能优化
    bool enable_memory_optimization;  // 内存优化
    bool enable_parallel_optimization; // 并行优化
    
    void* internal_state;           // 内部状态
} unified_optimizer_t;

// 统一数据加载器结构体
typedef struct {
    char* name;                     // 名称
    char* type;                     // 类型
    char* data_path;                // 数据路径
    
    // 分布式
    bool is_distributed;            // 分布式
    int world_size;                 // 总进程数
    int rank;                       // 当前排名
    
    // 中文优化
    bool enable_chinese_optimization; // 中文优化
    char* chinese_encoding;           // 中文编码
    
    // 性能优化
    bool enable_memory_optimization;  // 内存优化
    int num_workers;                  // 工作线程数
    int batch_size;                   // 批次大小
    bool shuffle;                     // 是否打乱
    
    void* internal_loader;          // 内部加载器
} unified_dataloader_t;

// 框架初始化与销毁
ai_framework_t* ai_framework_create(const ai_framework_config_t* config);
ai_error_t ai_framework_destroy(ai_framework_t* framework);
ai_error_t ai_framework_initialize(ai_framework_t* framework);
ai_error_t ai_framework_finalize(ai_framework_t* framework);

// 配置管理
ai_framework_config_t* ai_framework_config_create(void);
ai_error_t ai_framework_config_destroy(ai_framework_config_t* config);
ai_error_t ai_framework_config_set_default(ai_framework_config_t* config);
ai_error_t ai_framework_config_from_string(ai_framework_config_t* config, const char* config_str);
char* ai_framework_config_to_string(const ai_framework_config_t* config);

// 统一张量操作
unified_tensor_t* unified_tensor_create(const int* shape, int ndim, const char* device);
ai_error_t unified_tensor_destroy(unified_tensor_t* tensor);
ai_error_t unified_tensor_fill(unified_tensor_t* tensor, float value);
ai_error_t unified_tensor_random(unified_tensor_t* tensor, float min_val, float max_val);
ai_error_t unified_tensor_reshape(unified_tensor_t* tensor, const int* new_shape, int new_ndim);
unified_tensor_t* unified_tensor_add(unified_tensor_t* a, unified_tensor_t* b);
unified_tensor_t* unified_tensor_matmul(unified_tensor_t* a, unified_tensor_t* b);
unified_tensor_t* unified_tensor_relu(unified_tensor_t* tensor);

// 统一模型操作
unified_model_t* unified_model_create(const char* name, const char* architecture);
ai_error_t unified_model_destroy(unified_model_t* model);
ai_error_t unified_model_add_layer(unified_model_t* model, const char* layer_type, void* layer_config);
ai_error_t unified_model_compile(unified_model_t* model, unified_optimizer_t* optimizer, const char* loss_function);
ai_error_t unified_model_fit(unified_model_t* model, unified_dataloader_t* train_loader, 
                            unified_dataloader_t* val_loader, int epochs);
ai_error_t unified_model_evaluate(unified_model_t* model, unified_dataloader_t* test_loader);
unified_tensor_t* unified_model_predict(unified_model_t* model, unified_tensor_t* input);

// 统一优化器操作
unified_optimizer_t* unified_optimizer_create(const char* name, float learning_rate);
ai_error_t unified_optimizer_destroy(unified_optimizer_t* optimizer);
ai_error_t unified_optimizer_set_params(unified_optimizer_t* optimizer, const char* param_name, float value);

// 统一数据加载器操作
unified_dataloader_t* unified_dataloader_create(const char* name, const char* data_path, int batch_size);
ai_error_t unified_dataloader_destroy(unified_dataloader_t* loader);
ai_error_t unified_dataloader_set_chinese_optimization(unified_dataloader_t* loader, bool enable, const char* encoding);

// 高级功能
ai_error_t ai_framework_enable_chinese_optimization(ai_framework_t* framework, bool enable);
ai_error_t ai_framework_enable_distributed_training(ai_framework_t* framework, const char* backend, int world_size, int rank);
ai_error_t ai_framework_enable_deployment(ai_framework_t* framework, const char* target, const char* hardware);
ai_error_t ai_framework_enable_jit_compilation(ai_framework_t* framework, bool enable);
ai_error_t ai_framework_enable_mixed_precision(ai_framework_t* framework, bool enable);

// 模型保存与加载
ai_error_t unified_model_save(unified_model_t* model, const char* filepath);
unified_model_t* unified_model_load(const char* filepath);
ai_error_t unified_model_export_for_deployment(unified_model_t* model, const char* export_path, const char* format);

// 性能分析与调试
ai_error_t ai_framework_start_profiling(ai_framework_t* framework);
ai_error_t ai_framework_stop_profiling(ai_framework_t* framework);
ai_error_t ai_framework_print_model_summary(unified_model_t* model);
ai_error_t ai_framework_set_log_level(ai_framework_t* framework, const char* level);

// 错误处理
const char* ai_framework_get_error_string(ai_error_t error);
ai_error_t ai_framework_get_last_error(ai_framework_t* framework);
const char* ai_framework_get_error_message(ai_framework_t* framework);

// 版本信息
const char* ai_framework_get_version(void);
const char* ai_framework_get_build_info(void);

// 实用工具
float ai_framework_get_memory_usage(void);
int ai_framework_get_available_devices(char*** devices, int* num_devices);
ai_error_t ai_framework_set_device(ai_framework_t* framework, const char* device);

// 中文特定功能
ai_error_t ai_framework_load_chinese_pretrained_model(unified_model_t* model, const char* model_name);
ai_error_t ai_framework_chinese_text_preprocessing(unified_tensor_t* text_tensor, const char* encoding);
ai_error_t ai_framework_enable_chinese_tokenization(unified_dataloader_t* loader, const char* tokenizer_type);

// 分布式特定功能
ai_error_t ai_framework_all_reduce(unified_tensor_t* tensor, const char* op);
ai_error_t ai_framework_broadcast(unified_tensor_t* tensor, int root_rank);
ai_error_t ai_framework_barrier(ai_framework_t* framework);

// 部署特定功能
ai_error_t ai_framework_quantize_model(unified_model_t* model, const char* quantization_type);
ai_error_t ai_framework_compress_model(unified_model_t* model, const char* compression_type);
ai_error_t ai_framework_convert_model_format(unified_model_t* model, const char* target_format);

// 自动混合精度
ai_error_t ai_framework_autocast_enable(ai_framework_t* framework);
ai_error_t ai_framework_autocast_disable(ai_framework_t* framework);
bool ai_framework_is_autocast_enabled(ai_framework_t* framework);

// 高级优化
ai_error_t ai_framework_enable_operator_fusion(ai_framework_t* framework, bool enable);
ai_error_t ai_framework_enable_memory_planning(ai_framework_t* framework, bool enable);
ai_error_t ai_framework_enable_gradient_checkpointing(ai_framework_t* framework, bool enable);

// 联邦学习
ai_error_t ai_framework_enable_federated_learning(ai_framework_t* framework, const char* aggregator);
ai_error_t ai_framework_federated_model_aggregation(unified_model_t* global_model, unified_model_t** local_models, int num_clients);

// 模型解释性
ai_error_t ai_framework_enable_model_explainability(ai_framework_t* framework, const char* method);
ai_error_t ai_framework_get_feature_importance(unified_model_t* model, unified_tensor_t* input, float** importance_scores);

// 自动超参数优化
ai_error_t ai_framework_enable_auto_hyperparameter_tuning(ai_framework_t* framework, const char* algorithm);
ai_error_t ai_framework_set_hyperparameter_search_space(unified_optimizer_t* optimizer, const char* param_name, float min_val, float max_val);

// ===========================================
// 八大关键功能增强接口
// ===========================================

// 1. 计算图与执行系统增强
ai_error_t static_graph_add_optimization_pass(static_graph_optimizer_t* optimizer, const char* pass_type);
ai_error_t dynamic_graph_enable_tracing(dynamic_graph_t* graph, bool enable);
void* dynamic_graph_export_graph(dynamic_graph_t* graph, const char* format);

// 2. 自动微分系统完善
ai_error_t autograd_set_gradient_clipping(ai_framework_t* framework, float max_norm, float norm_type);
ai_error_t autograd_enable_gradient_accumulation(ai_framework_t* framework, int steps);
ai_error_t autograd_register_custom_gradient(unified_tensor_t* (*forward_op)(unified_tensor_t*),
                                            unified_tensor_t* (*backward_op)(unified_tensor_t*),
                                            const char* name);

// 3. 分布式训练增强
ai_error_t distributed_set_parallel_strategy(distributed_trainer_t* trainer, const char* strategy);
ai_error_t distributed_enable_gradient_compression(distributed_trainer_t* trainer, const char* method);
ai_error_t distributed_set_fault_tolerance(distributed_trainer_t* trainer, bool enable);

// 4. 性能优化模块
ai_error_t performance_set_precision_policy(ai_framework_t* framework, const char* policy);
ai_error_t performance_enable_gradient_checkpointing(unified_model_t* model, bool enable);
ai_error_t performance_tune_kernels(ai_framework_t* framework, const char* device);

// 5. 模型构建与管理
typedef struct {
    int64_t num_parameters;
    int64_t num_flops;
    float memory_usage_mb;
    float inference_latency_ms;
    float training_memory_mb;
} model_analysis_t;

unified_model_t* model_zoo_load(const char* model_name, const char* pretrained_dataset);
ai_error_t model_analyze(unified_model_t* model, unified_tensor_t* input_sample, model_analysis_t* result);
ai_error_t model_save_version(unified_model_t* model, const char* version, const char* description);

// 6. 数据处理增强
typedef struct {
    int batch_size;
    int num_workers;
    bool shuffle;
    bool enable_prefetch;
    bool enable_memory_mapping;
} dataloader_config_t;

unified_dataloader_t* create_advanced_dataloader(const dataloader_config_t* config);
ai_error_t add_data_transform(unified_dataloader_t* loader, const char* transform_type, void* params);
ai_error_t chinese_nlp_add_tokenizer(chinese_nlp_engine_t* engine, const char* tokenizer_type);

// 7. 部署与集成能力
ai_error_t model_export(unified_model_t* model, const char* format, const char* path);
ai_error_t model_quantize(unified_model_t* model, const char* quant_type, float accuracy_threshold);
ai_error_t deployment_create_service(unified_model_t* model, const char* service_name, int port);

// 8. 开发工具与调试支持
ai_error_t framework_enable_debug(ai_framework_t* framework, bool enable, const char* log_path);
ai_error_t framework_profile(ai_framework_t* framework, const char* output_file);
char* framework_generate_documentation(ai_framework_t* framework);

// 实用工具函数
ai_error_t ai_framework_check_status(ai_framework_t* framework);

#ifdef __cplusplus
}
#endif

#endif // AI_FRAMEWORK_UNIFIED_H