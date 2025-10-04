#ifndef MODEL_COMPRESSION_H
#define MODEL_COMPRESSION_H

#include "nn_module.h"
#include "tensor.h"
#include <stdbool.h>

// ===========================================
// 模型压缩类型定义
// ===========================================

typedef enum {
    COMPRESSION_NONE = 0,
    COMPRESSION_PRUNING = 1,           // 剪枝压缩
    COMPRESSION_QUANTIZATION = 2,      // 量化压缩
    COMPRESSION_KNOWLEDGE_DISTILLATION = 3, // 知识蒸馏
    COMPRESSION_LOW_RANK = 4,          // 低秩分解
    COMPRESSION_STRUCTURED = 5,        // 结构化压缩
    COMPRESSION_AUTO = 6               // 自动压缩
} compression_type_t;

// ===========================================
// 剪枝配置结构体
// ===========================================

typedef struct {
    float sparsity_target;             // 目标稀疏度 (0.0-1.0)
    int pruning_method;                // 剪枝方法
    bool iterative_pruning;            // 是否迭代剪枝
    int pruning_frequency;             // 剪枝频率
    float min_weight_threshold;        // 最小权重阈值
} pruning_config_t;

// 剪枝方法枚举
typedef enum {
    PRUNE_MAGNITUDE = 0,              // 幅度剪枝
    PRUNE_GRADIENT = 1,               // 梯度剪枝
    PRUNE_STRUCTURED = 2,             // 结构化剪枝
    PRUNE_LOTTERY_TICKET = 3          // 彩票假设剪枝
} pruning_method_t;

// ===========================================
// 量化配置结构体
// ===========================================

typedef struct {
    int weight_bits;                   // 权重比特数
    int activation_bits;               // 激活比特数
    bool symmetric_quantization;      // 对称量化
    bool per_channel_quantization;     // 逐通道量化
    float quantization_range;          // 量化范围
    bool quantization_aware_training; // 量化感知训练
} quantization_config_t;

// ===========================================
// 知识蒸馏配置结构体
// ===========================================

typedef struct {
    float temperature;                 // 温度参数
    float alpha;                       // 蒸馏损失权重
    float beta;                        // 学生损失权重
    bool use_attention;                // 使用注意力蒸馏
    int distillation_layers;           // 蒸馏层数
} distillation_config_t;

// ===========================================
// 压缩结果分析结构体
// ===========================================

typedef struct {
    float original_size_mb;            // 原始模型大小 (MB)
    float compressed_size_mb;          // 压缩后大小 (MB)
    float compression_ratio;           // 压缩比
    float accuracy_drop;               // 精度下降
    float inference_speedup;           // 推理加速比
    float memory_reduction;            // 内存减少比例
} compression_result_t;

// ===========================================
// 模型压缩管理器结构体
// ===========================================

typedef struct model_compression_manager_s {
    compression_type_t compression_type;
    pruning_config_t pruning_config;
    quantization_config_t quantization_config;
    distillation_config_t distillation_config;
    
    // 内部状态
    bool is_initialized;
    int compression_stage;
    float current_sparsity;
    
    // 回调函数
    void (*progress_callback)(int progress, const char* message);
    void (*error_callback)(const char* error_message);
} model_compression_manager_t;

// ===========================================
// API 函数声明
// ===========================================

// 压缩管理器创建和销毁
model_compression_manager_t* create_model_compression_manager(void);
void destroy_model_compression_manager(model_compression_manager_t* manager);

// 压缩配置设置
int set_compression_type(model_compression_manager_t* manager, compression_type_t type);
int configure_pruning(model_compression_manager_t* manager, const pruning_config_t* config);
int configure_quantization(model_compression_manager_t* manager, const quantization_config_t* config);
int configure_distillation(model_compression_manager_t* manager, const distillation_config_t* config);

// 模型压缩执行
int compress_model(model_compression_manager_t* manager, nn_module_t* model);
int iterative_compress_model(model_compression_manager_t* manager, nn_module_t* model, int max_iterations);

// 压缩分析和评估
compression_result_t analyze_compression_result(const model_compression_manager_t* manager, const nn_module_t* model);
int validate_compressed_model(const nn_module_t* compressed_model, const nn_module_t* original_model);

// 工具函数
float calculate_model_sparsity(const nn_module_t* model);
float calculate_model_size_mb(const nn_module_t* model);
int export_compressed_model(const nn_module_t* model, const char* filename);

// 高级压缩算法
int apply_magnitude_pruning(nn_module_t* model, float sparsity_target);
int apply_structured_pruning(nn_module_t* model, int pruning_pattern);
int apply_quantization_aware_training(nn_module_t* model, const quantization_config_t* config);
int apply_knowledge_distillation(nn_module_t* teacher_model, nn_module_t* student_model, 
                                const distillation_config_t* config);

// 增强的剪枝和稀疏化功能
int apply_iterative_pruning(nn_module_t* model, float final_sparsity, int num_iterations);
int apply_global_pruning(nn_module_t* model, float sparsity_target);
int apply_lottery_ticket_pruning(nn_module_t* model, float sparsity_target);
int apply_gradual_pruning(nn_module_t* model, float initial_sparsity, float final_sparsity, int epochs);

// 增强的量化功能
int apply_mixed_precision_quantization(nn_module_t* model, const int* layer_bits);
int apply_post_training_quantization(nn_module_t* model, const quantization_config_t* config);
int apply_channel_wise_quantization(nn_module_t* model, int weight_bits);

// 增强的知识蒸馏功能
int apply_multi_teacher_distillation(nn_module_t** teacher_models, int num_teachers, 
                                    nn_module_t* student_model, const distillation_config_t* config);
int apply_layer_wise_distillation(nn_module_t* teacher_model, nn_module_t* student_model, 
                                 const distillation_config_t* config);
int apply_self_distillation(nn_module_t* model, const distillation_config_t* config);

// 稀疏化工具函数
float calculate_model_sparsity_detailed(const nn_module_t* model, float* weight_sparsity, float* bias_sparsity);
int visualize_sparsity_pattern(const nn_module_t* model, const char* filename);
int apply_sparsity_regularization(nn_module_t* model, float lambda);

// 回调函数设置
void set_compression_progress_callback(model_compression_manager_t* manager, 
                                      void (*callback)(int progress, const char* message));
void set_compression_error_callback(model_compression_manager_t* manager, 
                                   void (*callback)(const char* error_message));

#endif // MODEL_COMPRESSION_H