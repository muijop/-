#ifndef AI_FRAMEWORK_ENHANCED_H
#define AI_FRAMEWORK_ENHANCED_H

#include "ai_framework_unified_fixed.h"
#include "optimizer.h"
#include "dataloader.h"
#include "chinese_nlp_support.h"
#include "tensor_autograd.h"
#include "distributed_training.h"
#include "full_scene_deployment.h"
#include "keras_style_api.h"
#include "static_graph_optimizer.h"
#include "dynamic_graph.h"

// 多模态数据类型
typedef enum {
    MODALITY_TEXT = 0,
    MODALITY_IMAGE,
    MODALITY_AUDIO,
    MODALITY_VIDEO,
    MODALITY_MULTIMODAL
} modality_type_t;

// 中文图片多模态配置
typedef struct {
    // 文本模态配置
    ChineseTokenizerConfig text_config;
    ChineseTextPreprocessorConfig text_preprocessor_config;
    
    // 图像模态配置
    struct {
        bool enable_chinese_ocr;           // 中文OCR支持
        bool enable_image_augmentation;     // 图像增强
        bool enable_multiscale_processing; // 多尺度处理
        char* image_backend;               // OpenCV/PIL/ImageIO
        int image_resolution;              // 图像分辨率
        bool enable_chinese_captioning;    // 中文图像描述
    } image_config;
    
    // 多模态融合配置
    struct {
        bool enable_cross_modal_attention; // 跨模态注意力
        bool enable_multimodal_fusion;     // 多模态融合
        char* fusion_strategy;             // 融合策略
        int fusion_dim;                    // 融合维度
    } fusion_config;
    
    // 部署配置
    struct {
        bool enable_mobile_deployment;     // 移动端部署
        bool enable_edge_deployment;       // 边缘部署
        bool enable_cloud_deployment;       // 云端部署
        char* target_platform;             // 目标平台
    } deployment_config;
    
} chinese_multimodal_config_t;

// 增强优化器配置
typedef struct {
    // 基础优化器
    char* optimizer_type;                  // SGD/Adam/RMSprop等
    float learning_rate;
    float momentum;
    float weight_decay;
    
    // 梯度优化
    bool enable_gradient_clipping;         // 梯度裁剪
    float clip_value;                      // 裁剪值
    bool enable_gradient_accumulation;      // 梯度累积
    int accumulation_steps;                // 累积步数
    
    // 学习率调度
    bool enable_lr_scheduling;             // 学习率调度
    char* scheduler_type;                  // Step/Cosine/Linear
    int warmup_steps;                      // 预热步数
    int decay_steps;                       // 衰减步数
    
    // 混合精度训练
    bool enable_mixed_precision;           // 混合精度
    char* precision_mode;                  // fp16/bf16/fp32
    
    // 分布式优化
    bool enable_distributed_optimization;  // 分布式优化
    char* distributed_strategy;             // 分布式策略
    
} enhanced_optimizer_config_t;

// 增强数据加载器配置
typedef struct {
    // 多模态数据支持
    modality_type_t modality_type;         // 数据类型
    bool enable_multimodal_loading;        // 多模态加载
    
    // 中文文本处理
    ChineseTokenizerConfig tokenizer_config;
    ChineseTextPreprocessorConfig preprocessor_config;
    
    // 图像处理
    struct {
        bool enable_chinese_image_processing; // 中文图像处理
        int image_width;                      // 图像宽度
        int image_height;                     // 图像高度
        bool enable_data_augmentation;        // 数据增强
        char* augmentation_pipeline;           // 增强管道
    } image_processing;
    
    // 批处理优化
    bool enable_dynamic_batching;           // 动态批处理
    int max_batch_size;                     // 最大批大小
    bool enable_memory_pinning;             // 内存固定
    
    // 缓存优化
    bool enable_data_caching;               // 数据缓存
    int cache_size;                         // 缓存大小
    char* cache_strategy;                   // 缓存策略
    
} enhanced_dataloader_config_t;

// 统一AI框架增强接口

// 1. 多模态模型创建
unified_model_t* ai_framework_create_multimodal_model(
    chinese_multimodal_config_t* config,
    modality_type_t* modalities,
    int num_modalities
);

// 2. 增强优化器创建
Optimizer* ai_framework_create_enhanced_optimizer(
    enhanced_optimizer_config_t* config,
    unified_tensor_t** parameters,
    int num_parameters
);

// 3. 增强数据加载器创建
DataLoader* ai_framework_create_enhanced_dataloader(
    enhanced_dataloader_config_t* config,
    char** data_paths,
    int num_paths
);

// 4. 中文图片多模态训练
int ai_framework_train_chinese_multimodal(
    unified_model_t* model,
    DataLoader* train_loader,
    DataLoader* val_loader,
    enhanced_optimizer_config_t* optimizer_config,
    chinese_multimodal_config_t* multimodal_config,
    int num_epochs
);

// 5. 多端部署接口
int ai_framework_deploy_multimodal_model(
    unified_model_t* model,
    chinese_multimodal_config_t* config,
    char* deployment_target
);

// 6. 性能优化接口
int ai_framework_optimize_multimodal_performance(
    unified_model_t* model,
    chinese_multimodal_config_t* config
);

// 7. 内存优化接口
int ai_framework_optimize_multimodal_memory(
    unified_model_t* model,
    chinese_multimodal_config_t* config
);

// 8. 调试与分析接口
int ai_framework_analyze_multimodal_model(
    unified_model_t* model,
    chinese_multimodal_config_t* config
);

// 工具函数

// 配置验证
bool ai_framework_validate_multimodal_config(chinese_multimodal_config_t* config);

// 性能基准测试
void ai_framework_benchmark_multimodal_performance(chinese_multimodal_config_t* config);

// 内存使用分析
void ai_framework_analyze_multimodal_memory_usage(unified_model_t* model);

// 模型压缩接口
int ai_framework_compress_multimodal_model(
    unified_model_t* model,
    float compression_ratio
);

// 量化接口
int ai_framework_quantize_multimodal_model(
    unified_model_t* model,
    char* quantization_type
);

#endif // AI_FRAMEWORK_ENHANCED_H