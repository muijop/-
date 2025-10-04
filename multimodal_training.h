#ifndef MULTIMODAL_TRAINING_H
#define MULTIMODAL_TRAINING_H

#include "nn.h"
#include "tensor.h"
#include "ai_framework_main.h"

#ifdef __cplusplus
extern "C" {
#endif

// 多模态数据类型枚举
typedef enum {
    MODALITY_TEXT = 0,
    MODALITY_IMAGE = 1,
    MODALITY_AUDIO = 2,
    MODALITY_VIDEO = 3,
    MODALITY_TIME_SERIES = 4,
    MODALITY_GRAPH = 5,
    MODALITY_POINT_CLOUD = 6
} modality_type_t;

// 多模态融合策略
typedef enum {
    FUSION_EARLY = 0,      // 早期融合
    FUSION_LATE = 1,       // 晚期融合
    FUSION_HYBRID = 2,     // 混合融合
    FUSION_CROSS_ATTENTION = 3, // 交叉注意力融合
    FUSION_TRANSFORMER = 4 // Transformer融合
} fusion_strategy_t;

// 多模态数据对齐方式
typedef enum {
    ALIGNMENT_NONE = 0,    // 不对齐
    ALIGNMENT_TEMPORAL = 1, // 时间对齐
    ALIGNMENT_SPATIAL = 2, // 空间对齐
    ALIGNMENT_SEMANTIC = 3 // 语义对齐
} alignment_method_t;

// 多模态数据样本
typedef struct {
    char* sample_id;
    modality_type_t modality_type;
    tensor_t* data;        // 模态数据
    tensor_t* labels;      // 标签
    size_t data_size;      // 数据大小
    char* metadata;        // 元数据
    int timestamp;         // 时间戳
} multimodal_sample_t;

// 多模态数据集
typedef struct {
    char* dataset_name;
    multimodal_sample_t** samples;
    int num_samples;
    int num_modalities;
    modality_type_t* modality_types;
    size_t* modality_sizes;
    char* dataset_path;
} multimodal_dataset_t;

// 多模态模型配置
typedef struct {
    char* model_name;
    int num_modalities;
    modality_type_t* input_modalities;
    size_t* input_dims;
    
    // 融合配置
    fusion_strategy_t fusion_strategy;
    alignment_method_t alignment_method;
    
    // 编码器配置
    nn_module_t** modality_encoders;
    size_t encoder_output_dim;
    
    // 融合网络配置
    nn_module_t* fusion_network;
    size_t fusion_output_dim;
    
    // 分类器配置
    nn_module_t* classifier;
    int num_classes;
    
    // 训练配置
    float learning_rate;
    int batch_size;
    int max_epochs;
    
    // 优化器配置
    optimizer_type_t optimizer_type;
    float weight_decay;
    
} multimodal_model_config_t;

// 多模态模型结构
typedef struct {
    char* model_id;
    multimodal_model_config_t* config;
    
    // 编码器网络
    nn_module_t** modality_encoders;
    
    // 融合网络
    nn_module_t* fusion_network;
    
    // 分类器网络
    nn_module_t* classifier;
    
    // 训练状态
    int is_trained;
    float current_loss;
    float current_accuracy;
    
    // 模型参数
    tensor_t** encoder_params;
    tensor_t* fusion_params;
    tensor_t* classifier_params;
    
} multimodal_model_t;

// 多模态训练配置
typedef struct {
    char* experiment_name;
    multimodal_dataset_t* dataset;
    multimodal_model_t* model;
    
    // 训练参数
    int batch_size;
    int num_epochs;
    float learning_rate;
    int validation_split;
    
    // 融合策略参数
    fusion_strategy_t fusion_strategy;
    alignment_method_t alignment_method;
    
    // 数据预处理
    int normalize_data;
    int augment_data;
    int shuffle_data;
    
    // 回调函数
    void (*on_epoch_end)(multimodal_model_t* model, int epoch, float loss, float accuracy);
    void (*on_batch_end)(multimodal_model_t* model, int batch, float loss);
    
} multimodal_training_config_t;

// 多模态训练统计
typedef struct {
    // 训练统计
    int current_epoch;
    int total_epochs;
    float training_loss;
    float training_accuracy;
    float validation_loss;
    float validation_accuracy;
    
    // 模态特定统计
    float* modality_losses;
    float* modality_accuracies;
    
    // 融合统计
    float fusion_loss;
    float fusion_accuracy;
    
    // 时间统计
    double epoch_time;
    double total_training_time;
    
    // 资源使用
    size_t memory_usage;
    size_t gpu_memory_usage;
    
} multimodal_training_stats_t;

// ==================== 多模态数据API ====================

// 创建多模态数据集
multimodal_dataset_t* multimodal_dataset_create(const char* dataset_name, 
                                               int num_modalities);

// 加载多模态数据
int multimodal_dataset_load(multimodal_dataset_t* dataset, 
                           const char* data_path);

// 添加多模态样本
int multimodal_dataset_add_sample(multimodal_dataset_t* dataset,
                                 multimodal_sample_t* sample);

// 预处理多模态数据
int multimodal_dataset_preprocess(multimodal_dataset_t* dataset,
                                int normalize,
                                int augment);

// 分割训练/验证集
int multimodal_dataset_split(multimodal_dataset_t* dataset,
                            float validation_ratio,
                            multimodal_dataset_t** train_set,
                            multimodal_dataset_t** val_set);

// 释放多模态数据集
void multimodal_dataset_free(multimodal_dataset_t* dataset);

// ==================== 多模态模型API ====================

// 创建多模态模型
multimodal_model_t* multimodal_model_create(multimodal_model_config_t* config);

// 初始化多模态模型
int multimodal_model_init(multimodal_model_t* model);

// 前向传播
int multimodal_model_forward(multimodal_model_t* model,
                            tensor_t** inputs,
                            tensor_t* output);

// 反向传播
int multimodal_model_backward(multimodal_model_t* model,
                             tensor_t** inputs,
                             tensor_t* targets,
                             tensor_t* gradients);

// 保存多模态模型
int multimodal_model_save(multimodal_model_t* model, const char* filepath);

// 加载多模态模型
multimodal_model_t* multimodal_model_load(const char* filepath);

// 释放多模态模型
void multimodal_model_free(multimodal_model_t* model);

// ==================== 多模态训练API ====================

// 多模态训练主函数
int multimodal_train(multimodal_model_t* model,
                     multimodal_dataset_t* dataset,
                     multimodal_training_config_t* config);

// 单批次训练
int multimodal_train_batch(multimodal_model_t* model,
                          multimodal_sample_t** batch_samples,
                          int batch_size);

// 多模态验证
float multimodal_validate(multimodal_model_t* model,
                         multimodal_dataset_t* val_dataset);

// 多模态测试
float multimodal_test(multimodal_model_t* model,
                    multimodal_dataset_t* test_dataset);

// ==================== 融合策略API ====================

// 早期融合
int multimodal_early_fusion(tensor_t** modality_features,
                           int num_modalities,
                           tensor_t* fused_features);

// 晚期融合
int multimodal_late_fusion(tensor_t** modality_logits,
                          int num_modalities,
                          tensor_t* fused_logits);

// 混合融合
int multimodal_hybrid_fusion(tensor_t** modality_features,
                            tensor_t** modality_logits,
                            int num_modalities,
                            tensor_t* fused_output);

// 交叉注意力融合
int multimodal_cross_attention_fusion(tensor_t** modality_features,
                                     int num_modalities,
                                     tensor_t* fused_features);

// Transformer融合
int multimodal_transformer_fusion(tensor_t** modality_features,
                                  int num_modalities,
                                  tensor_t* fused_features);

// ==================== 对齐方法API ====================

// 时间对齐
int multimodal_temporal_alignment(tensor_t** modality_data,
                                 int num_modalities,
                                 int* timestamps);

// 空间对齐
int multimodal_spatial_alignment(tensor_t** modality_data,
                                int num_modalities,
                                int* spatial_dims);

// 语义对齐
int multimodal_semantic_alignment(tensor_t** modality_features,
                                 int num_modalities,
                                 tensor_t* aligned_features);

// ==================== 特定模态处理API ====================

// 文本模态处理
int multimodal_process_text(tensor_t* raw_text, tensor_t* processed_text);

// 图像模态处理
int multimodal_process_image(tensor_t* raw_image, tensor_t* processed_image);

// 音频模态处理
int multimodal_process_audio(tensor_t* raw_audio, tensor_t* processed_audio);

// 视频模态处理
int multimodal_process_video(tensor_t* raw_video, tensor_t* processed_video);

// 时间序列处理
int multimodal_process_time_series(tensor_t* raw_series, tensor_t* processed_series);

// 图数据处理
int multimodal_process_graph(tensor_t* raw_graph, tensor_t* processed_graph);

// 点云数据处理
int multimodal_process_point_cloud(tensor_t* raw_points, tensor_t* processed_points);

// ==================== 中文多模态特定API ====================

// 中文文本编码
int multimodal_encode_chinese_text(const char* chinese_text, tensor_t* encoded_text);

// 中文图像标注处理
int multimodal_process_chinese_caption(const char* caption, tensor_t* encoded_caption);

// 中文多模态对齐
int multimodal_align_chinese_modalities(tensor_t* text_features,
                                       tensor_t* image_features,
                                       tensor_t* aligned_features);

// 中文多模态分类
int multimodal_chinese_classification(multimodal_model_t* model,
                                    tensor_t* text_input,
                                    tensor_t* image_input,
                                    tensor_t* output);

// ==================== 监控和调试API ====================

// 获取训练统计
multimodal_training_stats_t* multimodal_get_training_stats(multimodal_model_t* model);

// 打印模型信息
void multimodal_print_model_info(multimodal_model_t* model);

// 可视化融合结果
void multimodal_visualize_fusion(multimodal_model_t* model,
                                tensor_t** modality_inputs,
                                const char* output_path);

// 性能分析
void multimodal_performance_profile(multimodal_model_t* model);

// ==================== 工具函数 ====================

// 配置验证
bool multimodal_validate_config(multimodal_model_config_t* config);

// 数据格式转换
int multimodal_convert_data_format(tensor_t* input_data, 
                                  modality_type_t input_type,
                                  modality_type_t output_type,
                                  tensor_t* output_data);

// 模态特征提取
int multimodal_extract_features(tensor_t* raw_data,
                               modality_type_t modality_type,
                               tensor_t* features);

// 多模态相似度计算
float multimodal_calculate_similarity(tensor_t* features1,
                                     tensor_t* features2,
                                     modality_type_t modality_type);

#ifdef __cplusplus
}
#endif

#endif // MULTIMODAL_TRAINING_H