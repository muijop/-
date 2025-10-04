// 高级Transformer架构支持
// 提供BERT、GPT、Vision Transformer、Swin Transformer等现代Transformer变体

#ifndef ADVANCED_TRANSFORMER_H
#define ADVANCED_TRANSFORMER_H

#include "nn.h"
#include "tensor.h"
#include <stdbool.h>

// Transformer变体枚举
typedef enum {
    TRANSFORMER_BASE = 0,           // 基础Transformer
    TRANSFORMER_BERT,               // BERT风格
    TRANSFORMER_GPT,                 // GPT风格
    TRANSFORMER_VISION,             // Vision Transformer
    TRANSFORMER_SWIN,               // Swin Transformer
    TRANSFORMER_DEIT,               // DeiT (Data-efficient Image Transformer)
    TRANSFORMER_T5,                 // T5风格
    TRANSFORMER_ROBERTA,            // RoBERTa风格
    TRANSFORMER_ALBERT,             // ALBERT风格
    TRANSFORMER_XLNET,              // XLNet风格
    TRANSFORMER_ELECTRA,            // ELECTRA风格
    TRANSFORMER_LONGFORMER,         // Longformer风格
    TRANSFORMER_BIGBIRD,            // BigBird风格
    TRANSFORMER_PERFORMER,          // Performer风格
    TRANSFORMER_LINFORMER           // Linformer风格
} transformer_variant_t;

// 注意力机制类型
typedef enum {
    ATTENTION_SCALED_DOT_PRODUCT = 0, // 缩放点积注意力
    ATTENTION_MULTI_HEAD,            // 多头注意力
    ATTENTION_SPARSE,                // 稀疏注意力
    ATTENTION_LINEAR,                // 线性注意力
    ATTENTION_LOCAL,                 // 局部注意力
    ATTENTION_GLOBAL,                // 全局注意力
    ATTENTION_RELATIVE,              // 相对位置注意力
    ATTENTION_ROTARY                 // 旋转位置编码注意力
} attention_type_t;

// 位置编码类型
typedef enum {
    POSITION_SINUSOIDAL = 0,        // 正弦位置编码
    POSITION_LEARNABLE,             // 可学习位置编码
    POSITION_RELATIVE,               // 相对位置编码
    POSITION_ROTARY,                 // 旋转位置编码
    POSITION_ALIBI,                  // ALiBi位置编码
    POSITION_T5                      // T5相对位置编码
} position_encoding_type_t;

// 高级Transformer配置结构体
typedef struct {
    // 基础配置
    transformer_variant_t variant;   // Transformer变体
    int num_classes;                 // 分类数量（用于分类任务）
    bool pretrained;                 // 是否使用预训练权重
    
    // 模型维度配置
    int hidden_size;                 // 隐藏层大小
    int num_hidden_layers;           // 隐藏层数量
    int num_attention_heads;         // 注意力头数
    int intermediate_size;           // 中间层大小
    
    // 注意力配置
    attention_type_t attention_type; // 注意力类型
    float attention_dropout_prob;    // 注意力Dropout概率
    float hidden_dropout_prob;      // 隐藏层Dropout概率
    
    // 位置编码配置
    position_encoding_type_t position_encoding_type; // 位置编码类型
    int max_position_embeddings;    // 最大位置编码长度
    
    // 特定变体配置
    union {
        struct {
            bool add_cross_attention; // 是否添加交叉注意力
            bool is_decoder;          // 是否为解码器
            int num_decoder_layers;   // 解码器层数
        } bert_config;
        
        struct {
            int vocab_size;          // 词汇表大小
            int max_sequence_length; // 最大序列长度
            bool causal_mask;        // 是否使用因果掩码
        } gpt_config;
        
        struct {
            int image_size;          // 图像大小
            int patch_size;          // 补丁大小
            int num_channels;        // 通道数
            bool hybrid_backbone;    // 是否使用混合主干
        } vision_config;
        
        struct {
            int image_size;          // 图像大小
            int patch_size;          // 补丁大小
            int window_size;         // 窗口大小
            int shift_size;          // 移位大小
            int num_stages;          // 阶段数量
            int depths[4];           // 各阶段深度
            int num_heads[4];        // 各阶段头数
        } swin_config;
        
        struct {
            int encoder_vocab_size;  // 编码器词汇表大小
            int decoder_vocab_size;  // 解码器词汇表大小
            int num_beams;           // Beam搜索数量
            float length_penalty;    // 长度惩罚
        } t5_config;
    } arch_specific;
    
    // 性能优化配置
    bool use_gradient_checkpointing; // 是否使用梯度检查点
    bool use_mixed_precision;        // 是否使用混合精度
    bool use_flash_attention;         // 是否使用Flash Attention
    bool use_memory_efficient_attention; // 是否使用内存高效注意力
    
    // 内存优化配置
    bool use_activation_checkpointing; // 是否使用激活检查点
    int activation_offloading;       // 激活卸载配置
    
} advanced_transformer_config_t;

// Transformer编码器层（增强版）
typedef struct {
    // 自注意力层
    MultiheadAttention* self_attention;
    LayerNorm* attention_layer_norm;
    
    // 前馈网络
    Linear* intermediate;
    Linear* output;
    LayerNorm* output_layer_norm;
    
    // 配置
    float dropout_prob;
    bool use_pre_norm;               // 是否使用Pre-LN
    
    // 性能优化
    bool gradient_checkpointing;     // 梯度检查点
    
} advanced_transformer_encoder_layer_t;

// Transformer解码器层
typedef struct {
    // 自注意力层
    MultiheadAttention* self_attention;
    LayerNorm* self_attention_layer_norm;
    
    // 交叉注意力层
    MultiheadAttention* cross_attention;
    LayerNorm* cross_attention_layer_norm;
    
    // 前馈网络
    Linear* intermediate;
    Linear* output;
    LayerNorm* output_layer_norm;
    
    // 配置
    float dropout_prob;
    bool causal_mask;                // 因果掩码
    
} advanced_transformer_decoder_layer_t;

// BERT风格模型
typedef struct {
    advanced_transformer_config_t config;
    
    // 嵌入层
    Embedding* word_embeddings;      // 词嵌入
    Embedding* position_embeddings;  // 位置嵌入
    Embedding* token_type_embeddings; // 令牌类型嵌入
    LayerNorm* embeddings_layer_norm; // 嵌入层归一化
    
    // 编码器
    advanced_transformer_encoder_layer_t** encoder_layers;
    int num_encoder_layers;
    
    // 池化层
    Linear* pooler;                  // 池化层
    
    // 任务特定头
    Linear* classification_head;     // 分类头
    Linear* masked_lm_head;          // 掩码语言模型头
    Linear* next_sentence_head;      // 下一句预测头
    
} bert_model_t;

// GPT风格模型
typedef struct {
    advanced_transformer_config_t config;
    
    // 嵌入层
    Embedding* word_embeddings;      // 词嵌入
    Embedding* position_embeddings;  // 位置嵌入
    
    // 解码器
    advanced_transformer_decoder_layer_t** decoder_layers;
    int num_decoder_layers;
    
    // 输出层
    LayerNorm* final_layer_norm;     // 最终层归一化
    Linear* lm_head;                 // 语言模型头
    
} gpt_model_t;

// Vision Transformer模型
typedef struct {
    advanced_transformer_config_t config;
    
    // 补丁嵌入
    Linear* patch_embedding;         // 补丁嵌入
    Embedding* position_embeddings;  // 位置嵌入
    Embedding* class_token;          // 类别令牌
    
    // 编码器
    advanced_transformer_encoder_layer_t** encoder_layers;
    int num_encoder_layers;
    
    // 分类头
    LayerNorm* final_layer_norm;     // 最终层归一化
    Linear* classification_head;     // 分类头
    
} vision_transformer_model_t;

// Swin Transformer模型
typedef struct {
    advanced_transformer_config_t config;
    
    // 补丁嵌入
    Linear* patch_embedding;         // 补丁嵌入
    
    // 阶段
    struct {
        advanced_transformer_encoder_layer_t** layers;
        int num_layers;
        Linear* downsample;          // 下采样
    } stages[4];
    
    // 分类头
    LayerNorm* final_layer_norm;     // 最终层归一化
    Linear* classification_head;     // 分类头
    
} swin_transformer_model_t;

// 通用高级Transformer模型
typedef struct {
    advanced_transformer_config_t config;
    
    // 具体模型指针
    union {
        bert_model_t* bert;
        gpt_model_t* gpt;
        vision_transformer_model_t* vit;
        swin_transformer_model_t* swin;
        void* custom_model;
    } model;
    
    // 性能统计
    size_t num_parameters;           // 参数数量
    size_t flops;                    // 浮点运算次数
    size_t memory_usage;             // 内存使用量
    
} advanced_transformer_model_t;

// ==================== 高级Transformer创建函数 ====================

// 创建BERT模型
advanced_transformer_model_t* create_bert_model(int hidden_size,
                                               int num_hidden_layers,
                                               int num_attention_heads,
                                               int intermediate_size,
                                               int vocab_size,
                                               int max_position_embeddings,
                                               int num_classes);

// 创建GPT模型
advanced_transformer_model_t* create_gpt_model(int hidden_size,
                                              int num_hidden_layers,
                                              int num_attention_heads,
                                              int intermediate_size,
                                              int vocab_size,
                                              int max_position_embeddings);

// 创建Vision Transformer模型
advanced_transformer_model_t* create_vision_transformer(int image_size,
                                                       int patch_size,
                                                       int num_channels,
                                                       int hidden_size,
                                                       int num_hidden_layers,
                                                       int num_attention_heads,
                                                       int intermediate_size,
                                                       int num_classes);

// 创建Swin Transformer模型
advanced_transformer_model_t* create_swin_transformer(int image_size,
                                                     int patch_size,
                                                     int num_channels,
                                                     int hidden_size,
                                                     int depths[4],
                                                     int num_heads[4],
                                                     int window_size,
                                                     int num_classes);

// 使用自定义配置创建高级Transformer模型
advanced_transformer_model_t* create_advanced_transformer_model(advanced_transformer_config_t* config);

// ==================== 模型操作函数 ====================

// 前向传播
Tensor* advanced_transformer_forward(advanced_transformer_model_t* model, Tensor* input);

// 获取模型参数
Tensor** advanced_transformer_get_parameters(advanced_transformer_model_t* model, int* num_params);

// 获取模型状态字典
void* advanced_transformer_get_state_dict(advanced_transformer_model_t* model);

// 加载预训练权重
int advanced_transformer_load_pretrained(advanced_transformer_model_t* model, const char* weights_path);

// 保存模型权重
int advanced_transformer_save_weights(advanced_transformer_model_t* model, const char* save_path);

// 模型评估
float advanced_transformer_evaluate(advanced_transformer_model_t* model,
                                   Tensor* test_input,
                                   Tensor* test_target);

// 模型信息统计
void advanced_transformer_print_summary(advanced_transformer_model_t* model);

// 计算FLOPs
size_t advanced_transformer_calculate_flops(advanced_transformer_model_t* model, int sequence_length);

// 计算内存使用量
size_t advanced_transformer_calculate_memory_usage(advanced_transformer_model_t* model);

// ==================== 特定任务函数 ====================

// BERT掩码语言模型
Tensor* bert_masked_language_model(bert_model_t* model, Tensor* input_ids, Tensor* attention_mask);

// BERT下一句预测
Tensor* bert_next_sentence_prediction(bert_model_t* model, Tensor* input_ids, Tensor* token_type_ids);

// GPT文本生成
Tensor* gpt_text_generation(gpt_model_t* model, Tensor* input_ids, int max_length, float temperature);

// Vision Transformer图像分类
Tensor* vision_transformer_classify(vision_transformer_model_t* model, Tensor* images);

// ==================== 性能优化函数 ====================

// 启用梯度检查点
int advanced_transformer_enable_gradient_checkpointing(advanced_transformer_model_t* model);

// 启用混合精度训练
int advanced_transformer_enable_mixed_precision(advanced_transformer_model_t* model);

// 启用Flash Attention
int advanced_transformer_enable_flash_attention(advanced_transformer_model_t* model);

// 模型剪枝
int advanced_transformer_prune_model(advanced_transformer_model_t* model, float pruning_rate);

// 模型量化
int advanced_transformer_quantize_model(advanced_transformer_model_t* model, int quantization_bits);

// ==================== 工具函数 ====================

// 释放模型内存
void advanced_transformer_free(advanced_transformer_model_t* model);

// 模型克隆
advanced_transformer_model_t* advanced_transformer_clone(advanced_transformer_model_t* model);

// 模型比较
int advanced_transformer_compare(advanced_transformer_model_t* model1, advanced_transformer_model_t* model2);

// 获取变体名称
const char* advanced_transformer_get_variant_name(advanced_transformer_model_t* model);

#endif // ADVANCED_TRANSFORMER_H