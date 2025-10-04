// 大型CNN架构支持
// 提供ResNet、VGG、Inception、EfficientNet等大型CNN架构

#ifndef LARGE_CNN_ARCHITECTURES_H
#define LARGE_CNN_ARCHITECTURES_H

#include "nn_layers.h"
#include "tensor.h"
#include <stdbool.h>

// ResNet变体枚举
typedef enum {
    RESNET18 = 0,
    RESNET34,
    RESNET50,
    RESNET101,
    RESNET152,
    RESNEXT50_32X4D,
    RESNEXT101_32X8D,
    WIDE_RESNET50_2,
    WIDE_RESNET101_2
} resnet_variant_t;

// VGG变体枚举
typedef enum {
    VGG11 = 0,
    VGG13,
    VGG16,
    VGG19,
    VGG11_BN,
    VGG13_BN,
    VGG16_BN,
    VGG19_BN
} vgg_variant_t;

// Inception变体枚举
typedef enum {
    INCEPTION_V3 = 0,
    INCEPTION_RESNET_V2,
    GOOGLENET
} inception_variant_t;

// EfficientNet变体枚举
typedef enum {
    EFFICIENTNET_B0 = 0,
    EFFICIENTNET_B1,
    EFFICIENTNET_B2,
    EFFICIENTNET_B3,
    EFFICIENTNET_B4,
    EFFICIENTNET_B5,
    EFFICIENTNET_B6,
    EFFICIENTNET_B7
} efficientnet_variant_t;

// DenseNet变体枚举
typedef enum {
    DENSENET121 = 0,
    DENSENET161,
    DENSENET169,
    DENSENET201
} densenet_variant_t;

// 大型CNN配置结构体
typedef struct {
    // 基础配置
    int num_classes;                    // 分类数量
    bool pretrained;                    // 是否使用预训练权重
    bool progress;                      // 是否显示进度
    
    // 输入配置
    int input_channels;                 // 输入通道数
    int input_height;                   // 输入高度
    int input_width;                    // 输入宽度
    
    // 架构特定配置
    union {
        struct {
            resnet_variant_t variant;   // ResNet变体
            bool zero_init_residual;    // 是否零初始化残差
            int groups;                  // 分组数
            int width_per_group;        // 每组宽度
            bool replace_stride_with_dilation[3]; // 是否用膨胀卷积替换步长
        } resnet_config;
        
        struct {
            vgg_variant_t variant;      // VGG变体
            int batch_norm;             // 批归一化配置
            int dropout_rate;           // Dropout率
        } vgg_config;
        
        struct {
            inception_variant_t variant; // Inception变体
            bool aux_logits;             // 是否使用辅助分类器
            bool transform_input;        // 是否转换输入
        } inception_config;
        
        struct {
            efficientnet_variant_t variant; // EfficientNet变体
            float dropout_rate;         // Dropout率
            float stochastic_depth_prob; // 随机深度概率
        } efficientnet_config;
        
        struct {
            densenet_variant_t variant; // DenseNet变体
            float dropout_rate;          // Dropout率
            int memory_efficient;       // 内存效率配置
        } densenet_config;
    } arch_specific;
    
    // 性能优化配置
    bool use_separable_conv;            // 是否使用可分离卷积
    bool use_depthwise_conv;           // 是否使用深度可分离卷积
    bool use_group_norm;               // 是否使用组归一化
    bool use_attention;                // 是否使用注意力机制
    
    // 内存优化配置
    bool gradient_checkpointing;       // 是否使用梯度检查点
    bool use_mixed_precision;          // 是否使用混合精度
    int activation_checkpointing;      // 激活检查点配置
    
} large_cnn_config_t;

// ResNet基本块结构体
typedef struct {
    layer_t* conv1;                    // 第一个卷积层
    layer_t* bn1;                      // 第一个批归一化层
    layer_t* relu1;                    // 第一个ReLU层
    layer_t* conv2;                    // 第二个卷积层
    layer_t* bn2;                      // 第二个批归一化层
    layer_t* downsample;                // 下采样层（可选）
    layer_t* shortcut;                 // 快捷连接层
} resnet_basic_block_t;

// ResNet瓶颈块结构体
typedef struct {
    layer_t* conv1;                    // 1x1卷积层
    layer_t* bn1;                      // 第一个批归一化层
    layer_t* conv2;                    // 3x3卷积层
    layer_t* bn2;                      // 第二个批归一化层
    layer_t* conv3;                    // 1x1卷积层
    layer_t* bn3;                      // 第三个批归一化层
    layer_t* downsample;               // 下采样层（可选）
    layer_t* shortcut;                 // 快捷连接层
} resnet_bottleneck_block_t;

// VGG块结构体
typedef struct {
    layer_t** conv_layers;             // 卷积层数组
    int num_conv_layers;               // 卷积层数量
    layer_t* maxpool;                  // 最大池化层
} vgg_block_t;

// Inception块结构体
typedef struct {
    layer_t* branch1x1;                // 1x1分支
    layer_t* branch3x3_reduce;         // 3x3分支的1x1降维
    layer_t* branch3x3;                // 3x3分支
    layer_t* branch5x5_reduce;         // 5x5分支的1x1降维
    layer_t* branch5x5;                // 5x5分支
    layer_t* branch_pool;              // 池化分支
    layer_t* concat;                   // 连接层
} inception_block_t;

// EfficientNet MBConv块结构体
typedef struct {
    layer_t* expand_conv;              // 扩展卷积
    layer_t* expand_bn;                // 扩展批归一化
    layer_t* depthwise_conv;           // 深度可分离卷积
    layer_t* depthwise_bn;             // 深度可分离批归一化
    layer_t* se_reduce;                // SE模块降维
    layer_t* se_expand;                // SE模块扩展
    layer_t* project_conv;             // 投影卷积
    layer_t* project_bn;               // 投影批归一化
    float drop_connect_rate;           // Drop Connect率
} efficientnet_mbconv_block_t;

// 大型CNN模型结构体
typedef struct {
    large_cnn_config_t config;        // 配置
    
    // 输入处理层
    layer_t* stem_conv;                // 主干卷积层
    layer_t* stem_bn;                  // 主干批归一化
    layer_t* stem_relu;                // 主干ReLU
    layer_t* stem_pool;                // 主干池化
    
    // 特征提取层
    void** blocks;                     // 块数组（具体类型取决于架构）
    int num_blocks;                    // 块数量
    
    // 分类头
    layer_t* global_pool;              // 全局池化
    layer_t* classifier;               // 分类器
    layer_t* dropout;                  // Dropout层
    
    // 性能统计
    size_t num_parameters;             // 参数数量
    size_t flops;                      // 浮点运算次数
    size_t memory_usage;               // 内存使用量
    
} large_cnn_model_t;

// ==================== 大型CNN架构创建函数 ====================

// 创建ResNet模型
large_cnn_model_t* create_resnet(resnet_variant_t variant, 
                                 int num_classes, 
                                 bool pretrained);

// 创建VGG模型
large_cnn_model_t* create_vgg(vgg_variant_t variant, 
                             int num_classes, 
                             bool pretrained);

// 创建Inception模型
large_cnn_model_t* create_inception(inception_variant_t variant, 
                                   int num_classes, 
                                   bool pretrained);

// 创建EfficientNet模型
large_cnn_model_t* create_efficientnet(efficientnet_variant_t variant, 
                                      int num_classes, 
                                      bool pretrained);

// 创建DenseNet模型
large_cnn_model_t* create_densenet(densenet_variant_t variant, 
                                  int num_classes, 
                                  bool pretrained);

// 使用自定义配置创建大型CNN模型
large_cnn_model_t* create_large_cnn_model(large_cnn_config_t* config);

// ==================== 模型操作函数 ====================

// 前向传播
Tensor* large_cnn_forward(large_cnn_model_t* model, Tensor* input);

// 获取模型参数
Tensor** large_cnn_get_parameters(large_cnn_model_t* model, int* num_params);

// 获取模型状态字典
void* large_cnn_get_state_dict(large_cnn_model_t* model);

// 加载预训练权重
int large_cnn_load_pretrained(large_cnn_model_t* model, const char* weights_path);

// 保存模型权重
int large_cnn_save_weights(large_cnn_model_t* model, const char* save_path);

// 模型评估
float large_cnn_evaluate(large_cnn_model_t* model, 
                        Tensor* test_input, 
                        Tensor* test_target);

// 模型信息统计
void large_cnn_print_summary(large_cnn_model_t* model);

// 计算FLOPs
size_t large_cnn_calculate_flops(large_cnn_model_t* model, int input_height, int input_width);

// 计算内存使用量
size_t large_cnn_calculate_memory_usage(large_cnn_model_t* model);

// ==================== 性能优化函数 ====================

// 启用梯度检查点
int large_cnn_enable_gradient_checkpointing(large_cnn_model_t* model);

// 启用混合精度训练
int large_cnn_enable_mixed_precision(large_cnn_model_t* model);

// 模型剪枝
int large_cnn_prune_model(large_cnn_model_t* model, float pruning_rate);

// 模型量化
int large_cnn_quantize_model(large_cnn_model_t* model, int quantization_bits);

// ==================== 工具函数 ====================

// 释放模型内存
void large_cnn_free(large_cnn_model_t* model);

// 模型克隆
large_cnn_model_t* large_cnn_clone(large_cnn_model_t* model);

// 模型比较
int large_cnn_compare(large_cnn_model_t* model1, large_cnn_model_t* model2);

// 获取架构名称
const char* large_cnn_get_arch_name(large_cnn_model_t* model);

#endif // LARGE_CNN_ARCHITECTURES_H