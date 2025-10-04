#ifndef MODEL_ZOO_H
#define MODEL_ZOO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "tensor.h"
#include "nn_module.h"

#ifdef __cplusplus
extern "C" {
#endif

// 预训练模型类型枚举
typedef enum {
    MODEL_TYPE_VISION = 0,
    MODEL_TYPE_NLP = 1,
    MODEL_TYPE_MULTIMODAL = 2,
    MODEL_TYPE_SPEECH = 3,
    MODEL_TYPE_RECOMMENDATION = 4
} model_type_t;

// 模型架构枚举
typedef enum {
    ARCH_RESNET50 = 0,
    ARCH_VGG16 = 1,
    ARCH_BERT_BASE = 2,
    ARCH_GPT2 = 3,
    ARCH_VIT_BASE = 4,
    ARCH_EFFICIENTNET_B0 = 5,
    ARCH_MOBILENET_V2 = 6,
    ARCH_ALBERT_BASE = 7,
    ARCH_ROBERTA_BASE = 8,
    ARCH_DISTILBERT_BASE = 9
} model_architecture_t;

// 预训练数据集枚举
typedef enum {
    DATASET_IMAGENET = 0,
    DATASET_COCO = 1,
    DATASET_WIKITEXT = 2,
    DATASET_BOOKCORPUS = 3,
    DATASET_OPENWEBTEXT = 4,
    DATASET_COMMONVOICE = 5,
    DATASET_MOVIELENS = 6
} pretrained_dataset_t;

// 模型权重信息结构体
typedef struct {
    char* model_name;
    model_type_t model_type;
    model_architecture_t architecture;
    pretrained_dataset_t dataset;
    char* version;
    char* author;
    char* description;
    char* license;
    char* download_url;
    char* local_path;
    size_t file_size;
    int64_t num_parameters;
    float accuracy;
    bool is_downloaded;
    bool is_verified;
    char* sha256_hash;
} model_weight_info_t;

// 模型库管理器结构体
typedef struct {
    model_weight_info_t** models;
    int num_models;
    int max_models;
    char* cache_dir;
    char* config_file;
    bool enable_auto_download;
    bool enable_hash_verification;
} model_zoo_manager_t;

// 模型权重加载器结构体
typedef struct {
    char* weight_file_path;
    FILE* weight_file;
    uint32_t format_version;
    char* model_name;
    int64_t total_parameters;
    int64_t current_offset;
    bool is_binary_mode;
} weight_loader_t;

// ==================== 模型库管理接口 ====================

// 初始化模型库管理器
model_zoo_manager_t* model_zoo_create(const char* cache_dir, const char* config_file);

// 销毁模型库管理器
void model_zoo_destroy(model_zoo_manager_t* manager);

// 注册预训练模型
int model_zoo_register_model(model_zoo_manager_t* manager, 
                            const char* model_name,
                            model_type_t model_type,
                            model_architecture_t architecture,
                            pretrained_dataset_t dataset,
                            const char* download_url,
                            const char* description);

// 下载模型权重
int model_zoo_download_weights(model_zoo_manager_t* manager, 
                              const char* model_name,
                              bool force_download);

// 验证模型权重完整性
bool model_zoo_verify_weights(model_zoo_manager_t* manager, 
                              const char* model_name);

// 获取模型权重信息
model_weight_info_t* model_zoo_get_model_info(model_zoo_manager_t* manager, 
                                             const char* model_name);

// 列出所有可用模型
int model_zoo_list_models(model_zoo_manager_t* manager, 
                         model_type_t filter_type,
                         char*** model_names);

// ==================== 权重加载接口 ====================

// 创建权重加载器
weight_loader_t* weight_loader_create(const char* weight_file_path);

// 销毁权重加载器
void weight_loader_destroy(weight_loader_t* loader);

// 加载权重到神经网络模块
int weight_loader_load_to_module(weight_loader_t* loader, 
                                nn_module_t* module,
                                const char* layer_name);

// 加载权重到张量
int weight_loader_load_to_tensor(weight_loader_t* loader,
                                tensor_t* tensor,
                                const char* tensor_name);

// 获取权重文件信息
int weight_loader_get_info(weight_loader_t* loader,
                          char** model_name,
                          int64_t* total_parameters);

// ==================== 预训练模型加载接口 ====================

// 加载ResNet50预训练模型
nn_module_t* model_zoo_load_resnet50(model_zoo_manager_t* manager,
                                    bool pretrained);

// 加载BERT预训练模型
nn_module_t* model_zoo_load_bert_base(model_zoo_manager_t* manager,
                                     bool pretrained);

// 加载GPT-2预训练模型
nn_module_t* model_zoo_load_gpt2(model_zoo_manager_t* manager,
                                bool pretrained);

// 加载ViT预训练模型
nn_module_t* model_zoo_load_vit_base(model_zoo_manager_t* manager,
                                    bool pretrained);

// 加载MobileNetV2预训练模型
nn_module_t* model_zoo_load_mobilenet_v2(model_zoo_manager_t* manager,
                                        bool pretrained);

// ==================== 模型转换接口 ====================

// 转换为ONNX格式
int model_zoo_convert_to_onnx(nn_module_t* model,
                             const char* output_path,
                             const char* input_names[],
                             const char* output_names[],
                             int num_inputs,
                             int num_outputs);

// 转换为TensorRT格式
int model_zoo_convert_to_tensorrt(nn_module_t* model,
                                 const char* output_path,
                                 int max_batch_size,
                                 int workspace_size);

// ==================== 工具函数 ====================

// 计算文件SHA256哈希
int calculate_file_hash(const char* file_path, char* hash_buffer, size_t buffer_size);

// 下载文件
int download_file(const char* url, const char* output_path, 
                 void (*progress_callback)(float progress));

// 创建目录
int create_directory_recursive(const char* path);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ZOO_H