#ifndef LOSS_FUNCTIONS_AUTOGRAD_H
#define LOSS_FUNCTIONS_AUTOGRAD_H

#include "nn.h"
#include "tensor.h"
#include "tensor_autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 损失函数类型定义 ====================

typedef enum {
    LOSS_MSE = 0,           // 均方误差
    LOSS_MAE,               // 平均绝对误差
    LOSS_BCE,               // 二元交叉熵
    LOSS_CE,                // 交叉熵
    LOSS_NLL,               // 负对数似然
    LOSS_HUBER,             // Huber损失
    LOSS_SMOOTH_L1,         // Smooth L1损失
    LOSS_KL_DIVERGENCE,     // KL散度
    LOSS_COSINE_SIMILARITY,  // 余弦相似度
    LOSS_TRIPLET,           // Triplet损失
    LOSS_CONTRASTIVE,       // Contrastive损失
    LOSS_FOCAL,             // Focal损失
    LOSS_DICE,              // Dice损失
    LOSS_IOU,               // IoU损失
    LOSS_WASSERSTEIN,       // Wasserstein距离
    LOSS_ADVERSARIAL,       // 对抗损失
    LOSS_PERCEPTUAL,        // 感知损失
    LOSS_STYLE,             // 风格损失
    LOSS_CONTENT,           // 内容损失
    LOSS_GAN,               // GAN损失
    LOSS_VAE,               // VAE损失
    LOSS_CUSTOM             // 自定义损失
} AutogradLossFunctionType;

// ==================== 损失函数配置结构 ====================

typedef struct AutogradLossFunctionConfig {
    AutogradLossFunctionType type;    // 损失函数类型
    float reduction;                   // 缩减方式 (0: none, 1: mean, 2: sum)
    float weight;                      // 权重
    float pos_weight;                 // 正样本权重
    float ignore_index;               // 忽略索引
    float label_smoothing;            // 标签平滑
    float alpha;                      // Alpha参数
    float gamma;                      // Gamma参数
    float margin;                     // 边界参数
    float temperature;                // 温度参数
    float eps;                        // 数值稳定性参数
    int from_logits;                  // 是否来自logits
    int symmetric;                    // 是否对称
    int size_average;                 // 是否平均大小
    int reduce;                       // 是否缩减
} AutogradLossFunctionConfig;

// ==================== 损失函数结构 ====================

typedef struct AutogradLossFunction {
    AutogradLossFunctionType type;    // 损失函数类型
    AutogradLossFunctionConfig config; // 损失函数配置
    char* name;                       // 损失函数名称
    void* user_data;                  // 用户数据
    int initialized;                  // 是否已初始化
} AutogradLossFunction;

// ==================== 损失函数创建和销毁 ====================

// 创建损失函数
AutogradLossFunction* autograd_loss_function_create(AutogradLossFunctionType type);

// 销毁损失函数
void autograd_loss_function_destroy(AutogradLossFunction* loss);

// ==================== 损失函数配置管理 ====================

// 设置损失函数配置
void autograd_loss_function_set_config(AutogradLossFunction* loss, const AutogradLossFunctionConfig* config);

// 获取损失函数配置
AutogradLossFunctionConfig autograd_loss_function_get_config(const AutogradLossFunction* loss);

// ==================== 损失计算 ====================

// 计算损失
AutogradTensor* autograd_loss_function_forward(AutogradLossFunction* loss, AutogradTensor* input, AutogradTensor* target);

// 计算损失（反向传播）
AutogradTensor* autograd_loss_function_backward(AutogradLossFunction* loss, AutogradTensor* input, AutogradTensor* target);

// ==================== 特定损失函数 ====================

// 均方误差损失
AutogradTensor* autograd_loss_mse(AutogradTensor* input, AutogradTensor* target, float reduction);

// 平均绝对误差损失
AutogradTensor* autograd_loss_mae(AutogradTensor* input, AutogradTensor* target, float reduction);

// 二元交叉熵损失
AutogradTensor* autograd_loss_bce(AutogradTensor* input, AutogradTensor* target, float reduction, float weight, float pos_weight);

// 交叉熵损失
AutogradTensor* autograd_loss_ce(AutogradTensor* input, AutogradTensor* target, float reduction, float weight, float ignore_index, float label_smoothing);

// 负对数似然损失
AutogradTensor* autograd_loss_nll(AutogradTensor* input, AutogradTensor* target, float reduction, float weight, float ignore_index);

// Huber损失
AutogradTensor* autograd_loss_huber(AutogradTensor* input, AutogradTensor* target, float reduction, float delta);

// Smooth L1损失
AutogradTensor* autograd_loss_smooth_l1(AutogradTensor* input, AutogradTensor* target, float reduction, float beta);

// KL散度损失
AutogradTensor* autograd_loss_kl_divergence(AutogradTensor* input, AutogradTensor* target, float reduction, int log_target);

// 余弦相似度损失
AutogradTensor* autograd_loss_cosine_similarity(AutogradTensor* input, AutogradTensor* target, float reduction, float margin);

// Triplet损失
AutogradTensor* autograd_loss_triplet(AutogradTensor* anchor, AutogradTensor* positive, AutogradTensor* negative, float margin, float p, float eps, int swap);

// Contrastive损失
AutogradTensor* autograd_loss_contrastive(AutogradTensor* input1, AutogradTensor* input2, AutogradTensor* target, float margin, float reduction);

// Focal损失
AutogradTensor* autograd_loss_focal(AutogradTensor* input, AutogradTensor* target, float alpha, float gamma, float reduction);

// Dice损失
AutogradTensor* autograd_loss_dice(AutogradTensor* input, AutogradTensor* target, float smooth, float reduction);

// IoU损失
AutogradTensor* autograd_loss_iou(AutogradTensor* input, AutogradTensor* target, float smooth, float reduction);

// Wasserstein距离损失
AutogradTensor* autograd_loss_wasserstein(AutogradTensor* input, AutogradTensor* target, float reduction);

// ==================== 高级损失函数 ====================

// 对抗损失
AutogradTensor* autograd_loss_adversarial(AutogradTensor* real, AutogradTensor* fake, int discriminator_type);

// 感知损失
AutogradTensor* autograd_loss_perceptual(AutogradTensor* input, AutogradTensor* target, void* feature_extractor, int layer_idx);

// 风格损失
AutogradTensor* autograd_loss_style(AutogradTensor* input, AutogradTensor* target, void* feature_extractor, int layer_idx);

// 内容损失
AutogradTensor* autograd_loss_content(AutogradTensor* input, AutogradTensor* target, void* feature_extractor, int layer_idx);

// GAN损失
AutogradTensor* autograd_loss_gan(AutogradTensor* discriminator_output, AutogradTensor* target, int gan_type);

// VAE损失
AutogradTensor* autograd_loss_vae(AutogradTensor* reconstructed, AutogradTensor* original, AutogradTensor* mu, AutogradTensor* logvar, float kl_weight);

// ==================== 损失函数工厂函数 ====================

// 创建MSE损失函数
AutogradLossFunction* autograd_loss_mse_create(float reduction);

// 创建MAE损失函数
AutogradLossFunction* autograd_loss_mae_create(float reduction);

// 创建BCE损失函数
AutogradLossFunction* autograd_loss_bce_create(float reduction, float weight, float pos_weight);

// 创建CE损失函数
AutogradLossFunction* autograd_loss_ce_create(float reduction, float weight, float ignore_index, float label_smoothing);

// 创建NLL损失函数
AutogradLossFunction* autograd_loss_nll_create(float reduction, float weight, float ignore_index);

// 创建Huber损失函数
AutogradLossFunction* autograd_loss_huber_create(float reduction, float delta);

// 创建Smooth L1损失函数
AutogradLossFunction* autograd_loss_smooth_l1_create(float reduction, float beta);

// 创建KL散度损失函数
AutogradLossFunction* autograd_loss_kl_divergence_create(float reduction, int log_target);

// 创建余弦相似度损失函数
AutogradLossFunction* autograd_loss_cosine_similarity_create(float reduction, float margin);

// 创建Triplet损失函数
AutogradLossFunction* autograd_loss_triplet_create(float margin, float p, float eps, int swap);

// 创建Contrastive损失函数
AutogradLossFunction* autograd_loss_contrastive_create(float margin, float reduction);

// 创建Focal损失函数
AutogradLossFunction* autograd_loss_focal_create(float alpha, float gamma, float reduction);

// 创建Dice损失函数
AutogradLossFunction* autograd_loss_dice_create(float smooth, float reduction);

// 创建IoU损失函数
AutogradLossFunction* autograd_loss_iou_create(float smooth, float reduction);

// 创建Wasserstein距离损失函数
AutogradLossFunction* autograd_loss_wasserstein_create(float reduction);

// ==================== 损失函数工具函数 ====================

// 损失函数初始化
void autograd_loss_function_initialize(AutogradLossFunction* loss);

// 损失函数验证
int autograd_loss_function_validate(const AutogradLossFunction* loss);

// 损失函数重置
void autograd_loss_function_reset(AutogradLossFunction* loss);

// 损失函数克隆
AutogradLossFunction* autograd_loss_function_clone(const AutogradLossFunction* loss);

// 损失函数比较
int autograd_loss_function_compare(const AutogradLossFunction* loss1, const AutogradLossFunction* loss2);

// 损失函数序列化
void autograd_loss_function_serialize(const AutogradLossFunction* loss, const char* filename);

// 损失函数反序列化
AutogradLossFunction* autograd_loss_function_deserialize(const char* filename);

// ==================== 损失函数统计 ====================

// 损失函数统计信息
typedef struct AutogradLossFunctionStats {
    int total_calls;                  // 总调用次数
    float total_loss;                 // 总损失
    float average_loss;               // 平均损失
    float min_loss;                   // 最小损失
    float max_loss;                   // 最大损失
    float current_loss;               // 当前损失
    int num_batches;                  // 批次数量
    float batch_average_loss;         // 批次平均损失
} AutogradLossFunctionStats;

// 获取损失函数统计
AutogradLossFunctionStats autograd_loss_function_get_stats(const AutogradLossFunction* loss);

// 重置损失函数统计
void autograd_loss_function_reset_stats(AutogradLossFunction* loss);

// ==================== 损失函数回调 ====================

// 损失函数回调函数类型
typedef void (*AutogradLossFunctionCallback)(AutogradLossFunction* loss, int call_idx, float loss_value, void* user_data);

// 设置损失函数回调
void autograd_loss_function_set_callback(AutogradLossFunction* loss, AutogradLossFunctionCallback callback, void* user_data);

// ==================== 损失函数调试 ====================

// 打印损失函数信息
void autograd_loss_function_print_info(const AutogradLossFunction* loss);

// 打印损失函数统计
void autograd_loss_function_print_stats(const AutogradLossFunction* loss);

// 损失函数调试模式
void autograd_loss_function_set_debug_mode(AutogradLossFunction* loss, int debug);

// ==================== 损失函数配置预设 ====================

// 获取默认损失函数配置
AutogradLossFunctionConfig autograd_loss_function_config_default(AutogradLossFunctionType type);

// 获取MSE默认配置
AutogradLossFunctionConfig autograd_loss_function_config_mse(void);

// 获取BCE默认配置
AutogradLossFunctionConfig autograd_loss_function_config_bce(void);

// 获取CE默认配置
AutogradLossFunctionConfig autograd_loss_function_config_ce(void);

// ==================== 损失函数组合 ====================

// 损失函数组合结构
typedef struct AutogradLossFunctionCombination {
    AutogradLossFunction** losses;    // 损失函数数组
    float* weights;                   // 权重数组
    int num_losses;                   // 损失函数数量
    char* name;                       // 组合名称
} AutogradLossFunctionCombination;

// 创建损失函数组合
AutogradLossFunctionCombination* autograd_loss_combination_create(AutogradLossFunction** losses, float* weights, int num_losses);

// 销毁损失函数组合
void autograd_loss_combination_destroy(AutogradLossFunctionCombination* combination);

// 计算组合损失
AutogradTensor* autograd_loss_combination_forward(AutogradLossFunctionCombination* combination, AutogradTensor* input, AutogradTensor* target);

// 计算组合损失（反向传播）
AutogradTensor* autograd_loss_combination_backward(AutogradLossFunctionCombination* combination, AutogradTensor* input, AutogradTensor* target);

// ==================== 损失函数注册表 ====================

// 损失函数注册函数类型
typedef AutogradLossFunction* (*AutogradLossFunctionFactory)(const AutogradLossFunctionConfig* config);

// 注册损失函数
void autograd_loss_function_register(const char* name, AutogradLossFunctionFactory factory);

// 注销损失函数
void autograd_loss_function_unregister(const char* name);

// 获取已注册的损失函数
AutogradLossFunctionFactory autograd_loss_function_get_factory(const char* name);

// 获取所有已注册的损失函数名称
const char** autograd_loss_function_get_registered_names(int* num_names);

#ifdef __cplusplus
}
#endif

#endif // LOSS_FUNCTIONS_AUTOGRAD_H