#ifndef MODEL_EXPLAINABILITY_H
#define MODEL_EXPLAINABILITY_H

#include "nn_module.h"
#include "tensor.h"
#include <stdbool.h>

// ===========================================
// 解释方法类型定义
// ===========================================

typedef enum {
    EXPLAIN_NONE = 0,
    EXPLAIN_SHAP = 1,                  // SHAP值解释
    EXPLAIN_LIME = 2,                   // LIME局部解释
    EXPLAIN_GRADIENT = 3,               // 梯度解释
    EXPLAIN_ATTENTION = 4,              // 注意力解释
    EXPLAIN_SALIENCY = 5,               // 显著性图
    EXPLAIN_INTEGRATED_GRADIENTS = 6,   // 积分梯度
    EXPLAIN_DEEP_LIFT = 7,              // DeepLIFT
    EXPLAIN_COUNTERFACTUAL = 8          // 反事实解释
} explainability_method_t;

// ===========================================
// SHAP配置结构体
// ===========================================

typedef struct {
    int num_samples;                    // 采样数量
    int max_features;                    // 最大特征数
    bool use_kernel_shap;               // 使用核SHAP
    bool use_tree_shap;                 // 使用树SHAP
    float background_data_ratio;        // 背景数据比例
} shap_config_t;

// ===========================================
// LIME配置结构体
// ===========================================

typedef struct {
    int num_samples;                    // 采样数量
    int num_features;                   // 特征数量
    float kernel_width;                 // 核宽度
    bool use_superpixels;              // 使用超像素
    int superpixel_segments;            // 超像素段数
} lime_config_t;

// ===========================================
// 梯度解释配置结构体
// ===========================================

typedef struct {
    bool use_guided_backprop;          // 使用引导反向传播
    bool use_deconvnet;                // 使用反卷积网络
    bool use_smoothgrad;               // 使用平滑梯度
    int smoothgrad_samples;            // 平滑梯度采样数
    float smoothgrad_noise;            // 平滑梯度噪声
} gradient_config_t;

// ===========================================
// 注意力解释配置结构体
// ===========================================

typedef struct {
    bool use_multi_head_attention;      // 使用多头注意力
    int attention_heads;                // 注意力头数
    bool normalize_attention;          // 归一化注意力
    float attention_threshold;          // 注意力阈值
} attention_config_t;

// ===========================================
// 解释结果结构体
// ===========================================

typedef struct {
    float* feature_importance;          // 特征重要性分数
    int num_features;                   // 特征数量
    float confidence_score;             // 置信度分数
    float explanation_fidelity;         // 解释保真度
    char** feature_names;               // 特征名称
    float* baseline_values;             // 基线值
} explanation_result_t;

// ===========================================
// 公平性评估配置结构体
// ===========================================

typedef struct {
    char** sensitive_features;          // 敏感特征
    int num_sensitive_features;         // 敏感特征数量
    float fairness_threshold;           // 公平性阈值
    bool check_disparate_impact;       // 检查差异影响
    bool check_equal_opportunity;       // 检查平等机会
    bool check_demographic_parity;      // 检查人口统计均等
} fairness_config_t;

// ===========================================
// 公平性评估结果结构体
// ===========================================

typedef struct {
    float disparate_impact_ratio;       // 差异影响比
    float equal_opportunity_difference; // 平等机会差异
    float demographic_parity_difference; // 人口统计均等差异
    float* group_accuracy;              // 组准确率
    int num_groups;                     // 组数量
    bool is_fair;                       // 是否公平
} fairness_result_t;

// ===========================================
// 模型解释管理器结构体
// ===========================================

typedef struct model_explainability_manager_s {
    explainability_method_t method;
    shap_config_t shap_config;
    lime_config_t lime_config;
    gradient_config_t gradient_config;
    attention_config_t attention_config;
    fairness_config_t fairness_config;
    
    // 内部状态
    bool is_initialized;
    int explanation_id;
    
    // 回调函数
    void (*progress_callback)(int progress, const char* message);
    void (*explanation_ready_callback)(const explanation_result_t* result);
    void (*fairness_ready_callback)(const fairness_result_t* result);
} model_explainability_manager_t;

// ===========================================
// API 函数声明
// ===========================================

// 解释管理器创建和销毁
model_explainability_manager_t* create_model_explainability_manager(void);
void destroy_model_explainability_manager(model_explainability_manager_t* manager);

// 解释方法配置
int set_explanation_method(model_explainability_manager_t* manager, explainability_method_t method);
int configure_shap(model_explainability_manager_t* manager, const shap_config_t* config);
int configure_lime(model_explainability_manager_t* manager, const lime_config_t* config);
int configure_gradient_explanation(model_explainability_manager_t* manager, const gradient_config_t* config);
int configure_attention_explanation(model_explainability_manager_t* manager, const attention_config_t* config);

// 模型解释执行
explanation_result_t* explain_model_prediction(model_explainability_manager_t* manager, 
                                             const nn_module_t* model, 
                                             const tensor_t* input, 
                                             int target_class);

int explain_model_batch(model_explainability_manager_t* manager, 
                       const nn_module_t* model, 
                       const tensor_t* inputs, 
                       int num_samples, 
                       int target_class,
                       explanation_result_t** results);

// 公平性评估
fairness_result_t* evaluate_model_fairness(model_explainability_manager_t* manager,
                                         const nn_module_t* model,
                                         const tensor_t* features,
                                         const tensor_t* labels,
                                         const tensor_t* sensitive_attributes);

// 解释结果管理
void free_explanation_result(explanation_result_t* result);
void free_fairness_result(fairness_result_t* result);

// 可视化支持
int export_explanation_visualization(const explanation_result_t* result, const char* filename);
int export_fairness_report(const fairness_result_t* result, const char* filename);

// 高级解释功能
int compute_feature_importance_global(model_explainability_manager_t* manager,
                                     const nn_module_t* model,
                                     const tensor_t* training_data,
                                     explanation_result_t** results);

int compute_model_decision_boundary(model_explainability_manager_t* manager,
                                   const nn_module_t* model,
                                   const tensor_t* data,
                                   const tensor_t* labels,
                                   float** boundary_data);

// 工具函数
float calculate_prediction_confidence(const nn_module_t* model, const tensor_t* input);
float calculate_explanation_fidelity(const explanation_result_t* explanation,
                                   const nn_module_t* model,
                                   const tensor_t* input);

// 回调函数设置
void set_explanation_progress_callback(model_explainability_manager_t* manager,
                                      void (*callback)(int progress, const char* message));

void set_explanation_ready_callback(model_explainability_manager_t* manager,
                                   void (*callback)(const explanation_result_t* result));

void set_fairness_ready_callback(model_explainability_manager_t* manager,
                                void (*callback)(const fairness_result_t* result));

#endif // MODEL_EXPLAINABILITY_H