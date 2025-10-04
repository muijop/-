#ifndef NEURAL_ARCHITECTURE_SEARCH_H
#define NEURAL_ARCHITECTURE_SEARCH_H

#include "nn_module.h"
#include "model_builder.h"
#include <stdbool.h>

// ===========================================
// NAS 搜索空间定义
// ===========================================

typedef enum {
    NAS_SEARCH_REINFORCEMENT = 0,      // 强化学习搜索
    NAS_SEARCH_EVOLUTIONARY = 1,       // 进化算法搜索
    NAS_SEARCH_BAYESIAN = 2,          // 贝叶斯优化搜索
    NAS_SEARCH_GRADIENT = 3,           // 梯度优化搜索
    NAS_SEARCH_RANDOM = 4              // 随机搜索
} nas_search_method_t;

// 网络层类型搜索空间
typedef enum {
    LAYER_SEARCH_DENSE = 0,           // 全连接层
    LAYER_SEARCH_CONV1D = 1,          // 1D卷积层
    LAYER_SEARCH_CONV2D = 2,          // 2D卷积层
    LAYER_SEARCH_LSTM = 3,            // LSTM层
    LAYER_SEARCH_GRU = 4,             // GRU层
    LAYER_SEARCH_ATTENTION = 5,       // 注意力层
    LAYER_SEARCH_TRANSFORMER = 6      // Transformer层
} layer_search_type_t;

// 激活函数搜索空间
typedef enum {
    ACTIVATION_SEARCH_RELU = 0,       // ReLU
    ACTIVATION_SEARCH_SIGMOID = 1,    // Sigmoid
    ACTIVATION_SEARCH_TANH = 2,       // Tanh
    ACTIVATION_SEARCH_LEAKY_RELU = 3, // Leaky ReLU
    ACTIVATION_SEARCH_ELU = 4,       // ELU
    ACTIVATION_SEARCH_SELU = 5,       // SELU
    ACTIVATION_SEARCH_SWISH = 6       // Swish
} activation_search_type_t;

// ===========================================
// NAS 配置结构体
// ===========================================

typedef struct {
    nas_search_method_t search_method;        // 搜索方法
    int max_layers;                           // 最大层数
    int min_layers;                           // 最小层数
    int max_units_per_layer;                  // 每层最大单元数
    int min_units_per_layer;                  // 每层最小单元数
    int population_size;                       // 种群大小（进化算法）
    int num_generations;                       // 代数（进化算法）
    int num_episodes;                          // 回合数（强化学习）
    float mutation_rate;                       // 变异率
    float crossover_rate;                      // 交叉率
    float learning_rate;                       // 学习率（强化学习）
    bool use_skip_connections;                // 使用跳跃连接
    bool use_batch_norm;                       // 使用批归一化
    bool use_dropout;                          // 使用Dropout
    float dropout_rate;                        // Dropout率
} nas_config_t;

// ===========================================
// 网络架构编码
// ===========================================

typedef struct {
    layer_search_type_t layer_type;           // 层类型
    int units;                                // 单元数
    activation_search_type_t activation;      // 激活函数
    bool use_batch_norm;                      // 使用批归一化
    bool use_dropout;                         // 使用Dropout
    float dropout_rate;                       // Dropout率
} layer_encoding_t;

typedef struct {
    layer_encoding_t* layers;                 // 层编码数组
    int num_layers;                           // 层数
    float fitness;                            // 适应度
    float complexity;                         // 复杂度
    float accuracy;                           // 准确率
    float inference_time;                     // 推理时间
} architecture_encoding_t;

// ===========================================
// NAS 搜索结果
// ===========================================

typedef struct {
    architecture_encoding_t best_architecture; // 最佳架构
    float best_fitness;                        // 最佳适应度
    int total_evaluations;                    // 总评估次数
    float search_time;                         // 搜索时间（秒）
    int num_architectures_tested;              // 测试的架构数量
    float average_accuracy;                    // 平均准确率
    float average_complexity;                  // 平均复杂度
} nas_search_result_t;

// ===========================================
// NAS 管理器结构体
// ===========================================

typedef struct neural_architecture_search_manager_s {
    nas_config_t config;                      // 配置
    architecture_encoding_t* population;       // 种群（进化算法）
    int population_count;                      // 种群数量
    nas_search_result_t search_result;         // 搜索结果
    bool is_searching;                         // 是否正在搜索
    
    // 回调函数
    void (*progress_callback)(int progress, const char* message);
    void (*architecture_evaluated_callback)(const architecture_encoding_t* arch, float fitness);
    void (*search_completed_callback)(const nas_search_result_t* result);
} neural_architecture_search_manager_t;

// ===========================================
// API 函数声明
// ===========================================

// NAS 管理器创建和销毁
neural_architecture_search_manager_t* create_nas_manager(void);
void destroy_nas_manager(neural_architecture_search_manager_t* manager);

// 配置设置
int configure_nas_search(neural_architecture_search_manager_t* manager, const nas_config_t* config);
int set_nas_search_method(neural_architecture_search_manager_t* manager, nas_search_method_t method);

// 搜索执行
nas_search_result_t perform_architecture_search(neural_architecture_search_manager_t* manager, 
                                               const training_data_t* train_data, 
                                               const training_data_t* val_data);
int start_nas_search_async(neural_architecture_search_manager_t* manager,
                          const training_data_t* train_data,
                          const training_data_t* val_data);
int stop_nas_search(neural_architecture_search_manager_t* manager);

// 架构评估和构建
float evaluate_architecture(const architecture_encoding_t* architecture,
                           const training_data_t* train_data,
                           const training_data_t* val_data);
nn_module_t* build_architecture(const architecture_encoding_t* architecture);
architecture_encoding_t decode_architecture(const nn_module_t* model);

// 搜索空间操作
architecture_encoding_t generate_random_architecture(const nas_config_t* config);
architecture_encoding_t mutate_architecture(const architecture_encoding_t* parent, 
                                           const nas_config_t* config);
architecture_encoding_t crossover_architectures(const architecture_encoding_t* parent1,
                                               const architecture_encoding_t* parent2,
                                               const nas_config_t* config);

// 适应度计算
float calculate_architecture_fitness(const architecture_encoding_t* architecture,
                                     float accuracy, float complexity, float inference_time);
float calculate_architecture_complexity(const architecture_encoding_t* architecture);

// 结果分析和管理
nas_search_result_t get_nas_search_result(const neural_architecture_search_manager_t* manager);
int export_best_architecture(const neural_architecture_search_manager_t* manager, 
                            const char* filename);
int visualize_search_progress(const neural_architecture_search_manager_t* manager,
                            const char* filename);

// 高级搜索算法
int perform_reinforcement_nas(neural_architecture_search_manager_t* manager,
                             const training_data_t* train_data,
                             const training_data_t* val_data);
int perform_evolutionary_nas(neural_architecture_search_manager_t* manager,
                            const training_data_t* train_data,
                            const training_data_t* val_data);
int perform_bayesian_nas(neural_architecture_search_manager_t* manager,
                       const training_data_t* train_data,
                       const training_data_t* val_data);
int perform_gradient_nas(neural_architecture_search_manager_t* manager,
                        const training_data_t* train_data,
                        const training_data_t* val_data);

// 回调函数设置
void set_nas_progress_callback(neural_architecture_search_manager_t* manager,
                              void (*callback)(int progress, const char* message));
void set_architecture_evaluated_callback(neural_architecture_search_manager_t* manager,
                                        void (*callback)(const architecture_encoding_t* arch, float fitness));
void set_search_completed_callback(neural_architecture_search_manager_t* manager,
                                 void (*callback)(const nas_search_result_t* result));

// 工具函数
nas_config_t create_default_nas_config(void);
architecture_encoding_t create_simple_architecture(int num_layers, int units_per_layer);
void print_architecture(const architecture_encoding_t* architecture);

#endif // NEURAL_ARCHITECTURE_SEARCH_H