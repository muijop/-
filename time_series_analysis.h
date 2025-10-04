#ifndef TIME_SERIES_ANALYSIS_H
#define TIME_SERIES_ANALYSIS_H

#include <stddef.h>
#include <stdbool.h>

// 时间序列分析方法枚举
typedef enum {
    TS_LSTM = 0,           // LSTM网络
    TS_GRU = 1,            // GRU网络
    TS_TCN = 2,            // 时序卷积网络
    TS_TRANSFORMER = 3,    // Transformer
    TS_ARIMA = 4,          // ARIMA模型
    TS_SARIMA = 5,         // 季节性ARIMA
    TS_ES = 6,             // 指数平滑
    TS_PROPHET = 7,        // Prophet模型
    TS_NBEATS = 8,         // N-BEATS
    TS_DEEPAR = 9          // DeepAR
} time_series_method_t;

// 时间序列数据结构
typedef struct {
    size_t length;              // 序列长度
    size_t num_features;        // 特征数量
    size_t num_targets;         // 目标数量
    
    float** data;               // 时间序列数据 [length x num_features]
    float** targets;            // 目标数据 [length x num_targets]
    
    // 时间信息
    long* timestamps;           // 时间戳
    size_t frequency;           // 频率（如：24小时，7天等）
    char* time_unit;            // 时间单位
    
    // 序列属性
    bool is_multivariate;       // 是否多变量
    bool has_missing_values;    // 是否有缺失值
    bool is_stationary;         // 是否平稳
    
    // 统计信息
    float* means;               // 均值 [num_features]
    float* stds;               // 标准差 [num_features]
    float* min_vals;           // 最小值 [num_features]
    float* max_vals;           // 最大值 [num_features]
    
    char* series_name;          // 序列名称
} time_series_data_t;

// LSTM配置结构体
typedef struct {
    size_t hidden_size;         // 隐藏层大小
    int num_layers;             // 层数
    float dropout_rate;         // Dropout率
    bool bidirectional;         // 是否双向
    char activation[16];       // 激活函数
} lstm_config_t;

// GRU配置结构体
typedef struct {
    size_t hidden_size;         // 隐藏层大小
    int num_layers;             // 层数
    float dropout_rate;         // Dropout率
    bool bidirectional;         // 是否双向
} gru_config_t;

// TCN配置结构体
typedef struct {
    size_t num_channels[5];     // 通道数
    int kernel_size;            // 卷积核大小
    float dropout_rate;         // Dropout率
    int num_levels;             // 层级数
    bool causal;                // 是否因果卷积
} tcn_config_t;

// Transformer配置结构体
typedef struct {
    size_t d_model;             // 模型维度
    int num_heads;              // 注意力头数
    int num_layers;             // 编码器层数
    size_t d_ff;               // 前馈网络维度
    float dropout_rate;         // Dropout率
} transformer_config_t;

// ARIMA配置结构体
typedef struct {
    int p;                      // AR阶数
    int d;                      // 差分阶数
    int q;                      // MA阶数
    int seasonal_period;        // 季节性周期
    bool use_mle;              // 是否使用最大似然估计
} arima_config_t;

// 时间序列配置联合体
typedef union {
    lstm_config_t lstm;
    gru_config_t gru;
    tcn_config_t tcn;
    transformer_config_t transformer;
    arima_config_t arima;
} time_series_specific_config_t;

// 时间序列分析配置结构体
typedef struct {
    time_series_method_t method;        // 分析方法
    size_t input_length;               // 输入序列长度
    size_t output_length;              // 输出序列长度
    size_t input_dim;                  // 输入维度
    size_t output_dim;                 // 输出维度
    
    // 通用训练配置
    float learning_rate;               // 学习率
    int max_epochs;                    // 最大训练轮数
    int batch_size;                    // 批次大小
    bool use_gpu;                      // 是否使用GPU
    int seed;                          // 随机种子
    
    time_series_specific_config_t specific; // 特定方法配置
    
    // 预处理配置
    bool normalize;                    // 是否归一化
    bool detrend;                     // 是否去趋势
    bool deseasonalize;               // 是否去季节性
    
    // 训练控制
    int patience;                      // 早停耐心值
    float min_delta;                  // 最小改进阈值
    bool early_stopping;              // 是否早停
    
    // 资源限制
    int max_memory_mb;                // 最大内存使用(MB)
    int max_time_seconds;             // 最大训练时间(秒)
} time_series_config_t;

// 时间序列模型结构体
typedef struct {
    time_series_method_t method;        // 分析方法
    time_series_config_t config;        // 配置
    void* model_impl;                   // 模型实现指针
    bool is_trained;                    // 是否已训练
    bool is_initialized;                // 是否已初始化
    
    // 训练状态
    int current_epoch;                  // 当前训练轮数
    float best_loss;                    // 最佳损失值
    float current_loss;                 // 当前损失值
    
    // 性能指标
    float mae;                         // 平均绝对误差
    float mse;                         // 均方误差
    float rmse;                        // 均方根误差
    float mape;                        // 平均绝对百分比误差
    float smape;                       // 对称平均绝对百分比误差
    
    // 性能统计
    double training_time;               // 训练时间(秒)
    size_t memory_usage;               // 内存使用量(bytes)
} time_series_model_t;

// 时间序列训练结果结构体
typedef struct {
    float final_loss;                   // 最终损失值
    int total_epochs;                   // 总训练轮数
    double total_time;                  // 总训练时间(秒)
    bool success;                       // 是否成功
    char* status_message;               // 状态消息
    
    // 性能指标
    float final_mae;
    float final_mse;
    float final_rmse;
    float final_mape;
    float final_smape;
    
    // 训练历史
    float* loss_history;                // 损失历史
    float* mae_history;                 // MAE历史
    int history_size;                   // 历史记录大小
} time_series_training_result_t;

// 时间序列预测结果结构体
typedef struct {
    float* predictions;                 // 预测值 [output_length x output_dim]
    float* confidence_intervals;        // 置信区间 [output_length x 2 x output_dim]
    float* residuals;                   // 残差 [input_length x output_dim]
    
    size_t prediction_length;          // 预测长度
    size_t output_dim;                  // 输出维度
    
    // 预测质量指标
    float prediction_mae;
    float prediction_mse;
    float prediction_rmse;
    float prediction_mape;
    float prediction_smape;
} time_series_prediction_result_t;

// ===========================================
// 时间序列模型管理API
// ===========================================

// 创建时间序列模型
time_series_model_t* create_time_series_model(time_series_method_t method, 
                                              const time_series_config_t* config);

// 销毁时间序列模型
void destroy_time_series_model(time_series_model_t* model);

// 初始化时间序列模型
int initialize_time_series_model(time_series_model_t* model, 
                               const time_series_data_t* data);

// ===========================================
// 训练和评估API
// ===========================================

// 训练时间序列模型
time_series_training_result_t* train_time_series_model(time_series_model_t* model, 
                                                      const time_series_data_t* train_data,
                                                      const time_series_data_t* val_data);

// 评估时间序列模型
float evaluate_time_series_model(const time_series_model_t* model, 
                               const time_series_data_t* test_data);

// 预测
time_series_prediction_result_t* predict_with_time_series_model(
    const time_series_model_t* model, const time_series_data_t* input_data);

// ===========================================
// 时间序列数据处理API
// ===========================================

// 创建时间序列数据
time_series_data_t* create_time_series_data(size_t length, size_t num_features, 
                                           size_t num_targets);

// 销毁时间序列数据
void destroy_time_series_data(time_series_data_t* data);

// 设置时间序列数据点
int set_time_series_point(time_series_data_t* data, size_t time_index, 
                         const float* features, const float* targets);

// 时间序列预处理
int preprocess_time_series_data(time_series_data_t* data, int preprocessing_type);

// 时间序列特征工程
int extract_time_series_features(const time_series_data_t* data, 
                               float** features, size_t* num_new_features);

// ===========================================
// 工具函数
// ===========================================

// 创建默认LSTM配置
time_series_config_t create_default_lstm_config(size_t input_length, 
                                              size_t output_length,
                                              size_t input_dim, size_t output_dim);

// 创建默认GRU配置
time_series_config_t create_default_gru_config(size_t input_length, 
                                             size_t output_length,
                                             size_t input_dim, size_t output_dim);

// 创建默认TCN配置
time_series_config_t create_default_tcn_config(size_t input_length, 
                                              size_t output_length,
                                              size_t input_dim, size_t output_dim);

// 创建默认Transformer配置
time_series_config_t create_default_transformer_config(size_t input_length, 
                                                      size_t output_length,
                                                      size_t input_dim, size_t output_dim);

// 创建默认ARIMA配置
time_series_config_t create_default_arima_config(int p, int d, int q);

// 创建示例时间序列数据
time_series_data_t* create_example_time_series_data(int data_type);

// 获取模型信息
void print_time_series_model_info(const time_series_model_t* model);

// 获取训练统计
void get_time_series_training_stats(const time_series_model_t* model, 
                                  float* loss, float* mae, int* epoch);

// ===========================================
// 高级时间序列分析API
// ===========================================

// 时间序列分解
int decompose_time_series(const time_series_data_t* data, 
                         float** trend, float** seasonal, float** residual);

// 异常检测
int detect_time_series_anomalies(const time_series_data_t* data, 
                               float* anomaly_scores, float threshold);

// 时间序列聚类
int cluster_time_series(const time_series_data_t** series_array, 
                       size_t num_series, int* cluster_labels, int num_clusters);

// 时间序列相似性搜索
int find_similar_time_series(const time_series_data_t* query, 
                           const time_series_data_t** database,
                           size_t database_size, size_t* similar_indices,
                           float* similarity_scores, size_t top_k);

// 时间序列生成
int generate_time_series(time_series_model_t* model, size_t length,
                        time_series_data_t** generated_data);

#endif // TIME_SERIES_ANALYSIS_H