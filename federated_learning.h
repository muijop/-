#ifndef FEDERATED_LEARNING_H
#define FEDERATED_LEARNING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include "nn_module.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// 联邦学习聚合算法枚举
typedef enum {
    FED_AVG = 0,           // 联邦平均
    FED_PROX = 1,           // FedProx
    FED_ADAM = 2,           // FedAdam
    FED_YOGI = 3,           // FedYogi
    FED_SGD = 4,            // FedSGD
    FED_ENSEMBLE = 5,       // 联邦集成
    FED_DP = 6              // 差分隐私联邦学习
} federated_algorithm_t;

// 客户端状态枚举
typedef enum {
    CLIENT_IDLE = 0,        // 空闲
    CLIENT_TRAINING = 1,    // 训练中
    CLIENT_UPLOADING = 2,   // 上传中
    CLIENT_ERROR = 3        // 错误
} client_state_t;

// 联邦学习配置结构体
typedef struct {
    federated_algorithm_t algorithm;
    int num_clients;                // 客户端数量
    int rounds;                    // 训练轮数
    int epochs_per_round;          // 每轮本地训练轮数
    float learning_rate;           // 学习率
    float proximal_mu;             // FedProx参数
    float beta1;                   // Adam beta1
    float beta2;                   // Adam beta2
    float epsilon;                 // Adam epsilon
    bool enable_differential_privacy; // 启用差分隐私
    float dp_epsilon;              // 差分隐私epsilon
    float dp_delta;                 // 差分隐私delta
    float dp_clip_norm;             // 梯度裁剪范数
    int batch_size;                // 批次大小
    bool use_gpu;                  // 使用GPU
    char* aggregation_strategy;     // 聚合策略
    char* communication_protocol;   // 通信协议
} federated_config_t;

// 客户端信息结构体
typedef struct {
    int client_id;                  // 客户端ID
    char* client_address;          // 客户端地址
    client_state_t state;          // 客户端状态
    nn_module_t* local_model;      // 本地模型
    tensor_t** local_gradients;   // 本地梯度
    int num_gradients;             // 梯度数量
    float local_loss;              // 本地损失
    float local_accuracy;          // 本地准确率
    int data_size;                 // 数据量
    char* data_distribution;       // 数据分布
    bool is_active;                // 是否活跃
    time_t last_contact;           // 最后联系时间
} client_info_t;

// 联邦学习服务器结构体
typedef struct {
    federated_config_t config;     // 配置
    nn_module_t* global_model;     // 全局模型
    client_info_t** clients;       // 客户端列表
    int num_clients;               // 客户端数量
    int current_round;             // 当前轮数
    float global_loss;             // 全局损失
    float global_accuracy;         // 全局准确率
    pthread_mutex_t lock;          // 线程锁
    bool is_running;               // 是否运行中
    char* model_checkpoint_path;    // 模型检查点路径
    char* log_file_path;           // 日志文件路径
} federated_server_t;

// 联邦学习客户端结构体
typedef struct {
    int client_id;                  // 客户端ID
    federated_config_t config;      // 配置
    nn_module_t* local_model;      // 本地模型
    char* server_address;          // 服务器地址
    char* data_path;               // 数据路径
    bool is_training;              // 是否训练中
    pthread_t training_thread;     // 训练线程
} federated_client_t;

// 模型更新结构体
typedef struct {
    int client_id;                  // 客户端ID
    int round;                     // 轮数
    tensor_t** gradients;          // 梯度更新
    int num_gradients;             // 梯度数量
    float loss;                    // 损失
    float accuracy;                // 准确率
    int data_size;                 // 数据量
    time_t timestamp;              // 时间戳
} model_update_t;

// ==================== 服务器端接口 ====================

// 创建联邦学习服务器
federated_server_t* federated_server_create(federated_config_t* config,
                                           nn_module_t* initial_model);

// 销毁联邦学习服务器
void federated_server_destroy(federated_server_t* server);

// 注册客户端
int federated_server_register_client(federated_server_t* server,
                                    int client_id,
                                    const char* client_address);

// 开始联邦学习训练
int federated_server_start_training(federated_server_t* server);

// 停止联邦学习训练
int federated_server_stop_training(federated_server_t* server);

// 聚合客户端更新
int federated_server_aggregate_updates(federated_server_t* server,
                                      model_update_t** updates,
                                      int num_updates);

// 保存模型检查点
int federated_server_save_checkpoint(federated_server_t* server,
                                    int round);

// 加载模型检查点
int federated_server_load_checkpoint(federated_server_t* server,
                                    const char* checkpoint_path);

// ==================== 客户端端接口 ====================

// 创建联邦学习客户端
federated_client_t* federated_client_create(federated_config_t* config,
                                           int client_id,
                                           const char* server_address,
                                           const char* data_path);

// 销毁联邦学习客户端
void federated_client_destroy(federated_client_t* client);

// 连接到服务器
int federated_client_connect(federated_client_t* client);

// 开始本地训练
int federated_client_start_training(federated_client_t* client,
                                   int round);

// 上传模型更新
int federated_client_upload_update(federated_client_t* client,
                                  model_update_t* update);

// 下载全局模型
int federated_client_download_model(federated_client_t* client);

// ==================== 聚合算法实现 ====================

// 联邦平均算法
int federated_average_aggregate(nn_module_t* global_model,
                               model_update_t** updates,
                               int num_updates,
                               federated_config_t* config);

// FedProx算法
int fedprox_aggregate(nn_module_t* global_model,
                     model_update_t** updates,
                     int num_updates,
                     federated_config_t* config);

// FedAdam算法
int fedadam_aggregate(nn_module_t* global_model,
                     model_update_t** updates,
                     int num_updates,
                     federated_config_t* config);

// 差分隐私聚合
int differential_privacy_aggregate(nn_module_t* global_model,
                                  model_update_t** updates,
                                  int num_updates,
                                  federated_config_t* config);

// ==================== 通信协议 ====================

// RPC通信接口
typedef struct {
    int (*send_update)(const char* server_addr, model_update_t* update);
    int (*receive_model)(const char* server_addr, nn_module_t* model);
    int (*heartbeat)(const char* server_addr, int client_id);
    int (*get_global_model)(const char* server_addr, nn_module_t* model);
    int (*upload_local_update)(const char* server_addr, model_update_t* update);
    int (*download_config)(const char* server_addr, federated_config_t* config);
} rpc_interface_t;

// 联邦学习训练统计
typedef struct {
    // 全局统计
    int total_rounds;
    int completed_rounds;
    float global_loss;
    float global_accuracy;
    double total_training_time;
    
    // 客户端统计
    int active_clients;
    int total_clients;
    float average_client_loss;
    float average_client_accuracy;
    
    // 通信统计
    size_t total_bytes_transferred;
    int total_messages_sent;
    double average_communication_time;
    
    // 资源使用
    size_t server_memory_usage;
    size_t average_client_memory_usage;
    
} federated_training_stats_t;

// 联邦学习回调函数类型
typedef void (*FederatedTrainingCallback)(federated_server_t* server, void* user_data);

// ==================== 服务器端增强API ====================

// 初始化联邦学习服务器
int federated_server_init(federated_server_t* server);

// 启动联邦学习训练轮次
int federated_server_start_round(federated_server_t* server, int round_number);

// 聚合客户端更新（增强版）
int federated_server_aggregate_updates_enhanced(federated_server_t* server,
                                               model_update_t** updates,
                                               int num_updates,
                                               federated_algorithm_t algorithm);

// 模型验证
float federated_server_validate_model(federated_server_t* server, 
                                     tensor_t* validation_data,
                                     tensor_t* validation_labels);

// 客户端管理
int federated_server_add_client(federated_server_t* server, 
                               int client_id, 
                               const char* client_address);

int federated_server_remove_client(federated_server_t* server, int client_id);

int federated_server_update_client_status(federated_server_t* server, 
                                         int client_id, 
                                         client_state_t new_state);

// 安全与隐私
int federated_server_enable_differential_privacy(federated_server_t* server,
                                                float epsilon,
                                                float delta);

int federated_server_enable_secure_aggregation(federated_server_t* server);

int federated_server_enable_homomorphic_encryption(federated_server_t* server);

// ==================== 客户端端增强API ====================

// 初始化联邦学习客户端
int federated_client_init(federated_client_t* client);

// 本地训练（增强版）
int federated_client_train_enhanced(federated_client_t* client,
                                   tensor_t* local_data,
                                   tensor_t* local_labels,
                                   int epochs);

// 模型评估
float federated_client_evaluate_model(federated_client_t* client,
                                     tensor_t* test_data,
                                     tensor_t* test_labels);

// 数据预处理
int federated_client_preprocess_data(federated_client_t* client,
                                    tensor_t* raw_data,
                                    tensor_t* processed_data);

// 资源管理
int federated_client_optimize_memory(federated_client_t* client);

int federated_client_enable_compression(federated_client_t* client);

// ==================== 通信协议增强API ====================

// HTTP通信接口
int federated_http_send_update(const char* server_url, model_update_t* update);
int federated_http_receive_model(const char* server_url, nn_module_t* model);

// WebSocket通信接口
int federated_websocket_connect(const char* server_url, federated_client_t* client);
int federated_websocket_send_update(const char* server_url, model_update_t* update);
int federated_websocket_receive_model(const char* server_url, nn_module_t* model);

// gRPC通信接口
int federated_grpc_connect(const char* server_addr, federated_client_t* client);
int federated_grpc_send_update(const char* server_addr, model_update_t* update);
int federated_grpc_receive_model(const char* server_addr, nn_module_t* model);

// MQTT通信接口
int federated_mqtt_connect(const char* broker_addr, federated_client_t* client);
int federated_mqtt_send_update(const char* broker_addr, model_update_t* update);
int federated_mqtt_receive_model(const char* broker_addr, nn_module_t* model);

// ==================== 聚合算法增强API ====================

// 加权联邦平均
int weighted_federated_average(nn_module_t* global_model,
                             model_update_t** updates,
                             int num_updates,
                             federated_config_t* config);

// 自适应联邦学习
int adaptive_federated_learning(nn_module_t* global_model,
                               model_update_t** updates,
                               int num_updates,
                               federated_config_t* config);

// 个性化联邦学习
int personalized_federated_learning(nn_module_t* global_model,
                                   model_update_t** updates,
                                   int num_updates,
                                   federated_config_t* config);

// 联邦多任务学习
int federated_multi_task_learning(nn_module_t* global_model,
                                 model_update_t** updates,
                                 int num_updates,
                                 federated_config_t* config);

// ==================== 监控和调试API ====================

// 获取训练统计
federated_training_stats_t* federated_get_training_stats(federated_server_t* server);

// 注册训练回调
int federated_register_callback(federated_server_t* server,
                               FederatedTrainingCallback callback,
                               void* user_data);

// 性能分析
void federated_performance_profile(federated_server_t* server);

// 调试信息输出
void federated_debug_info(federated_server_t* server);

// ==================== 工具函数 ====================

// 配置验证
bool federated_validate_config(federated_config_t* config);

// 模型序列化
int federated_serialize_model(nn_module_t* model, char** buffer, size_t* size);

// 模型反序列化
int federated_deserialize_model(const char* buffer, size_t size, nn_module_t** model);

// 更新序列化
int federated_serialize_update(model_update_t* update, char** buffer, size_t* size);

// 更新反序列化
int federated_deserialize_update(const char* buffer, size_t size, model_update_t** update);

// 计算更新大小
size_t federated_calculate_update_size(model_update_t* update);

// 最终化联邦学习环境
int federated_learning_finalize(federated_server_t* server);

#ifdef __cplusplus
}
#endif

#endif // FEDERATED_LEARNING_H
    int (*broadcast_model)(federated_server_t* server, nn_module_t* model);
} rpc_interface_t;

// HTTP通信接口
typedef struct {
    int (*post_update)(const char* url, model_update_t* update);
    int (*get_model)(const char* url, nn_module_t* model);
} http_interface_t;

// ==================== 安全与隐私 ====================

// 差分隐私噪声添加
int add_differential_privacy_noise(tensor_t* gradient,
                                  float epsilon,
                                  float delta,
                                  float sensitivity);

// 安全聚合协议
int secure_aggregation_protocol(model_update_t** updates,
                               int num_updates,
                               const char* protocol);

// 同态加密支持
int homomorphic_encryption_aggregate(model_update_t** encrypted_updates,
                                    int num_updates);

// ==================== 监控与日志 ====================

// 训练监控结构体
typedef struct {
    int round;                      // 当前轮数
    float global_loss;              // 全局损失
    float global_accuracy;          // 全局准确率
    int active_clients;             // 活跃客户端数
    time_t start_time;              // 开始时间
    time_t end_time;                // 结束时间
} training_monitor_t;

// 创建训练监控器
training_monitor_t* training_monitor_create(const char* log_file);

// 更新监控信息
int training_monitor_update(training_monitor_t* monitor,
                           int round,
                           float loss,
                           float accuracy,
                           int active_clients);

// 保存监控日志
int training_monitor_save_log(training_monitor_t* monitor);

// ==================== 工具函数 ====================

// 计算模型差异
float calculate_model_difference(nn_module_t* model1, nn_module_t* model2);

// 验证模型完整性
bool validate_model_integrity(nn_module_t* model);

// 序列化模型更新
char* serialize_model_update(model_update_t* update, size_t* serialized_size);

// 反序列化模型更新
model_update_t* deserialize_model_update(const char* data, size_t data_size);

#ifdef __cplusplus
}
#endif

#endif // FEDERATED_LEARNING_H