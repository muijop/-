#ifndef DISTRIBUTED_TRAINING_H
#define DISTRIBUTED_TRAINING_H

#include "dynamic_graph.h"
#include "static_graph_optimizer.h"
#include <stdbool.h>
#include <stddef.h>

// 分布式训练配置
typedef enum {
    DISTRIBUTED_NONE = 0,
    DISTRIBUTED_DATA_PARALLEL,      // 数据并行 (PyTorch/TensorFlow风格)
    DISTRIBUTED_MODEL_PARALLEL,     // 模型并行 (Megatron-LM风格)
    DISTRIBUTED_PIPELINE_PARALLEL,  // 流水线并行 (GPipe风格)
    DISTRIBUTED_HYBRID,            // 混合并行
    DISTRIBUTED_FEDERATED          // 联邦学习
} DistributedStrategy;

typedef enum {
    COMM_NCCL = 0,     // NVIDIA NCCL
    COMM_MPI,          // MPI
    COMM_GLOO,         // Facebook Gloo
    COMM_RPC,          // 远程过程调用
    COMM_GRPC,         // Google gRPC
    COMM_CUSTOM        // 自定义通信
} CommunicationBackend;

typedef enum {
    REDUCE_SUM = 0,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_PRODUCT
} ReductionOp;

typedef enum {
    ALLREDUCE_RING = 0,      // 环形AllReduce
    ALLREDUCE_TREE,          // 树形AllReduce
    ALLREDUCE_HIERARCHICAL,  // 分层AllReduce
    ALLREDUCE_STAR,          // 星形AllReduce
    ALLREDUCE_OPTIMAL        // 最优AllReduce
} AllReduceAlgorithm;

typedef struct {
    // 基础配置
    DistributedStrategy strategy;
    CommunicationBackend backend;
    int world_size;          // 总进程数
    int rank;                // 当前进程ID
    int local_rank;          // 本地进程ID
    int master_addr;          // 主节点地址
    int master_port;          // 主节点端口
    
    // 高级配置
    bool enable_gradient_compression;
    float compression_ratio;    // 压缩比例
    bool enable_mixed_precision;
    bool enable_async_comm;
    int async_queue_size;
    
    // AllReduce优化
    AllReduceAlgorithm allreduce_algo;
    bool enable_allreduce_opt;
    int allreduce_chunk_size;
    int allreduce_bucket_size;
    
    // 容错配置
    bool enable_fault_tolerance;
    int checkpoint_interval;
    int max_retry_attempts;
    
    // 性能调优
    int num_threads;
    int num_streams;
    bool enable_cuda_graph;
    bool enable_p2p_comm;
    
    // 中文支持
    bool enable_chinese_support;
    const char* log_level;
    
} DistributedConfig;

// 分布式训练器
typedef struct DistributedTrainer {
    DistributedConfig config;
    DynamicGraph* model;
    GraphCompiler* compiler;
    
    // 通信相关
    void* comm_handle;          // 通信句柄
    void* nccl_comm;           // NCCL通信器
    void* mpi_comm;            // MPI通信器
    void* rpc_client;          // RPC客户端
    
    // 梯度管理
    DynamicTensor** gradients;      // 梯度张量
    size_t num_gradients;
    DynamicTensor** compressed_gradients; // 压缩梯度
    
    // 参数服务器
    void* parameter_server;     // 参数服务器
    bool* param_is_local;      // 参数是否本地
    
    // 流水线并行
    int pipeline_stage;        // 流水线阶段
    int pipeline_micro_batch_size;
    DynamicTensor*** pipeline_buffers;
    
    // 性能统计
    double comm_time;
    double compute_time;
    double total_time;
    size_t bytes_transferred;
    
    // 状态管理
    bool is_initialized;
    bool is_training;
    int epoch;
    int step;
    
    // 错误处理
    char error_message[1024];
    int error_code;
    
} DistributedTrainer;

// 分布式优化器
typedef struct DistributedOptimizer {
    // 基础优化器
    OptimizerType base_optimizer;
    void* base_optimizer_handle;
    
    // 分布式配置
    DistributedConfig* config;
    DistributedTrainer* trainer;
    
    // 学习率调度
    float base_lr;
    float warmup_lr;
    int warmup_steps;
    float min_lr;
    
    // 梯度缩放
    float gradient_scale;
    bool enable_gradient_clipping;
    float max_gradient_norm;
    
    // 状态管理
    DynamicTensor** momentum_buffers;
    DynamicTensor** variance_buffers;
    size_t num_parameters;
    
} DistributedOptimizer;

// 分布式数据加载器
typedef struct DistributedDataLoader {
    // 数据集
    void* dataset;
    size_t dataset_size;
    size_t batch_size;
    size_t num_workers;
    
    // 分布式配置
    int world_size;
    int rank;
    size_t samples_per_rank;
    size_t* sample_indices;
    
    // 数据分片
    bool enable_data_sharding;
    bool enable_caching;
    char cache_dir[256];
    
    // 性能优化
    bool pin_memory;
    bool prefetch;
    int prefetch_factor;
    
    // 中文支持
    bool enable_chinese_labels;
    
} DistributedDataLoader;

// 分布式检查点
typedef struct DistributedCheckpoint {
    // 检查点配置
    char checkpoint_dir[512];
    int save_interval;
    int max_checkpoints;
    bool enable_async_save;
    
    // 模型状态
    DynamicTensor** model_state;
    DynamicTensor** optimizer_state;
    size_t num_state_tensors;
    
    // 分布式状态
    int epoch;
    int step;
    int world_size;
    int rank;
    double training_time;
    float validation_loss;
    float validation_accuracy;
    
} DistributedCheckpoint;

// 分布式训练统计
typedef struct {
    // 性能统计
    double total_training_time;
    double communication_time;
    double computation_time;
    double synchronization_time;
    
    // 资源使用
    size_t memory_usage_per_rank[1024]; // 每个rank的内存使用
    size_t gpu_memory_usage_per_rank[1024]; // 每个rank的GPU内存使用
    float cpu_utilization_per_rank[1024]; // 每个rank的CPU利用率
    
    // 通信统计
    size_t bytes_sent_per_rank[1024];
    size_t bytes_received_per_rank[1024];
    int messages_sent_per_rank[1024];
    int messages_received_per_rank[1024];
    
    // 训练进度
    int current_epoch;
    int current_step;
    float current_loss;
    float current_accuracy;
    
} DistributedTrainingStats;

// 分布式训练回调函数类型
typedef void (*DistributedTrainingCallback)(DistributedTrainer* trainer, void* user_data);

// ==================== 分布式训练核心API ====================

// 初始化分布式训练环境
int distributed_training_init(DistributedConfig* config);

// 创建分布式训练器
DistributedTrainer* distributed_trainer_create(DistributedConfig* config, DynamicGraph* model);

// 销毁分布式训练器
void distributed_trainer_destroy(DistributedTrainer* trainer);

// 开始分布式训练
int distributed_trainer_train(DistributedTrainer* trainer, 
                             DynamicTensor* input, 
                             DynamicTensor* target, 
                             int num_epochs);

// 分布式推理
DynamicTensor* distributed_trainer_predict(DistributedTrainer* trainer, DynamicTensor* input);

// 保存分布式检查点
int distributed_trainer_save_checkpoint(DistributedTrainer* trainer, const char* checkpoint_path);

// 加载分布式检查点
int distributed_trainer_load_checkpoint(DistributedTrainer* trainer, const char* checkpoint_path);

// 获取分布式训练统计
DistributedTrainingStats* distributed_trainer_get_stats(DistributedTrainer* trainer);

// ==================== 通信原语API ====================

// AllReduce操作
int distributed_all_reduce(DynamicTensor* tensor, ReductionOp op, DistributedConfig* config);

// Broadcast操作
int distributed_broadcast(DynamicTensor* tensor, int root_rank, DistributedConfig* config);

// Scatter操作
int distributed_scatter(DynamicTensor* send_buffer, DynamicTensor* recv_buffer, int root_rank, DistributedConfig* config);

// Gather操作
int distributed_gather(DynamicTensor* send_buffer, DynamicTensor* recv_buffer, int root_rank, DistributedConfig* config);

// Reduce操作
int distributed_reduce(DynamicTensor* tensor, ReductionOp op, int root_rank, DistributedConfig* config);

// AllGather操作
int distributed_all_gather(DynamicTensor* send_buffer, DynamicTensor* recv_buffer, DistributedConfig* config);

// ==================== 模型并行API ====================

// 模型分片
int distributed_model_sharding(DynamicGraph* model, int num_shards, DistributedConfig* config);

// 流水线并行配置
int distributed_pipeline_parallel_config(DynamicGraph* model, int num_stages, DistributedConfig* config);

// 张量并行配置
int distributed_tensor_parallel_config(DynamicGraph* model, int tensor_parallel_size, DistributedConfig* config);

// ==================== 性能优化API ====================

// 启用梯度压缩
int distributed_enable_gradient_compression(DistributedTrainer* trainer, float compression_ratio);

// 启用异步通信
int distributed_enable_async_communication(DistributedTrainer* trainer, int queue_size);

// 启用混合精度训练
int distributed_enable_mixed_precision(DistributedTrainer* trainer);

// 优化通信调度
int distributed_optimize_communication_schedule(DistributedTrainer* trainer);

// ==================== 容错和恢复API ====================

// 启用容错训练
int distributed_enable_fault_tolerance(DistributedTrainer* trainer, int checkpoint_interval);

// 从故障中恢复
int distributed_recover_from_failure(DistributedTrainer* trainer);

// 检查点管理
int distributed_manage_checkpoints(DistributedTrainer* trainer, int max_checkpoints);

// ==================== 监控和调试API ====================

// 注册训练回调
int distributed_register_callback(DistributedTrainer* trainer, 
                                 DistributedTrainingCallback callback, 
                                 void* user_data);

// 获取训练进度
float distributed_get_training_progress(DistributedTrainer* trainer);

// 性能分析
void distributed_performance_profile(DistributedTrainer* trainer);

// 调试信息输出
void distributed_debug_info(DistributedTrainer* trainer);

// ==================== 工具函数 ====================

// 获取世界大小
int distributed_get_world_size(DistributedConfig* config);

// 获取当前rank
int distributed_get_rank(DistributedConfig* config);

// 获取本地rank
int distributed_get_local_rank(DistributedConfig* config);

// 同步所有进程
int distributed_barrier(DistributedConfig* config);

// 最终化分布式训练环境
int distributed_training_finalize(DistributedConfig* config);
    float best_metric;
    
    // 容错
    bool enable_redundancy;
    int redundancy_factor;
    
} DistributedCheckpoint;

// 联邦学习
typedef struct FederatedLearningConfig {
    // 联邦学习配置
    int num_clients;
    int rounds;
    int local_epochs;
    float client_fraction;
    
    // 隐私保护
    bool enable_dp;              // 差分隐私
    float dp_epsilon;
    float dp_delta;
    
    // 聚合算法
    enum {
        FEDAVG = 0,      // FedAvg
        FEDPROX,         // FedProx
        FEDADAM,         // FedAdam
        SCAFFOLD         // SCAFFOLD
    } aggregation_algo;
    
    // 通信效率
    bool enable_sparse_comm;
    float sparsity_ratio;
    bool enable_quantization;
    int quantization_bits;
    
} FederatedLearningConfig;

// 分布式监控
typedef struct DistributedMetrics {
    // 训练指标
    float loss;
    float accuracy;
    float learning_rate;
    int epoch;
    int step;
    
    // 分布式指标
    double comm_time;
    double compute_time;
    size_t bytes_sent;
    size_t bytes_received;
    int num_syncs;
    
    // 系统指标
    double cpu_usage;
    double memory_usage;
    double gpu_usage;
    double gpu_memory_usage;
    
    // 网络指标
    double network_latency;
    double bandwidth_utilization;
    int packet_loss;
    
} DistributedMetrics;

// 分布式错误码
typedef enum {
    DIST_SUCCESS = 0,
    DIST_ERROR_INIT_FAILED,
    DIST_ERROR_COMM_FAILED,
    DIST_ERROR_SYNC_FAILED,
    DIST_ERROR_CHECKPOINT_FAILED,
    DIST_ERROR_MEMORY_ALLOC_FAILED,
    DIST_ERROR_UNSUPPORTED_STRATEGY,
    DIST_ERROR_FAULT_TOLERANCE_FAILED
} DistributedError;

// 核心API函数

// 初始化分布式环境
DistributedTrainer* distributed_trainer_create(DynamicGraph* model, DistributedConfig* config);
void distributed_trainer_destroy(DistributedTrainer* trainer);

// 初始化分布式配置
DistributedConfig distributed_config_default(void);
DistributedConfig distributed_config_data_parallel(int world_size, int rank);
DistributedConfig distributed_config_model_parallel(int world_size, int rank, int pipeline_stages);
DistributedConfig distributed_config_hybrid(int world_size, int rank);

// 分布式训练
bool distributed_trainer_train_epoch(DistributedTrainer* trainer, 
                                     DistributedDataLoader* dataloader,
                                     DistributedOptimizer* optimizer,
                                     size_t num_epochs);

bool distributed_trainer_train_step(DistributedTrainer* trainer,
                                    DynamicTensor** inputs,
                                    DynamicTensor** targets,
                                    DistributedOptimizer* optimizer);

// 梯度同步
bool distributed_trainer_sync_gradients(DistributedTrainer* trainer);
bool distributed_trainer_allreduce_gradients(DistributedTrainer* trainer, ReductionOp op);
bool distributed_trainer_compress_gradients(DistributedTrainer* trainer);

// 参数服务器
bool distributed_trainer_init_parameter_server(DistributedTrainer* trainer);
bool distributed_trainer_pull_parameters(DistributedTrainer* trainer);
bool distributed_trainer_push_gradients(DistributedTrainer* trainer);

// 检查点管理
bool distributed_trainer_save_checkpoint(DistributedTrainer* trainer, DistributedCheckpoint* checkpoint);
bool distributed_trainer_load_checkpoint(DistributedTrainer* trainer, DistributedCheckpoint* checkpoint);
bool distributed_trainer_resume_from_checkpoint(DistributedTrainer* trainer, const char* checkpoint_path);

// 性能监控
DistributedMetrics distributed_trainer_get_metrics(DistributedTrainer* trainer);
bool distributed_trainer_log_metrics(DistributedTrainer* trainer, const char* log_dir);
bool distributed_trainer_export_metrics(DistributedTrainer* trainer, const char* filename);

// 联邦学习
bool distributed_trainer_federated_train(DistributedTrainer* trainer, 
                                        FederatedLearningConfig* fed_config,
                                        void** client_datasets,
                                        size_t num_clients);

// 容错处理
bool distributed_trainer_enable_fault_tolerance(DistributedTrainer* trainer);
bool distributed_trainer_handle_failure(DistributedTrainer* trainer, int failed_rank);
bool distributed_trainer_recover_from_failure(DistributedTrainer* trainer);

// 通信优化
bool distributed_trainer_enable_gradient_compression(DistributedTrainer* trainer, float compression_ratio);
bool distributed_trainer_enable_async_communication(DistributedTrainer* trainer);
bool distributed_trainer_optimize_allreduce(DistributedTrainer* trainer, AllReduceAlgorithm algo);

// 分布式优化器
DistributedOptimizer* distributed_optimizer_create(OptimizerType base_optimizer,
                                                  DistributedTrainer* trainer,
                                                  float learning_rate);
void distributed_optimizer_destroy(DistributedOptimizer* optimizer);
bool distributed_optimizer_step(DistributedOptimizer* optimizer);
bool distributed_optimizer_zero_grad(DistributedOptimizer* optimizer);

// 分布式数据加载
DistributedDataLoader* distributed_dataloader_create(void* dataset,
                                                  size_t batch_size,
                                                  DistributedConfig* config);
void distributed_dataloader_destroy(DistributedDataLoader* dataloader);
bool distributed_dataloader_set_epoch(DistributedDataLoader* dataloader, int epoch);
void** distributed_dataloader_next_batch(DistributedDataLoader* dataloader);

// 性能分析
bool distributed_trainer_profile_communication(DistributedTrainer* trainer, const char* output_file);
bool distributed_trainer_profile_computation(DistributedTrainer* trainer, const char* output_file);
bool distributed_trainer_analyze_bottlenecks(DistributedTrainer* trainer);

// 错误处理（中文支持）
const char* distributed_error_string(DistributedError error);
const char* distributed_error_string_chinese(DistributedError error);
bool distributed_trainer_set_error_handler(DistributedTrainer* trainer, void (*handler)(DistributedError, const char*));

// 高级功能
bool distributed_trainer_enable_dynamic_loss_scaling(DistributedTrainer* trainer);
bool distributed_trainer_enable_gradient_accumulation(DistributedTrainer* trainer, int accumulation_steps);
bool distributed_trainer_enable_mixed_precision(DistributedTrainer* trainer);

// 集群管理
bool distributed_trainer_discover_nodes(DistributedTrainer* trainer);
bool distributed_trainer_health_check(DistributedTrainer* trainer);
bool distributed_trainer_load_balance(DistributedTrainer* trainer);

// 扩展性支持
bool distributed_trainer_register_custom_strategy(DistributedTrainer* trainer, 
                                                const char* name,
                                                void* strategy_impl);
bool distributed_trainer_register_custom_comm_backend(DistributedTrainer* trainer,
                                                     const char* name,
                                                     void* backend_impl);

// 兼容性支持
bool distributed_trainer_pytorch_compatible(DistributedTrainer* trainer);
bool distributed_trainer_tensorflow_compatible(DistributedTrainer* trainer);
bool distributed_trainer_horovod_compatible(DistributedTrainer* trainer);

#endif // DISTRIBUTED_TRAINING_H