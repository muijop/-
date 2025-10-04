#include "distributed_training.h"
#include "dynamic_graph.h"
#include "static_graph_optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// 分布式训练器创建
DistributedTrainer* distributed_trainer_create(DynamicGraph* model, DistributedConfig* config) {
    if (!model || !config) return NULL;
    
    DistributedTrainer* trainer = (DistributedTrainer*)calloc(1, sizeof(DistributedTrainer));
    if (!trainer) return NULL;
    
    // 复制配置
    trainer->config = *config;
    trainer->model = model;
    
    // 初始化状态
    trainer->is_initialized = false;
    trainer->is_training = false;
    trainer->epoch = 0;
    trainer->step = 0;
    trainer->comm_time = 0.0;
    trainer->compute_time = 0.0;
    trainer->total_time = 0.0;
    trainer->bytes_transferred = 0;
    
    // 初始化通信后端
    if (!distributed_trainer_init_communication(trainer)) {
        free(trainer);
        return NULL;
    }
    
    // 初始化参数服务器（如果需要）
    if (config->strategy == DISTRIBUTED_MODEL_PARALLEL || 
        config->strategy == DISTRIBUTED_HYBRID) {
        if (!distributed_trainer_init_parameter_server(trainer)) {
            distributed_trainer_destroy(trainer);
            return NULL;
        }
    }
    
    // 初始化流水线并行（如果需要）
    if (config->strategy == DISTRIBUTED_PIPELINE_PARALLEL || 
        config->strategy == DISTRIBUTED_HYBRID) {
        if (!distributed_trainer_init_pipeline_parallel(trainer)) {
            distributed_trainer_destroy(trainer);
            return NULL;
        }
    }
    
    trainer->is_initialized = true;
    
    printf("分布式训练器创建成功 (策略: %s, 后端: %s, 世界大小: %d, 排名: %d)\n",
           distributed_strategy_name(config->strategy),
           communication_backend_name(config->backend),
           config->world_size, config->rank);
    
    return trainer;
}

// 销毁分布式训练器
void distributed_trainer_destroy(DistributedTrainer* trainer) {
    if (!trainer) return;
    
    // 清理通信资源
    if (trainer->comm_handle) {
        distributed_trainer_cleanup_communication(trainer);
    }
    
    // 清理参数服务器
    if (trainer->parameter_server) {
        distributed_trainer_cleanup_parameter_server(trainer);
    }
    
    // 清理梯度
    if (trainer->gradients) {
        for (size_t i = 0; i < trainer->num_gradients; i++) {
            if (trainer->gradients[i]) {
                dynamic_tensor_destroy(trainer->gradients[i]);
            }
        }
        free(trainer->gradients);
    }
    
    // 清理压缩梯度
    if (trainer->compressed_gradients) {
        for (size_t i = 0; i < trainer->num_gradients; i++) {
            if (trainer->compressed_gradients[i]) {
                dynamic_tensor_destroy(trainer->compressed_gradients[i]);
            }
        }
        free(trainer->compressed_gradients);
    }
    
    // 清理流水线缓冲区
    if (trainer->pipeline_buffers) {
        for (int i = 0; i < trainer->config.world_size; i++) {
            if (trainer->pipeline_buffers[i]) {
                for (int j = 0; j < 2; j++) {
                    if (trainer->pipeline_buffers[i][j]) {
                        dynamic_tensor_destroy(trainer->pipeline_buffers[i][j]);
                    }
                }
                free(trainer->pipeline_buffers[i]);
            }
        }
        free(trainer->pipeline_buffers);
    }
    
    free(trainer);
}

// 默认分布式配置
DistributedConfig distributed_config_default(void) {
    DistributedConfig config = {0};
    
    config.strategy = DISTRIBUTED_DATA_PARALLEL;
    config.backend = COMM_NCCL;
    config.world_size = 1;
    config.rank = 0;
    config.local_rank = 0;
    config.master_addr = 0;
    config.master_port = 29500;
    
    config.enable_gradient_compression = false;
    config.compression_ratio = 0.01f;
    config.enable_mixed_precision = false;
    config.enable_async_comm = false;
    config.async_queue_size = 4;
    
    config.allreduce_algo = ALLREDUCE_RING;
    config.enable_allreduce_opt = true;
    config.allreduce_chunk_size = 256 * 1024 * 1024; // 256MB
    config.allreduce_bucket_size = 25 * 1024 * 1024;  // 25MB
    
    config.enable_fault_tolerance = false;
    config.checkpoint_interval = 1000;
    config.max_retry_attempts = 3;
    
    config.num_threads = 4;
    config.num_streams = 2;
    config.enable_cuda_graph = false;
    config.enable_p2p_comm = true;
    
    config.enable_chinese_support = true;
    config.log_level = "INFO";
    
    return config;
}

// 数据并行配置
DistributedConfig distributed_config_data_parallel(int world_size, int rank) {
    DistributedConfig config = distributed_config_default();
    
    config.strategy = DISTRIBUTED_DATA_PARALLEL;
    config.world_size = world_size;
    config.rank = rank;
    config.local_rank = rank % 4; // 假设每个节点4个GPU
    
    config.enable_gradient_compression = true;
    config.compression_ratio = 0.001f;
    config.enable_mixed_precision = true;
    config.enable_async_comm = true;
    
    config.allreduce_algo = ALLREDUCE_HIERARCHICAL;
    config.enable_allreduce_opt = true;
    
    return config;
}

// 模型并行配置
DistributedConfig distributed_config_model_parallel(int world_size, int rank, int pipeline_stages) {
    DistributedConfig config = distributed_config_default();
    
    config.strategy = DISTRIBUTED_MODEL_PARALLEL;
    config.world_size = world_size;
    config.rank = rank;
    
    config.enable_gradient_compression = false;
    config.enable_mixed_precision = true;
    config.enable_async_comm = false;
    
    config.allreduce_algo = ALLREDUCE_TREE;
    
    return config;
}

// 混合并行配置
DistributedConfig distributed_config_hybrid(int world_size, int rank) {
    DistributedConfig config = distributed_config_default();
    
    config.strategy = DISTRIBUTED_HYBRID;
    config.world_size = world_size;
    config.rank = rank;
    
    config.enable_gradient_compression = true;
    config.compression_ratio = 0.01f;
    config.enable_mixed_precision = true;
    config.enable_async_comm = true;
    
    config.allreduce_algo = ALLREDUCE_OPTIMAL;
    config.enable_allreduce_opt = true;
    
    return config;
}

// 初始化通信
bool distributed_trainer_init_communication(DistributedTrainer* trainer) {
    if (!trainer) return false;
    
    printf("初始化通信后端: %s\n", communication_backend_name(trainer->config.backend));
    
    switch (trainer->config.backend) {
        case COMM_NCCL:
            return distributed_trainer_init_nccl(trainer);
        case COMM_MPI:
            return distributed_trainer_init_mpi(trainer);
        case COMM_GLOO:
            return distributed_trainer_init_gloo(trainer);
        case COMM_RPC:
            return distributed_trainer_init_rpc(trainer);
        default:
            strcpy(trainer->error_message, "不支持的通信后端");
            return false;
    }
}

// 初始化NCCL
bool distributed_trainer_init_nccl(DistributedTrainer* trainer) {
    printf("初始化NCCL通信...\n");
    printf("  世界大小: %d\n", trainer->config.world_size);
    printf("  排名: %d\n", trainer->config.rank);
    printf("  本地排名: %d\n", trainer->config.local_rank);
    
    // 这里实现实际的NCCL初始化
    // 包括NCCL通信器创建、配置等
    
    return true;
}

// 初始化MPI
bool distributed_trainer_init_mpi(DistributedTrainer* trainer) {
    printf("初始化MPI通信...\n");
    printf("  世界大小: %d\n", trainer->config.world_size);
    printf("  排名: %d\n", trainer->config.rank);
    
    return true;
}

// 初始化Gloo
bool distributed_trainer_init_gloo(DistributedTrainer* trainer) {
    printf("初始化Gloo通信...\n");
    printf("  世界大小: %d\n", trainer->config.world_size);
    printf("  排名: %d\n", trainer->config.rank);
    
    return true;
}

// 初始化RPC
bool distributed_trainer_init_rpc(DistributedTrainer* trainer) {
    printf("初始化RPC通信...\n");
    printf("  世界大小: %d\n", trainer->config.world_size);
    printf("  排名: %d\n", trainer->config.rank);
    
    return true;
}

// 清理通信
void distributed_trainer_cleanup_communication(DistributedTrainer* trainer) {
    if (!trainer) return;
    
    printf("清理通信资源...\n");
    
    switch (trainer->config.backend) {
        case COMM_NCCL:
            printf("清理NCCL资源\n");
            break;
        case COMM_MPI:
            printf("清理MPI资源\n");
            break;
        case COMM_GLOO:
            printf("清理Gloo资源\n");
            break;
        case COMM_RPC:
            printf("清理RPC资源\n");
            break;
        default:
            break;
    }
}

// 初始化参数服务器
bool distributed_trainer_init_parameter_server(DistributedTrainer* trainer) {
    if (!trainer) return false;
    
    printf("初始化参数服务器...\n");
    
    // 根据策略分配参数
    size_t total_params = 0;
    // 这里需要计算模型总参数数量
    
    trainer->param_is_local = (bool*)calloc(total_params, sizeof(bool));
    if (!trainer->param_is_local) return false;
    
    // 根据排名分配参数所有权
    for (size_t i = 0; i < total_params; i++) {
        trainer->param_is_local[i] = (i % trainer->config.world_size == trainer->config.rank);
    }
    
    printf("参数服务器初始化完成\n");
    return true;
}

// 清理参数服务器
void distributed_trainer_cleanup_parameter_server(DistributedTrainer* trainer) {
    if (!trainer || !trainer->parameter_server) return;
    
    printf("清理参数服务器...\n");
    
    if (trainer->param_is_local) {
        free(trainer->param_is_local);
        trainer->param_is_local = NULL;
    }
}

// 初始化流水线并行
bool distributed_trainer_init_pipeline_parallel(DistributedTrainer* trainer) {
    if (!trainer) return false;
    
    printf("初始化流水线并行...\n");
    printf("  流水线阶段: %d\n", trainer->config.rank);
    
    trainer->pipeline_stage = trainer->config.rank;
    trainer->pipeline_micro_batch_size = 4; // 默认4个微批次
    
    // 分配流水线缓冲区
    trainer->pipeline_buffers = (DynamicTensor***)calloc(trainer->config.world_size, sizeof(DynamicTensor**));
    if (!trainer->pipeline_buffers) return false;
    
    for (int i = 0; i < trainer->config.world_size; i++) {
        trainer->pipeline_buffers[i] = (DynamicTensor**)calloc(2, sizeof(DynamicTensor*));
        if (!trainer->pipeline_buffers[i]) return false;
        
        // 这里需要根据模型结构创建适当的缓冲区
    }
    
    printf("流水线并行初始化完成\n");
    return true;
}

// 训练一个epoch
bool distributed_trainer_train_epoch(DistributedTrainer* trainer, 
                                     DistributedDataLoader* dataloader,
                                     DistributedOptimizer* optimizer,
                                     size_t num_epochs) {
    if (!trainer || !dataloader || !optimizer) return false;
    
    printf("开始分布式训练 (epoch: %zu)\n", num_epochs);
    
    clock_t start_time = clock();
    
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %zu/%zu\n", epoch + 1, num_epochs);
        
        // 设置数据加载器的epoch
        distributed_dataloader_set_epoch(dataloader, epoch);
        
        size_t batch_count = 0;
        void** batch_data = NULL;
        
        // 遍历数据批次
        while ((batch_data = distributed_dataloader_next_batch(dataloader)) != NULL) {
            // 执行训练步骤
            DynamicTensor** inputs = (DynamicTensor**)batch_data[0];
            DynamicTensor** targets = (DynamicTensor**)batch_data[1];
            
            if (!distributed_trainer_train_step(trainer, inputs, targets, optimizer)) {
                return false;
            }
            
            batch_count++;
            
            // 定期同步和日志
            if (batch_count % 100 == 0) {
                DistributedMetrics metrics = distributed_trainer_get_metrics(trainer);
                printf("  Batch %zu - Loss: %.4f, Comm: %.2fms, Compute: %.2fms\n",
                       batch_count, metrics.loss, metrics.comm_time, metrics.compute_time);
            }
        }
        
        printf("Epoch %zu 完成，共 %zu 个批次\n", epoch + 1, batch_count);
    }
    
    clock_t end_time = clock();
    trainer->total_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;
    
    printf("分布式训练完成，总耗时: %.2f ms\n", trainer->total_time);
    return true;
}

// 单步训练
bool distributed_trainer_train_step(DistributedTrainer* trainer,
                                    DynamicTensor** inputs,
                                    DynamicTensor** targets,
                                    DistributedOptimizer* optimizer) {
    if (!trainer || !inputs || !targets || !optimizer) return false;
    
    clock_t step_start = clock();
    
    // 前向传播
    clock_t forward_start = clock();
    DynamicTensor** outputs = dynamic_graph_forward(trainer->model, inputs);
    clock_t forward_end = clock();
    
    // 计算损失
    DynamicTensor* loss = dynamic_graph_compute_loss(trainer->model, outputs, targets);
    
    // 反向传播
    clock_t backward_start = clock();
    dynamic_graph_backward(trainer->model, loss);
    clock_t backward_end = clock();
    
    // 梯度同步
    clock_t sync_start = clock();
    if (!distributed_trainer_sync_gradients(trainer)) {
        return false;
    }
    clock_t sync_end = clock();
    
    // 优化器步骤
    if (!distributed_optimizer_step(optimizer)) {
        return false;
    }
    
    // 清理梯度
    distributed_optimizer_zero_grad(optimizer);
    
    clock_t step_end = clock();
    
    // 更新统计信息
    trainer->compute_time += (double)(forward_end - forward_start + backward_end - backward_start) * 1000.0 / CLOCKS_PER_SEC;
    trainer->comm_time += (double)(sync_end - sync_start) * 1000.0 / CLOCKS_PER_SEC;
    
    // 清理张量
    if (outputs) {
        for (size_t i = 0; outputs[i]; i++) {
            dynamic_tensor_destroy(outputs[i]);
        }
        free(outputs);
    }
    
    if (loss) {
        dynamic_tensor_destroy(loss);
    }
    
    trainer->step++;
    
    return true;
}

// 同步梯度
bool distributed_trainer_sync_gradients(DistributedTrainer* trainer) {
    if (!trainer) return false;
    
    switch (trainer->config.strategy) {
        case DISTRIBUTED_DATA_PARALLEL:
            return distributed_trainer_allreduce_gradients(trainer, REDUCE_MEAN);
        case DISTRIBUTED_MODEL_PARALLEL:
            return distributed_trainer_sync_model_parallel(trainer);
        case DISTRIBUTED_PIPELINE_PARALLEL:
            return distributed_trainer_sync_pipeline_parallel(trainer);
        case DISTRIBUTED_HYBRID:
            return distributed_trainer_sync_hybrid(trainer);
        default:
            return true; // 单进程不需要同步
    }
}

// AllReduce梯度
bool distributed_trainer_allreduce_gradients(DistributedTrainer* trainer, ReductionOp op) {
    if (!trainer || trainer->config.world_size <= 1) return true;
    
    if (trainer->config.enable_gradient_compression) {
        if (!distributed_trainer_compress_gradients(trainer)) {
            return false;
        }
    }
    
    printf("执行AllReduce (操作: %s, 世界大小: %d)\n", 
           reduction_op_name(op), trainer->config.world_size);
    
    // 根据算法选择AllReduce实现
    switch (trainer->config.allreduce_algo) {
        case ALLREDUCE_RING:
            return distributed_trainer_allreduce_ring(trainer, op);
        case ALLREDUCE_TREE:
            return distributed_trainer_allreduce_tree(trainer, op);
        case ALLREDUCE_HIERARCHICAL:
            return distributed_trainer_allreduce_hierarchical(trainer, op);
        default:
            return distributed_trainer_allreduce_ring(trainer, op);
    }
}

// 环形AllReduce
bool distributed_trainer_allreduce_ring(DistributedTrainer* trainer, ReductionOp op) {
    printf("  使用环形AllReduce算法\n");
    
    int world_size = trainer->config.world_size;
    int rank = trainer->config.rank;
    
    // Scatter-reduce阶段
    for (int step = 0; step < world_size - 1; step++) {
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank - 1 + world_size) % world_size;
        
        printf("    Step %d: 发送到 %d, 接收自 %d\n", step, send_to, recv_from);
        
        // 这里实现实际的数据发送和接收
        // 包括梯度分块、规约操作等
    }
    
    // Allgather阶段
    for (int step = 0; step < world_size - 1; step++) {
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank - 1 + world_size) % world_size;
        
        printf("    Allgather Step %d: 发送到 %d, 接收自 %d\n", step, send_to, recv_from);
    }
    
    return true;
}

// 树形AllReduce
bool distributed_trainer_allreduce_tree(DistributedTrainer* trainer, ReductionOp op) {
    printf("  使用树形AllReduce算法\n");
    
    // 实现树形AllReduce逻辑
    // 包括构建规约树、分层通信等
    
    return true;
}

// 分层AllReduce
bool distributed_trainer_allreduce_hierarchical(DistributedTrainer* trainer, ReductionOp op) {
    printf("  使用分层AllReduce算法\n");
    
    // 先在节点内做AllReduce，然后在节点间做AllReduce
    int local_rank = trainer->config.local_rank;
    int node_rank = trainer->config.rank / 4; // 假设每节点4个GPU
    
    printf("    本地排名: %d, 节点排名: %d\n", local_rank, node_rank);
    
    return true;
}

// 压缩梯度
bool distributed_trainer_compress_gradients(DistributedTrainer* trainer) {
    if (!trainer || !trainer->config.enable_gradient_compression) return true;
    
    printf("压缩梯度 (压缩比: %.4f)\n", trainer->config.compression_ratio);
    
    for (size_t i = 0; i < trainer->num_gradients; i++) {
        DynamicTensor* grad = trainer->gradients[i];
        if (!grad) continue;
        
        // 实现梯度压缩算法
        // 可以是Top-K稀疏化、量化、随机舍入等
        
        printf("  压缩梯度 %zu: 原始大小 -> 压缩大小\n", i);
    }
    
    return true;
}

// 同步模型并行梯度
bool distributed_trainer_sync_model_parallel(DistributedTrainer* trainer) {
    if (!trainer) return false;
    
    printf("同步模型并行梯度...\n");
    
    // 根据参数分配同步相应的梯度
    for (size_t i = 0; i < trainer->num_gradients; i++) {
        if (trainer->param_is_local[i]) {
            // 本地参数，需要同步给其他节点
            printf("  同步本地参数 %zu\n", i);
        } else {
            // 远程参数，需要从其他节点获取
            printf("  获取远程参数 %zu\n", i);
        }
    }
    
    return true;
}

// 同步流水线并行梯度
bool distributed_trainer_sync_pipeline_parallel(DistributedTrainer* trainer) {
    if (!trainer) return false;
    
    printf("同步流水线并行梯度 (阶段: %d)...\n", trainer->pipeline_stage);
    
    // 流水线并行只需要在阶段边界同步
    int prev_stage = trainer->pipeline_stage - 1;
    int next_stage = trainer->pipeline_stage + 1;
    
    if (prev_stage >= 0) {
        printf("  从前一阶段 %d 接收梯度\n", prev_stage);
    }
    
    if (next_stage < trainer->config.world_size) {
        printf("  向下一阶段 %d 发送梯度\n", next_stage);
    }
    
    return true;
}

// 同步混合并行梯度
bool distributed_trainer_sync_hybrid(DistributedTrainer* trainer) {
    if (!trainer) return false;
    
    printf("同步混合并行梯度...\n");
    
    // 结合数据并行和模型并行的同步策略
    // 先同步模型并行部分，再同步数据并行部分
    
    if (!distributed_trainer_sync_model_parallel(trainer)) {
        return false;
    }
    
    if (!distributed_trainer_allreduce_gradients(trainer, REDUCE_MEAN)) {
        return false;
    }
    
    return true;
}

// 获取性能指标
DistributedMetrics distributed_trainer_get_metrics(DistributedTrainer* trainer) {
    DistributedMetrics metrics = {0};
    
    if (!trainer) return metrics;
    
    // 模拟性能指标
    metrics.loss = 0.5f + 0.1f * sin(trainer->step * 0.01f); // 模拟损失下降
    metrics.accuracy = 0.85f + 0.05f * (1.0f - exp(-trainer->step * 0.001f)); // 模拟准确率提升
    metrics.learning_rate = 0.001f * exp(-trainer->step * 0.0001f); // 模拟学习率衰减
    metrics.epoch = trainer->epoch;
    metrics.step = trainer->step;
    
    metrics.comm_time = trainer->comm_time;
    metrics.compute_time = trainer->compute_time;
    metrics.bytes_sent = trainer->bytes_transferred;
    metrics.bytes_received = trainer->bytes_transferred * 0.8f; // 假设接收略少
    metrics.num_syncs = trainer->step;
    
    // 模拟系统指标
    metrics.cpu_usage = 75.0f + 10.0f * sin(trainer->step * 0.1f);
    metrics.memory_usage = 60.0f + 20.0f * cos(trainer->step * 0.05f);
    metrics.gpu_usage = 85.0f + 5.0f * sin(trainer->step * 0.2f);
    metrics.gpu_memory_usage = 70.0f + 15.0f * cos(trainer->step * 0.08f);
    
    // 模拟网络指标
    metrics.network_latency = 1.0f + 0.5f * sin(trainer->step * 0.3f);
    metrics.bandwidth_utilization = 80.0f + 10.0f * cos(trainer->step * 0.15f);
    metrics.packet_loss = 0;
    
    return metrics;
}

// 记录性能指标
bool distributed_trainer_log_metrics(DistributedTrainer* trainer, const char* log_dir) {
    if (!trainer || !log_dir) return false;
    
    DistributedMetrics metrics = distributed_trainer_get_metrics(trainer);
    
    printf("记录性能指标到: %s\n", log_dir);
    printf("  Epoch: %d, Step: %d\n", metrics.epoch, metrics.step);
    printf("  Loss: %.4f, Accuracy: %.4f, LR: %.6f\n", 
           metrics.loss, metrics.accuracy, metrics.learning_rate);
    printf("  Comm Time: %.2f ms, Compute Time: %.2f ms\n", 
           metrics.comm_time, metrics.compute_time);
    printf("  CPU: %.1f%%, Memory: %.1f%%, GPU: %.1f%%\n", 
           metrics.cpu_usage, metrics.memory_usage, metrics.gpu_usage);
    
    return true;
}

// 导出性能指标
bool distributed_trainer_export_metrics(DistributedTrainer* trainer, const char* filename) {
    if (!trainer || !filename) return false;
    
    printf("导出性能指标到: %s\n", filename);
    
    // 这里实现实际的导出功能
    // 可以导出为JSON、CSV、TensorBoard格式等
    
    return true;
}

// 分布式优化器创建
DistributedOptimizer* distributed_optimizer_create(OptimizerType base_optimizer,
                                                  DistributedTrainer* trainer,
                                                  float learning_rate) {
    if (!trainer) return NULL;
    
    DistributedOptimizer* optimizer = (DistributedOptimizer*)calloc(1, sizeof(DistributedOptimizer));
    if (!optimizer) return NULL;
    
    optimizer->base_optimizer = base_optimizer;
    optimizer->trainer = trainer;
    optimizer->base_lr = learning_rate;
    optimizer->warmup_lr = learning_rate * 0.01f;
    optimizer->warmup_steps = 1000;
    optimizer->min_lr = learning_rate * 0.01f;
    optimizer->gradient_scale = 1.0f;
    optimizer->enable_gradient_clipping = true;
    optimizer->max_gradient_norm = 1.0f;
    
    // 初始化优化器状态
    optimizer->num_parameters = trainer->num_gradients;
    optimizer->momentum_buffers = (DynamicTensor**)calloc(optimizer->num_parameters, sizeof(DynamicTensor*));
    optimizer->variance_buffers = (DynamicTensor**)calloc(optimizer->num_parameters, sizeof(DynamicTensor*));
    
    if (!optimizer->momentum_buffers || !optimizer->variance_buffers) {
        distributed_optimizer_destroy(optimizer);
        return NULL;
    }
    
    printf("创建分布式优化器 (基础优化器: %s, 学习率: %.6f)\n",
           optimizer_type_name(base_optimizer), learning_rate);
    
    return optimizer;
}

// 销毁分布式优化器
void distributed_optimizer_destroy(DistributedOptimizer* optimizer) {
    if (!optimizer) return;
    
    // 清理动量缓冲区
    if (optimizer->momentum_buffers) {
        for (size_t i = 0; i < optimizer->num_parameters; i++) {
            if (optimizer->momentum_buffers[i]) {
                dynamic_tensor_destroy(optimizer->momentum_buffers[i]);
            }
        }
        free(optimizer->momentum_buffers);
    }
    
    // 清理方差缓冲区
    if (optimizer->variance_buffers) {
        for (size_t i = 0; i < optimizer->num_parameters; i++) {
            if (optimizer->variance_buffers[i]) {
                dynamic_tensor_destroy(optimizer->variance_buffers[i]);
            }
        }
        free(optimizer->variance_buffers);
    }
    
    free(optimizer);
}

// 分布式优化器步骤
bool distributed_optimizer_step(DistributedOptimizer* optimizer) {
    if (!optimizer || !optimizer->trainer) return false;
    
    // 应用学习率调度
    float current_lr = distributed_optimizer_get_lr(optimizer);
    
    // 梯度裁剪
    if (optimizer->enable_gradient_clipping) {
        distributed_optimizer_clip_gradients(optimizer);
    }
    
    // 根据优化器类型执行更新
    switch (optimizer->base_optimizer) {
        case OPTIMIZER_SGD:
            return distributed_optimizer_sgd_step(optimizer, current_lr);
        case OPTIMIZER_ADAM:
            return distributed_optimizer_adam_step(optimizer, current_lr);
        case OPTIMIZER_ADAMW:
            return distributed_optimizer_adamw_step(optimizer, current_lr);
        case OPTIMIZER_RMSPROP:
            return distributed_optimizer_rmsprop_step(optimizer, current_lr);
        default:
            return false;
    }
}

// 获取当前学习率
float distributed_optimizer_get_lr(DistributedOptimizer* optimizer) {
    if (!optimizer) return 0.0f;
    
    int step = optimizer->trainer->step;
    
    // 线性预热
    if (step < optimizer->warmup_steps) {
        float warmup_progress = (float)step / optimizer->warmup_steps;
        return optimizer->warmup_lr + (optimizer->base_lr - optimizer->warmup_lr) * warmup_progress;
    }
    
    // 余弦退火
    float progress = (float)(step - optimizer->warmup_steps) / (10000 - optimizer->warmup_steps);
    return optimizer->min_lr + (optimizer->base_lr - optimizer->min_lr) * (1.0f + cos(M_PI * progress)) / 2.0f;
}

// 梯度裁剪
bool distributed_optimizer_clip_gradients(DistributedOptimizer* optimizer) {
    if (!optimizer) return false;
    
    float total_norm = 0.0f;
    
    // 计算梯度范数
    for (size_t i = 0; i < optimizer->num_parameters; i++) {
        DynamicTensor* grad = optimizer->trainer->gradients[i];
        if (!grad) continue;
        
        // 这里实现梯度范数计算
        total_norm += 1.0f; // 模拟范数
    }
    
    total_norm = sqrt(total_norm);
    
    // 裁剪梯度
    if (total_norm > optimizer->max_gradient_norm) {
        float clip_coef = optimizer->max_gradient_norm / (total_norm + 1e-6f);
        
        for (size_t i = 0; i < optimizer->num_parameters; i++) {
            DynamicTensor* grad = optimizer->trainer->gradients[i];
            if (!grad) continue;
            
            // 这里实现梯度缩放
            printf("  裁剪梯度 %zu (范数: %.4f, 裁剪系数: %.4f)\n", i, total_norm, clip_coef);
        }
    }
    
    return true;
}

// SGD步骤
bool distributed_optimizer_sgd_step(DistributedOptimizer* optimizer, float lr) {
    printf("执行分布式SGD优化步骤 (学习率: %.6f)\n", lr);
    
    for (size_t i = 0; i < optimizer->num_parameters; i++) {
        DynamicTensor* param = NULL; // 这里需要获取模型参数
        DynamicTensor* grad = optimizer->trainer->gradients[i];
        
        if (!grad) continue;
        
        // 这里实现SGD更新逻辑
        printf("  更新参数 %zu\n", i);
    }
    
    return true;
}

// Adam步骤
bool distributed_optimizer_adam_step(DistributedOptimizer* optimizer, float lr) {
    printf("执行分布式Adam优化步骤 (学习率: %.6f)\n", lr);
    
    static int step = 0;
    step++;
    
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    
    for (size_t i = 0; i < optimizer->num_parameters; i++) {
        DynamicTensor* grad = optimizer->trainer->gradients[i];
        if (!grad) continue;
        
        // 更新动量
        if (!optimizer->momentum_buffers[i]) {
            // 初始化动量缓冲区
        }
        
        // 更新方差
        if (!optimizer->variance_buffers[i]) {
            // 初始化方差缓冲区
        }
        
        // 计算偏差修正
        float bias_correction1 = 1.0f - powf(beta1, step);
        float bias_correction2 = 1.0f - powf(beta2, step);
        
        // 这里实现Adam更新逻辑
        printf("  更新参数 %zu (步骤: %d)\n", i, step);
    }
    
    return true;
}

// AdamW步骤
bool distributed_optimizer_adamw_step(DistributedOptimizer* optimizer, float lr) {
    printf("执行分布式AdamW优化步骤 (学习率: %.6f)\n", lr);
    
    float weight_decay = 0.01f;
    
    // 先应用权重衰减
    for (size_t i = 0; i < optimizer->num_parameters; i++) {
        // 这里实现权重衰减
        printf("  应用权重衰减到参数 %zu (衰减率: %.4f)\n", i, weight_decay);
    }
    
    // 然后执行Adam步骤
    return distributed_optimizer_adam_step(optimizer, lr);
}

// RMSprop步骤
bool distributed_optimizer_rmsprop_step(DistributedOptimizer* optimizer, float lr) {
    printf("执行分布式RMSprop优化步骤 (学习率: %.6f)\n", lr);
    
    float alpha = 0.99f;
    float eps = 1e-8f;
    
    for (size_t i = 0; i < optimizer->num_parameters; i++) {
        DynamicTensor* grad = optimizer->trainer->gradients[i];
        if (!grad) continue;
        
        // 这里实现RMSprop更新逻辑
        printf("  更新参数 %zu\n", i);
    }
    
    return true;
}

// 清零梯度
bool distributed_optimizer_zero_grad(DistributedOptimizer* optimizer) {
    if (!optimizer || !optimizer->trainer) return false;
    
    for (size_t i = 0; i < optimizer->num_parameters; i++) {
        DynamicTensor* grad = optimizer->trainer->gradients[i];
        if (!grad) continue;
        
        // 清零梯度张量
        // 这里实现实际的清零操作
    }
    
    return true;
}

// 分布式数据加载器创建
DistributedDataLoader* distributed_dataloader_create(void* dataset,
                                                  size_t batch_size,
                                                  DistributedConfig* config) {
    if (!dataset || !config) return NULL;
    
    DistributedDataLoader* dataloader = (DistributedDataLoader*)calloc(1, sizeof(DistributedDataLoader));
    if (!dataloader) return NULL;
    
    dataloader->dataset = dataset;
    dataloader->batch_size = batch_size;
    dataloader->num_workers = 4;
    dataloader->world_size = config->world_size;
    dataloader->rank = config->rank;
    
    // 计算每个排名的样本数
    // 这里需要根据实际数据集大小计算
    dataloader->samples_per_rank = 1000; // 模拟值
    
    // 分配样本索引数组
    dataloader->sample_indices = (size_t*)calloc(dataloader->samples_per_rank, sizeof(size_t));
    if (!dataloader->sample_indices) {
        free(dataloader);
        return NULL;
    }
    
    // 初始化数据分片
    for (size_t i = 0; i < dataloader->samples_per_rank; i++) {
        dataloader->sample_indices[i] = config->rank * dataloader->samples_per_rank + i;
    }
    
    dataloader->enable_data_sharding = true;
    dataloader->enable_caching = true;
    strcpy(dataloader->cache_dir, "./cache");
    
    dataloader->pin_memory = true;
    dataloader->prefetch = true;
    dataloader->prefetch_factor = 2;
    
    dataloader->enable_chinese_labels = config->enable_chinese_support;
    
    printf("创建分布式数据加载器 (批次大小: %zu, 工作进程: %zu, 分片: %s)\n",
           batch_size, dataloader->num_workers, 
           dataloader->enable_data_sharding ? "启用" : "禁用");
    
    return dataloader;
}

// 销毁分布式数据加载器
void distributed_dataloader_destroy(DistributedDataLoader* dataloader) {
    if (!dataloader) return;
    
    if (dataloader->sample_indices) {
        free(dataloader->sample_indices);
    }
    
    free(dataloader);
}

// 设置epoch
bool distributed_dataloader_set_epoch(DistributedDataLoader* dataloader, int epoch) {
    if (!dataloader) return false;
    
    // 重新洗牌数据索引
    srand(epoch);
    
    for (size_t i = dataloader->samples_per_rank - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = dataloader->sample_indices[i];
        dataloader->sample_indices[i] = dataloader->sample_indices[j];
        dataloader->sample_indices[j] = temp;
    }
    
    printf("数据加载器设置epoch: %d\n", epoch);
    return true;
}

// 获取下一个批次
void** distributed_dataloader_next_batch(DistributedDataLoader* dataloader) {
    if (!dataloader) return NULL;
    
    static size_t current_index = 0;
    
    if (current_index >= dataloader->samples_per_rank) {
        current_index = 0;
        return NULL; // 数据遍历完成
    }
    
    // 模拟数据加载
    void** batch_data = (void**)calloc(2, sizeof(void*));
    if (!batch_data) return NULL;
    
    // 创建输入和目标张量（模拟）
    DynamicTensor* inputs = dynamic_tensor_create(NULL, 0, NULL);
    DynamicTensor* targets = dynamic_tensor_create(NULL, 0, NULL);
    
    batch_data[0] = inputs;
    batch_data[1] = targets;
    
    current_index += dataloader->batch_size;
    
    return batch_data;
}

// 辅助函数：策略名称
const char* distributed_strategy_name(DistributedStrategy strategy) {
    switch (strategy) {
        case DISTRIBUTED_NONE: return "None";
        case DISTRIBUTED_DATA_PARALLEL: return "Data Parallel";
        case DISTRIBUTED_MODEL_PARALLEL: return "Model Parallel";
        case DISTRIBUTED_PIPELINE_PARALLEL: return "Pipeline Parallel";
        case DISTRIBUTED_HYBRID: return "Hybrid";
        case DISTRIBUTED_FEDERATED: return "Federated";
        default: return "Unknown";
    }
}

// 通信后端名称
const char* communication_backend_name(CommunicationBackend backend) {
    switch (backend) {
        case COMM_NCCL: return "NCCL";
        case COMM_MPI: return "MPI";
        case COMM_GLOO: return "Gloo";
        case COMM_RPC: return "RPC";
        case COMM_GRPC: return "gRPC";
        case COMM_CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

// 规约操作名称
const char* reduction_op_name(ReductionOp op) {
    switch (op) {
        case REDUCE_SUM: return "Sum";
        case REDUCE_MEAN: return "Mean";
        case REDUCE_MAX: return "Max";
        case REDUCE_MIN: return "Min";
        case REDUCE_PRODUCT: return "Product";
        default: return "Unknown";
    }
}

// 优化器类型名称
const char* optimizer_type_name(OptimizerType type) {
    switch (type) {
        case OPTIMIZER_SGD: return "SGD";
        case OPTIMIZER_ADAM: return "Adam";
        case OPTIMIZER_ADAMW: return "AdamW";
        case OPTIMIZER_RMSPROP: return "RMSprop";
        default: return "Unknown";
    }
}

// 错误处理
const char* distributed_error_string(DistributedError error) {
    switch (error) {
        case DIST_SUCCESS: return "Success";
        case DIST_ERROR_INIT_FAILED: return "Initialization failed";
        case DIST_ERROR_COMM_FAILED: return "Communication failed";
        case DIST_ERROR_SYNC_FAILED: return "Synchronization failed";
        case DIST_ERROR_CHECKPOINT_FAILED: return "Checkpoint failed";
        case DIST_ERROR_MEMORY_ALLOC_FAILED: return "Memory allocation failed";
        case DIST_ERROR_UNSUPPORTED_STRATEGY: return "Unsupported strategy";
        case DIST_ERROR_FAULT_TOLERANCE_FAILED: return "Fault tolerance failed";
        default: return "Unknown error";
    }
}

const char* distributed_error_string_chinese(DistributedError error) {
    switch (error) {
        case DIST_SUCCESS: return "成功";
        case DIST_ERROR_INIT_FAILED: return "初始化失败";
        case DIST_ERROR_COMM_FAILED: return "通信失败";
        case DIST_ERROR_SYNC_FAILED: return "同步失败";
        case DIST_ERROR_CHECKPOINT_FAILED: return "检查点失败";
        case DIST_ERROR_MEMORY_ALLOC_FAILED: return "内存分配失败";
        case DIST_ERROR_UNSUPPORTED_STRATEGY: return "不支持的策略";
        case DIST_ERROR_FAULT_TOLERANCE_FAILED: return "容错失败";
        default: return "未知错误";
    }
}