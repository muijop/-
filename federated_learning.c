#include "federated_learning.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

// ==================== 内部工具函数 ====================

static double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static float calculate_weighted_average(float* values, int* weights, int count) {
    if (count == 0) return 0.0f;
    
    float weighted_sum = 0.0f;
    int total_weight = 0;
    
    for (int i = 0; i < count; i++) {
        weighted_sum += values[i] * weights[i];
        total_weight += weights[i];
    }
    
    return total_weight > 0 ? weighted_sum / total_weight : 0.0f;
}

static void add_gaussian_noise(tensor_t* tensor, float stddev) {
    if (!tensor) return;
    
    // 简单的均匀分布噪声（实际应该使用高斯分布）
    int num_elements = tensor->shape[0] * tensor->shape[1];
    for (int i = 0; i < num_elements; i++) {
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * stddev;
        tensor->data[i] += noise;
    }
}

// ==================== 服务器端实现 ====================

federated_server_t* federated_server_create(federated_config_t* config,
                                           nn_module_t* initial_model) {
    if (!config || !initial_model) {
        return NULL;
    }
    
    federated_server_t* server = malloc(sizeof(federated_server_t));
    if (!server) {
        return NULL;
    }
    
    memset(server, 0, sizeof(federated_server_t));
    
    // 复制配置
    memcpy(&server->config, config, sizeof(federated_config_t));
    
    // 设置全局模型
    server->global_model = initial_model;
    
    // 初始化客户端列表
    server->clients = malloc(sizeof(client_info_t*) * config->num_clients);
    if (!server->clients) {
        free(server);
        return NULL;
    }
    
    server->num_clients = 0;
    server->current_round = 0;
    server->is_running = false;
    
    // 初始化互斥锁
    if (pthread_mutex_init(&server->lock, NULL) != 0) {
        free(server->clients);
        free(server);
        return NULL;
    }
    
    // 设置默认路径
    server->model_checkpoint_path = strdup("./checkpoints");
    server->log_file_path = strdup("./federated_log.txt");
    
    printf("联邦学习服务器创建成功\n");
    printf("算法: %d, 客户端数: %d, 轮数: %d\n", 
           config->algorithm, config->num_clients, config->rounds);
    
    return server;
}

void federated_server_destroy(federated_server_t* server) {
    if (!server) return;
    
    // 停止训练
    if (server->is_running) {
        federated_server_stop_training(server);
    }
    
    // 销毁客户端信息
    for (int i = 0; i < server->num_clients; i++) {
        client_info_t* client = server->clients[i];
        if (client) {
            free(client->client_address);
            free(client->data_distribution);
            free(client);
        }
    }
    
    free(server->clients);
    free(server->model_checkpoint_path);
    free(server->log_file_path);
    
    pthread_mutex_destroy(&server->lock);
    free(server);
    
    printf("联邦学习服务器已销毁\n");
}

int federated_server_register_client(federated_server_t* server,
                                    int client_id,
                                    const char* client_address) {
    if (!server || !client_address) {
        return -1;
    }
    
    pthread_mutex_lock(&server->lock);
    
    // 检查客户端是否已存在
    for (int i = 0; i < server->num_clients; i++) {
        if (server->clients[i]->client_id == client_id) {
            pthread_mutex_unlock(&server->lock);
            return -1; // 客户端已存在
        }
    }
    
    // 创建客户端信息
    client_info_t* client = malloc(sizeof(client_info_t));
    if (!client) {
        pthread_mutex_unlock(&server->lock);
        return -1;
    }
    
    memset(client, 0, sizeof(client_info_t));
    client->client_id = client_id;
    client->client_address = strdup(client_address);
    client->state = CLIENT_IDLE;
    client->is_active = true;
    client->last_contact = time(NULL);
    
    server->clients[server->num_clients++] = client;
    
    pthread_mutex_unlock(&server->lock);
    
    printf("客户端注册成功: ID=%d, 地址=%s\n", client_id, client_address);
    return 0;
}

static void* server_training_thread(void* arg) {
    federated_server_t* server = (federated_server_t*)arg;
    
    printf("联邦学习训练开始，总轮数: %d\n", server->config.rounds);
    
    for (server->current_round = 0; 
         server->current_round < server->config.rounds && server->is_running; 
         server->current_round++) {
        
        printf("=== 第 %d 轮训练开始 ===\n", server->current_round + 1);
        
        double round_start_time = get_current_time();
        
        // 1. 选择参与本轮训练的客户端
        int active_clients = 0;
        model_update_t** updates = malloc(sizeof(model_update_t*) * server->num_clients);
        
        if (!updates) {
            printf("内存分配失败\n");
            continue;
        }
        
        // 模拟接收客户端更新（实际应该通过网络通信）
        for (int i = 0; i < server->num_clients && server->is_running; i++) {
            if (server->clients[i]->is_active) {
                // 模拟客户端训练和上传
                model_update_t* update = malloc(sizeof(model_update_t));
                if (update) {
                    memset(update, 0, sizeof(model_update_t));
                    update->client_id = server->clients[i]->client_id;
                    update->round = server->current_round;
                    update->loss = 0.1f + (float)rand() / RAND_MAX * 0.1f; // 模拟损失
                    update->accuracy = 0.8f + (float)rand() / RAND_MAX * 0.1f; // 模拟准确率
                    update->data_size = 1000 + rand() % 1000; // 模拟数据量
                    update->timestamp = time(NULL);
                    
                    updates[active_clients++] = update;
                    
                    printf("接收到客户端 %d 的更新: 损失=%.4f, 准确率=%.4f\n", 
                           update->client_id, update->loss, update->accuracy);
                }
            }
        }
        
        if (active_clients > 0) {
            // 2. 聚合客户端更新
            federated_server_aggregate_updates(server, updates, active_clients);
            
            // 3. 更新全局模型（这里简化处理，实际需要根据梯度更新）
            server->global_loss = calculate_weighted_average(
                (float[]){updates[0]->loss, updates[active_clients-1]->loss}, 
                (int[]){updates[0]->data_size, updates[active_clients-1]->data_size}, 
                2
            );
            
            server->global_accuracy = calculate_weighted_average(
                (float[]){updates[0]->accuracy, updates[active_clients-1]->accuracy}, 
                (int[]){updates[0]->data_size, updates[active_clients-1]->data_size}, 
                2
            );
            
            printf("第 %d 轮聚合完成: 全局损失=%.4f, 全局准确率=%.4f\n", 
                   server->current_round + 1, server->global_loss, server->global_accuracy);
            
            // 4. 保存检查点
            if ((server->current_round + 1) % 10 == 0) {
                federated_server_save_checkpoint(server, server->current_round);
            }
        }
        
        // 清理更新数据
        for (int i = 0; i < active_clients; i++) {
            free(updates[i]);
        }
        free(updates);
        
        double round_time = get_current_time() - round_start_time;
        printf("第 %d 轮训练完成，耗时: %.2f秒\n", 
               server->current_round + 1, round_time);
        
        // 等待下一轮（模拟网络延迟）
        if (server->is_running) {
            sleep(2); // 2秒间隔
        }
    }
    
    printf("联邦学习训练完成\n");
    return NULL;
}

int federated_server_start_training(federated_server_t* server) {
    if (!server || server->is_running) {
        return -1;
    }
    
    if (server->num_clients == 0) {
        printf("错误: 没有注册的客户端\n");
        return -1;
    }
    
    server->is_running = true;
    
    // 创建训练线程
    pthread_t training_thread;
    if (pthread_create(&training_thread, NULL, server_training_thread, server) != 0) {
        server->is_running = false;
        return -1;
    }
    
    printf("联邦学习训练已启动\n");
    return 0;
}

int federated_server_stop_training(federated_server_t* server) {
    if (!server || !server->is_running) {
        return -1;
    }
    
    server->is_running = false;
    printf("联邦学习训练正在停止...\n");
    
    // 等待训练线程结束（实际应该更优雅地停止）
    sleep(3);
    
    printf("联邦学习训练已停止\n");
    return 0;
}

int federated_server_aggregate_updates(federated_server_t* server,
                                      model_update_t** updates,
                                      int num_updates) {
    if (!server || !updates || num_updates == 0) {
        return -1;
    }
    
    // 根据算法选择聚合方法
    switch (server->config.algorithm) {
        case FED_AVG:
            return federated_average_aggregate(server->global_model, updates, 
                                              num_updates, &server->config);
        case FED_PROX:
            return fedprox_aggregate(server->global_model, updates, 
                                   num_updates, &server->config);
        case FED_ADAM:
            return fedadam_aggregate(server->global_model, updates, 
                                   num_updates, &server->config);
        default:
            printf("不支持的聚合算法: %d\n", server->config.algorithm);
            return -1;
    }
}

// ==================== 聚合算法实现 ====================

int federated_average_aggregate(nn_module_t* global_model,
                               model_update_t** updates,
                               int num_updates,
                               federated_config_t* config) {
    if (!global_model || !updates || num_updates == 0) {
        return -1;
    }
    
    printf("使用联邦平均算法聚合 %d 个客户端更新\n", num_updates);
    
    // 计算总数据量
    int total_data_size = 0;
    for (int i = 0; i < num_updates; i++) {
        total_data_size += updates[i]->data_size;
    }
    
    if (total_data_size == 0) {
        return -1;
    }
    
    // 这里简化实现，实际需要根据梯度进行加权平均
    // 实际实现需要遍历所有模型参数，根据数据量进行加权平均
    
    printf("联邦平均聚合完成，总数据量: %d\n", total_data_size);
    return 0;
}

int fedprox_aggregate(nn_module_t* global_model,
                     model_update_t** updates,
                     int num_updates,
                     federated_config_t* config) {
    printf("使用FedProx算法聚合 %d 个客户端更新\n", num_updates);
    printf("Proximal参数 mu=%.4f\n", config->proximal_mu);
    
    // FedProx实现（简化版）
    // 实际实现需要添加近端项来约束客户端更新
    
    return 0;
}

int fedadam_aggregate(nn_module_t* global_model,
                     model_update_t** updates,
                     int num_updates,
                     federated_config_t* config) {
    printf("使用FedAdam算法聚合 %d 个客户端更新\n", num_updates);
    printf("Adam参数: beta1=%.4f, beta2=%.4f, epsilon=%.6f\n", 
           config->beta1, config->beta2, config->epsilon);
    
    // FedAdam实现（简化版）
    // 实际实现需要使用Adam优化器来更新全局模型
    
    return 0;
}

// ==================== 检查点管理 ====================

int federated_server_save_checkpoint(federated_server_t* server, int round) {
    if (!server) {
        return -1;
    }
    
    char checkpoint_path[256];
    snprintf(checkpoint_path, sizeof(checkpoint_path), 
             "%s/round_%d.checkpoint", server->model_checkpoint_path, round);
    
    printf("保存检查点到: %s\n", checkpoint_path);
    
    // 实际实现需要序列化模型状态和训练信息
    // 这里简化实现
    
    return 0;
}

int federated_server_load_checkpoint(federated_server_t* server,
                                    const char* checkpoint_path) {
    if (!server || !checkpoint_path) {
        return -1;
    }
    
    printf("从检查点加载: %s\n", checkpoint_path);
    
    // 实际实现需要反序列化模型状态和训练信息
    // 这里简化实现
    
    return 0;
}

// ==================== 客户端端实现（简化版） ====================

federated_client_t* federated_client_create(federated_config_t* config,
                                           int client_id,
                                           const char* server_address,
                                           const char* data_path) {
    if (!config || !server_address) {
        return NULL;
    }
    
    federated_client_t* client = malloc(sizeof(federated_client_t));
    if (!client) {
        return NULL;
    }
    
    memset(client, 0, sizeof(federated_client_t));
    
    memcpy(&client->config, config, sizeof(federated_config_t));
    client->client_id = client_id;
    client->server_address = strdup(server_address);
    client->data_path = data_path ? strdup(data_path) : NULL;
    client->is_training = false;
    
    printf("联邦学习客户端创建成功: ID=%d, 服务器=%s\n", 
           client_id, server_address);
    
    return client;
}

void federated_client_destroy(federated_client_t* client) {
    if (!client) return;
    
    if (client->is_training) {
        // 停止训练
        client->is_training = false;
    }
    
    free(client->server_address);
    free(client->data_path);
    free(client);
    
    printf("联邦学习客户端已销毁\n");
}

// ==================== 差分隐私实现 ====================

int add_differential_privacy_noise(tensor_t* gradient,
                                  float epsilon,
                                  float delta,
                                  float sensitivity) {
    if (!gradient) {
        return -1;
    }
    
    // 计算噪声标准差
    float sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon;
    
    printf("添加差分隐私噪声: epsilon=%.2f, delta=%.6f, sigma=%.6f\n", 
           epsilon, delta, sigma);
    
    // 添加高斯噪声
    add_gaussian_noise(gradient, sigma);
    
    return 0;
}

// ==================== 工具函数实现 ====================

float calculate_model_difference(nn_module_t* model1, nn_module_t* model2) {
    if (!model1 || !model2) {
        return -1.0f;
    }
    
    // 简化实现，实际需要计算参数差异
    return 0.0f;
}

bool validate_model_integrity(nn_module_t* model) {
    if (!model) {
        return false;
    }
    
    // 简化实现，实际需要验证模型结构完整性
    return true;
}

// 其他函数实现...