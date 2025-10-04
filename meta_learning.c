#include "meta_learning.h"
#include "nn_module.h"
#include "ai_trainer.h"
#include "tensor.h"
#include "autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// ===========================================
// 内部工具函数
// ===========================================

// 生成随机数（0-1）
static float random_float(void) {
    return (float)rand() / RAND_MAX;
}

// 获取当前时间（秒）
static double get_current_time(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

// 计算向量的均值
static float vector_mean(const float* data, int size) {
    if (!data || size <= 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    
    return sum / size;
}

// 计算准确率
static float calculate_accuracy(const tensor_t* predictions, const tensor_t* labels) {
    if (!predictions || !labels || predictions->shape[0] != labels->shape[0]) {
        return 0.0f;
    }
    
    int num_samples = predictions->shape[0];
    int num_classes = predictions->shape[1];
    int correct = 0;
    
    for (int i = 0; i < num_samples; i++) {
        int pred_class = 0;
        float max_prob = predictions->data[i * num_classes];
        
        for (int j = 1; j < num_classes; j++) {
            if (predictions->data[i * num_classes + j] > max_prob) {
                max_prob = predictions->data[i * num_classes + j];
                pred_class = j;
            }
        }
        
        int true_class = (int)labels->data[i];
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return (float)correct / num_samples;
}

// ===========================================
// 元学习管理器实现
// ===========================================

meta_learning_manager_t* create_meta_learning_manager(void) {
    meta_learning_manager_t* manager = 
        (meta_learning_manager_t*)calloc(1, sizeof(meta_learning_manager_t));
    
    if (!manager) {
        return NULL;
    }
    
    // 设置默认配置
    manager->config.method = META_LEARNING_MAML;
    manager->config.num_tasks = 10;
    manager->config.inner_loop_steps = 5;
    manager->config.inner_learning_rate = 0.01f;
    manager->config.outer_learning_rate = 0.001f;
    manager->config.batch_size = 32;
    manager->config.adaptation_steps = 10;
    
    manager->config.maml_first_order = false;
    manager->config.maml_use_batch_norm = true;
    
    manager->config.reptile_step_size = 0.1f;
    manager->config.reptile_inner_batches = 10;
    
    manager->config.protonet_use_euclidean = true;
    manager->config.protonet_distance_scale = 1.0f;
    
    manager->config.use_second_order = true;
    manager->config.normalize_gradients = true;
    manager->config.gradient_clip_value = 1.0f;
    
    manager->config.max_epochs = 100;
    manager->config.early_stopping_patience = 10.0f;
    manager->config.use_validation = true;
    
    manager->config.max_time_seconds = 3600;
    manager->config.max_memory_mb = 4096;
    
    // 初始化任务数组
    manager->tasks = NULL;
    manager->num_tasks = 0;
    manager->max_tasks = 0;
    
    manager->is_initialized = true;
    manager->is_training = false;
    manager->is_adapted = false;
    manager->current_epoch = 0;
    manager->start_time = 0.0f;
    manager->result = NULL;
    
    // 初始化随机种子
    srand((unsigned int)time(NULL));
    
    return manager;
}

void destroy_meta_learning_manager(meta_learning_manager_t* manager) {
    if (manager) {
        // 释放任务数据
        if (manager->tasks) {
            for (int i = 0; i < manager->num_tasks; i++) {
                free(manager->tasks[i].task_name);
                // 注意：张量需要外部管理
            }
            free(manager->tasks);
        }
        
        // 释放结果
        if (manager->result) {
            if (manager->result->adaptation_accuracy) {
                free(manager->result->adaptation_accuracy);
            }
            if (manager->result->task_performance) {
                free(manager->result->task_performance);
            }
            // 注意：模型需要外部管理
            free(manager->result);
        }
        
        free(manager);
    }
}

// ===========================================
// 配置设置
// ===========================================

int set_meta_learning_method(meta_learning_manager_t* manager, meta_learning_type_t method) {
    if (!manager) {
        return -1;
    }
    
    manager->config.method = method;
    return 0;
}

int configure_meta_learning(meta_learning_manager_t* manager, const meta_learning_config_t* config) {
    if (!manager || !config) {
        return -1;
    }
    
    manager->config = *config;
    return 0;
}

// ===========================================
// 任务管理
// ===========================================

int add_meta_learning_task(meta_learning_manager_t* manager, const meta_learning_task_t* task) {
    if (!manager || !task) {
        return -1;
    }
    
    // 扩展任务数组
    if (manager->num_tasks >= manager->max_tasks) {
        int new_max = manager->max_tasks == 0 ? 10 : manager->max_tasks * 2;
        meta_learning_task_t* new_tasks = 
            (meta_learning_task_t*)realloc(manager->tasks, new_max * sizeof(meta_learning_task_t));
        if (!new_tasks) {
            return -1;
        }
        manager->tasks = new_tasks;
        manager->max_tasks = new_max;
    }
    
    // 添加新任务
    meta_learning_task_t* new_task = &manager->tasks[manager->num_tasks];
    
    new_task->task_name = strdup(task->task_name);
    new_task->support_set = task->support_set;
    new_task->support_labels = task->support_labels;
    new_task->query_set = task->query_set;
    new_task->query_labels = task->query_labels;
    new_task->num_classes = task->num_classes;
    new_task->k_shot = task->k_shot;
    new_task->n_way = task->n_way;
    new_task->adaptation_steps = task->adaptation_steps;
    
    manager->num_tasks++;
    
    printf("添加元学习任务: %s, %d-way %d-shot\n", task->task_name, task->n_way, task->k_shot);
    
    return 0;
}

// ===========================================
// MAML算法实现
// ===========================================

// MAML内循环适应
static nn_module_t* maml_inner_loop(nn_module_t* model, const tensor_t* support_set, 
                                  const tensor_t* support_labels, int num_steps, 
                                  float learning_rate, bool first_order) {
    if (!model || !support_set || !support_labels) {
        return NULL;
    }
    
    // 克隆模型用于内循环更新
    nn_module_t* adapted_model = nn_module_clone(model);
    if (!adapted_model) {
        return NULL;
    }
    
    // 内循环梯度更新
    for (int step = 0; step < num_steps; step++) {
        // 前向传播
        tensor_t* predictions = nn_module_forward(adapted_model, support_set);
        if (!predictions) {
            nn_module_free(adapted_model);
            return NULL;
        }
        
        // 计算损失（交叉熵）
        float loss = 0.0f;
        int num_samples = support_set->shape[0];
        int num_classes = predictions->shape[1];
        
        for (int i = 0; i < num_samples; i++) {
            int true_class = (int)support_labels->data[i];
            if (true_class >= 0 && true_class < num_classes) {
                float prob = predictions->data[i * num_classes + true_class];
                if (prob > 0.0f) {
                    loss -= logf(prob);
                }
            }
        }
        loss /= num_samples;
        
        // 反向传播（简化实现）
        // 实际MAML需要计算二阶导数，这里使用简化的一阶近似
        
        // 计算梯度（简化：使用数值梯度）
        // 实际实现应该使用自动微分系统
        
        // 更新模型参数
        // 简化实现：随机更新演示
        for (int i = 0; i < adapted_model->num_layers; i++) {
            nn_layer_t* layer = adapted_model->layers[i];
            if (layer->weights) {
                for (int j = 0; j < layer->weights->shape[0] * layer->weights->shape[1]; j++) {
                    // 简化梯度更新
                    layer->weights->data[j] -= learning_rate * (random_float() - 0.5f) * 0.1f;
                }
            }
            if (layer->bias) {
                for (int j = 0; j < layer->bias->shape[0]; j++) {
                    layer->bias->data[j] -= learning_rate * (random_float() - 0.5f) * 0.1f;
                }
            }
        }
        
        tensor_free(predictions);
        
        if (step % 10 == 0) {
            printf("MAML内循环步 %d，损失: %.4f\n", step, loss);
        }
    }
    
    return adapted_model;
}

// MAML外循环元更新
static int maml_outer_update(nn_module_t* model, const meta_learning_task_t* task, 
                           int inner_steps, float inner_lr, float outer_lr, 
                           bool first_order) {
    if (!model || !task) {
        return -1;
    }
    
    printf("执行MAML外循环更新\n");
    
    // 内循环适应
    nn_module_t* adapted_model = maml_inner_loop(model, task->support_set, 
                                                task->support_labels, inner_steps, 
                                                inner_lr, first_order);
    if (!adapted_model) {
        return -1;
    }
    
    // 在查询集上评估适应后的模型
    tensor_t* query_predictions = nn_module_forward(adapted_model, task->query_set);
    if (!query_predictions) {
        nn_module_free(adapted_model);
        return -1;
    }
    
    // 计算查询损失
    float query_loss = 0.0f;
    int num_query_samples = task->query_set->shape[0];
    int num_classes = query_predictions->shape[1];
    
    for (int i = 0; i < num_query_samples; i++) {
        int true_class = (int)task->query_labels->data[i];
        if (true_class >= 0 && true_class < num_classes) {
            float prob = query_predictions->data[i * num_classes + true_class];
            if (prob > 0.0f) {
                query_loss -= logf(prob);
            }
        }
    }
    query_loss /= num_query_samples;
    
    // 计算查询准确率
    float query_accuracy = calculate_accuracy(query_predictions, task->query_labels);
    
    printf("MAML查询损失: %.4f，准确率: %.4f\n", query_loss, query_accuracy);
    
    // 外循环梯度更新（简化实现）
    // 实际MAML需要计算相对于初始参数的二阶梯度
    
    // 简化：使用查询损失来更新原始模型
    // 实际实现应该使用自动微分计算高阶梯度
    
    tensor_free(query_predictions);
    nn_module_free(adapted_model);
    
    return 0;
}

// ===========================================
// Reptile算法实现
// ===========================================

static int reptile_update(nn_module_t* model, const meta_learning_task_t* task, 
                        int inner_batches, float inner_lr, float step_size) {
    if (!model || !task) {
        return -1;
    }
    
    printf("执行Reptile更新\n");
    
    // 保存初始参数
    tensor_t** initial_weights = (tensor_t**)malloc(model->num_layers * sizeof(tensor_t*));
    tensor_t** initial_biases = (tensor_t**)malloc(model->num_layers * sizeof(tensor_t*));
    
    if (!initial_weights || !initial_biases) {
        if (initial_weights) free(initial_weights);
        if (initial_biases) free(initial_biases);
        return -1;
    }
    
    // 复制初始参数
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights) {
            initial_weights[i] = tensor_copy(layer->weights);
        }
        if (layer->bias) {
            initial_biases[i] = tensor_copy(layer->bias);
        }
    }
    
    // 内循环适应（简化）
    for (int batch = 0; batch < inner_batches; batch++) {
        // 简化内循环：随机梯度更新
        for (int i = 0; i < model->num_layers; i++) {
            nn_layer_t* layer = model->layers[i];
            if (layer->weights) {
                for (int j = 0; j < layer->weights->shape[0] * layer->weights->shape[1]; j++) {
                    layer->weights->data[j] -= inner_lr * (random_float() - 0.5f) * 0.1f;
                }
            }
            if (layer->bias) {
                for (int j = 0; j < layer->bias->shape[0]; j++) {
                    layer->bias->data[j] -= inner_lr * (random_float() - 0.5f) * 0.1f;
                }
            }
        }
    }
    
    // Reptile更新：向初始参数方向移动
    for (int i = 0; i < model->num_layers; i++) {
        nn_layer_t* layer = model->layers[i];
        
        if (layer->weights && initial_weights[i]) {
            for (int j = 0; j < layer->weights->shape[0] * layer->weights->shape[1]; j++) {
                float delta = layer->weights->data[j] - initial_weights[i]->data[j];
                layer->weights->data[j] = initial_weights[i]->data[j] + step_size * delta;
            }
        }
        
        if (layer->bias && initial_biases[i]) {
            for (int j = 0; j < layer->bias->shape[0]; j++) {
                float delta = layer->bias->data[j] - initial_biases[i]->data[j];
                layer->bias->data[j] = initial_biases[i]->data[j] + step_size * delta;
            }
        }
    }
    
    // 清理临时张量
    for (int i = 0; i < model->num_layers; i++) {
        if (initial_weights[i]) tensor_free(initial_weights[i]);
        if (initial_biases[i]) tensor_free(initial_biases[i]);
    }
    free(initial_weights);
    free(initial_biases);
    
    return 0;
}

// ===========================================
// 元学习执行
// ===========================================

int start_meta_learning(meta_learning_manager_t* manager, nn_module_t* base_model) {
    if (!manager || !base_model || manager->num_tasks <= 0) {
        return -1;
    }
    
    if (manager->is_training) {
        printf("错误：元学习已在运行\n");
        return -1;
    }
    
    printf("开始元学习，方法: %d，任务数量: %d\n", manager->config.method, manager->num_tasks);
    
    manager->is_training = true;
    manager->current_epoch = 0;
    manager->start_time = (float)get_current_time();
    
    // 创建结果存储
    manager->result = (meta_learning_result_t*)calloc(1, sizeof(meta_learning_result_t));
    if (!manager->result) {
        manager->is_training = false;
        return -1;
    }
    
    manager->result->adaptation_accuracy = (float*)calloc(manager->config.adaptation_steps, sizeof(float));
    manager->result->task_performance = (float*)calloc(manager->num_tasks, sizeof(float));
    
    if (!manager->result->adaptation_accuracy || !manager->result->task_performance) {
        if (manager->result->adaptation_accuracy) free(manager->result->adaptation_accuracy);
        if (manager->result->task_performance) free(manager->result->task_performance);
        free(manager->result);
        manager->result = NULL;
        manager->is_training = false;
        return -1;
    }
    
    manager->result->num_adaptation_steps = manager->config.adaptation_steps;
    manager->result->num_tasks = manager->num_tasks;
    
    // 元学习训练循环
    for (int epoch = 0; epoch < manager->config.max_epochs; epoch++) {
        manager->current_epoch = epoch;
        
        printf("元学习轮次 %d\n", epoch);
        
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        
        // 遍历所有任务
        for (int task_idx = 0; task_idx < manager->num_tasks; task_idx++) {
            meta_learning_task_t* task = &manager->tasks[task_idx];
            
            // 根据方法执行元学习更新
            switch (manager->config.method) {
                case META_LEARNING_MAML:
                    maml_outer_update(base_model, task, manager->config.inner_loop_steps,
                                    manager->config.inner_learning_rate, 
                                    manager->config.outer_learning_rate,
                                    manager->config.maml_first_order);
                    break;
                    
                case META_LEARNING_REPTILE:
                    reptile_update(base_model, task, manager->config.reptile_inner_batches,
                                manager->config.inner_learning_rate,
                                manager->config.reptile_step_size);
                    break;
                    
                default:
                    printf("暂不支持的元学习方法: %d\n", manager->config.method);
                    break;
            }
            
            // 评估任务性能
            tensor_t* predictions = nn_module_forward(base_model, task->query_set);
            if (predictions) {
                float accuracy = calculate_accuracy(predictions, task->query_labels);
                epoch_accuracy += accuracy;
                manager->result->task_performance[task_idx] = accuracy;
                tensor_free(predictions);
            }
            
            // 调用进度回调
            if (manager->progress_callback) {
                int progress = (int)((epoch + 1) * 100 / manager->config.max_epochs);
                char message[100];
                snprintf(message, sizeof(message), "轮次 %d，任务 %d", epoch, task_idx);
                manager->progress_callback(progress, message);
            }
            
            // 调用任务完成回调
            if (manager->task_complete_callback) {
                manager->task_complete_callback(task_idx, manager->result->task_performance[task_idx]);
            }
        }
        
        epoch_accuracy /= manager->num_tasks;
        
        printf("轮次 %d 完成，平均准确率: %.4f\n", epoch, epoch_accuracy);
        
        // 检查早停条件
        if (epoch > 10 && epoch_accuracy < manager->result->meta_train_accuracy - 0.01f) {
            printf("早停触发，停止训练\n");
            break;
        }
        
        manager->result->meta_train_accuracy = epoch_accuracy;
        
        // 检查时间限制
        if (manager->config.max_time_seconds > 0) {
            double elapsed = get_current_time() - manager->start_time;
            if (elapsed > manager->config.max_time_seconds) {
                printf("达到时间限制，停止训练\n");
                break;
            }
        }
    }
    
    // 保存元学习后的模型
    manager->result->meta_model = base_model;
    manager->result->is_model_ready = true;
    manager->result->total_training_time = (float)(get_current_time() - manager->start_time);
    
    manager->is_training = false;
    manager->is_adapted = true;
    
    printf("元学习完成，总时间: %.2f秒\n", manager->result->total_training_time);
    
    // 调用元学习完成回调
    if (manager->meta_learning_complete_callback) {
        manager->meta_learning_complete_callback(manager->result);
    }
    
    return 0;
}

int stop_meta_learning(meta_learning_manager_t* manager) {
    if (!manager) {
        return -1;
    }
    
    if (manager->is_training) {
        manager->is_training = false;
        printf("元学习已停止\n");
        return 0;
    }
    
    return -1;
}

// ===========================================
// 快速适应
// ===========================================

int adapt_to_new_task(meta_learning_manager_t* manager, const meta_learning_task_t* new_task) {
    if (!manager || !new_task || !manager->result || !manager->result->is_model_ready) {
        return -1;
    }
    
    printf("开始快速适应新任务: %s\n", new_task->task_name);
    
    nn_module_t* model = manager->result->meta_model;
    
    // 快速适应过程
    for (int step = 0; step < new_task->adaptation_steps; step++) {
        // 使用支持集进行适应
        tensor_t* predictions = nn_module_forward(model, new_task->support_set);
        if (!predictions) {
            return -1;
        }
        
        // 计算损失
        float loss = 0.0f;
        int num_samples = new_task->support_set->shape[0];
        int num_classes = predictions->shape[1];
        
        for (int i = 0; i < num_samples; i++) {
            int true_class = (int)new_task->support_labels->data[i];
            if (true_class >= 0 && true_class < num_classes) {
                float prob = predictions->data[i * num_classes + true_class];
                if (prob > 0.0f) {
                    loss -= logf(prob);
                }
            }
        }
        loss /= num_samples;
        
        // 简化梯度更新
        for (int i = 0; i < model->num_layers; i++) {
            nn_layer_t* layer = model->layers[i];
            if (layer->weights) {
                for (int j = 0; j < layer->weights->shape[0] * layer->weights->shape[1]; j++) {
                    layer->weights->data[j] -= 0.01f * (random_float() - 0.5f) * loss;
                }
            }
            if (layer->bias) {
                for (int j = 0; j < layer->bias->shape[0]; j++) {
                    layer->bias->data[j] -= 0.01f * (random_float() - 0.5f) * loss;
                }
            }
        }
        
        tensor_free(predictions);
        
        // 记录适应准确率
        tensor_t* query_predictions = nn_module_forward(model, new_task->query_set);
        if (query_predictions) {
            float accuracy = calculate_accuracy(query_predictions, new_task->query_labels);
            if (step < manager->result->num_adaptation_steps) {
                manager->result->adaptation_accuracy[step] = accuracy;
            }
            tensor_free(query_predictions);
        }
        
        printf("适应步 %d，损失: %.4f\n", step, loss);
    }
    
    printf("快速适应完成\n");
    return 0;
}

// ===========================================
// 结果获取
// ===========================================

meta_learning_result_t* get_meta_learning_result(const meta_learning_manager_t* manager) {
    if (!manager) {
        return NULL;
    }
    
    return manager->result;
}

nn_module_t* get_meta_learned_model(const meta_learning_manager_t* manager) {
    if (!manager || !manager->result || !manager->result->is_model_ready) {
        return NULL;
    }
    
    return manager->result->meta_model;
}

// ===========================================
// 性能评估
// ===========================================

float evaluate_meta_learning_performance(meta_learning_manager_t* manager, 
                                       const meta_learning_task_t* test_task) {
    if (!manager || !test_task || !manager->result || !manager->result->is_model_ready) {
        return 0.0f;
    }
    
    nn_module_t* model = manager->result->meta_model;
    
    // 快速适应
    adapt_to_new_task(manager, test_task);
    
    // 评估性能
    tensor_t* predictions = nn_module_forward(model, test_task->query_set);
    if (!predictions) {
        return 0.0f;
    }
    
    float accuracy = calculate_accuracy(predictions, test_task->query_labels);
    tensor_free(predictions);
    
    printf("元学习性能评估: %.4f\n", accuracy);
    
    return accuracy;
}

// ===========================================
// 回调函数设置
// ===========================================

void set_meta_learning_progress_callback(meta_learning_manager_t* manager,
                                       void (*callback)(int progress, const char* message)) {
    if (manager) {
        manager->progress_callback = callback;
    }
}

void set_task_complete_callback(meta_learning_manager_t* manager,
                              void (*callback)(int task_id, float performance)) {
    if (manager) {
        manager->task_complete_callback = callback;
    }
}

void set_meta_learning_complete_callback(meta_learning_manager_t* manager,
                                       void (*callback)(const meta_learning_result_t* result)) {
    if (manager) {
        manager->meta_learning_complete_callback = callback;
    }
}

// ===========================================
// 工具函数
// ===========================================

meta_learning_config_t create_default_meta_learning_config(void) {
    meta_learning_config_t config;
    
    config.method = META_LEARNING_MAML;
    config.num_tasks = 10;
    config.inner_loop_steps = 5;
    config.inner_learning_rate = 0.01f;
    config.outer_learning_rate = 0.001f;
    config.batch_size = 32;
    config.adaptation_steps = 10;
    
    config.maml_first_order = false;
    config.maml_use_batch_norm = true;
    
    config.reptile_step_size = 0.1f;
    config.reptile_inner_batches = 10;
    
    config.protonet_use_euclidean = true;
    config.protonet_distance_scale = 1.0f;
    
    config.use_second_order = true;
    config.normalize_gradients = true;
    config.gradient_clip_value = 1.0f;
    
    config.max_epochs = 100;
    config.early_stopping_patience = 10.0f;
    config.use_validation = true;
    
    config.max_time_seconds = 3600;
    config.max_memory_mb = 4096;
    
    return config;
}

meta_learning_task_t* create_meta_learning_task(const char* name, int n_way, int k_shot) {
    meta_learning_task_t* task = (meta_learning_task_t*)calloc(1, sizeof(meta_learning_task_t));
    if (task) {
        task->task_name = strdup(name);
        task->n_way = n_way;
        task->k_shot = k_shot;
        task->num_classes = n_way;
        task->adaptation_steps = 10;
    }
    return task;
}

void destroy_meta_learning_task(meta_learning_task_t* task) {
    if (task) {
        free(task->task_name);
        // 注意：张量需要外部管理
        free(task);
    }
}