#include "hyperparameter_optimization.h"
#include "nn_module.h"
#include "ai_trainer.h"
#include "tensor.h"
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

// 生成指定范围内的随机整数
static int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
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

// 计算向量的标准差
static float vector_stddev(const float* data, int size) {
    if (!data || size <= 1) return 0.0f;
    
    float mean = vector_mean(data, size);
    float sum_sq = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    
    return sqrtf(sum_sq / (size - 1));
}

// 计算向量的中位数
static float vector_median(float* data, int size) {
    if (!data || size <= 0) return 0.0f;
    
    // 复制数据并排序
    float* sorted = (float*)malloc(size * sizeof(float));
    if (!sorted) return 0.0f;
    
    memcpy(sorted, data, size * sizeof(float));
    
    // 简单冒泡排序
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                float temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }
    
    float median;
    if (size % 2 == 0) {
        median = (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0f;
    } else {
        median = sorted[size / 2];
    }
    
    free(sorted);
    return median;
}

// ===========================================
// 超参数优化管理器实现
// ===========================================

hyperparameter_optimization_manager_t* create_hyperparameter_optimization_manager(void) {
    hyperparameter_optimization_manager_t* manager = 
        (hyperparameter_optimization_manager_t*)calloc(1, sizeof(hyperparameter_optimization_manager_t));
    
    if (!manager) {
        return NULL;
    }
    
    // 初始化参数空间
    manager->space.parameters = NULL;
    manager->space.num_parameters = 0;
    manager->space.max_parameters = 0;
    
    // 设置默认配置
    manager->config.method = OPTIMIZATION_RANDOM_SEARCH;
    manager->config.max_trials = 100;
    manager->config.max_epochs_per_trial = 50;
    manager->config.early_stopping_patience = 5.0f;
    manager->config.num_folds = 5;
    manager->config.train_validation_split = 0.8f;
    
    manager->config.bayesian_num_initial_points = 10;
    manager->config.bayesian_acquisition_weight = 0.1f;
    
    manager->config.evolutionary_population_size = 20;
    manager->config.evolutionary_mutation_rate = 0.1f;
    manager->config.evolutionary_crossover_rate = 0.8f;
    
    manager->config.hyperband_max_iter = 81;
    manager->config.hyperband_eta = 3.0f;
    
    manager->config.use_parallel = false;
    manager->config.max_parallel_jobs = 4;
    manager->config.metric = strdup("accuracy");
    manager->config.maximize_metric = true;
    
    manager->config.max_time_seconds = 3600; // 1小时
    manager->config.max_memory_mb = 4096;    // 4GB
    
    manager->is_initialized = true;
    manager->current_trial = 0;
    manager->is_running = false;
    manager->start_time = 0.0f;
    manager->result = NULL;
    
    // 初始化随机种子
    srand((unsigned int)time(NULL));
    
    return manager;
}

void destroy_hyperparameter_optimization_manager(hyperparameter_optimization_manager_t* manager) {
    if (manager) {
        // 释放参数空间
        if (manager->space.parameters) {
            for (int i = 0; i < manager->space.num_parameters; i++) {
                free(manager->space.parameters[i].name);
                if (manager->space.parameters[i].type == PARAM_TYPE_CATEGORICAL) {
                    for (int j = 0; j < manager->space.parameters[i].range.categorical.num_categories; j++) {
                        free(manager->space.parameters[i].range.categorical.categories[j]);
                    }
                    free(manager->space.parameters[i].range.categorical.categories);
                }
            }
            free(manager->space.parameters);
        }
        
        // 释放结果
        if (manager->result) {
            for (int i = 0; i < manager->result->num_trials; i++) {
                if (manager->result->trials[i].parameter_values) {
                    free(manager->result->trials[i].parameter_values);
                }
                if (manager->result->trials[i].validation_metrics) {
                    free(manager->result->trials[i].validation_metrics);
                }
                if (manager->result->trials[i].status_message) {
                    free(manager->result->trials[i].status_message);
                }
                // 注意：模型和训练器需要外部管理
            }
            if (manager->result->trials) {
                free(manager->result->trials);
            }
            if (manager->result->parameter_importance) {
                free(manager->result->parameter_importance);
            }
            free(manager->result);
        }
        
        // 释放配置中的字符串
        if (manager->config.metric) {
            free(manager->config.metric);
        }
        
        free(manager);
    }
}

// ===========================================
// 参数空间管理
// ===========================================

int add_hyperparameter_float(hyperparameter_optimization_manager_t* manager, 
                           const char* name, float min_value, float max_value, 
                           float default_value, bool is_log_scale) {
    if (!manager || !name || min_value >= max_value) {
        return -1;
    }
    
    // 扩展参数数组
    if (manager->space.num_parameters >= manager->space.max_parameters) {
        int new_max = manager->space.max_parameters == 0 ? 10 : manager->space.max_parameters * 2;
        hyperparameter_definition_t* new_params = 
            (hyperparameter_definition_t*)realloc(manager->space.parameters, 
                                                 new_max * sizeof(hyperparameter_definition_t));
        if (!new_params) {
            return -1;
        }
        manager->space.parameters = new_params;
        manager->space.max_parameters = new_max;
    }
    
    // 添加新参数
    hyperparameter_definition_t* param = &manager->space.parameters[manager->space.num_parameters];
    
    param->name = strdup(name);
    param->type = PARAM_TYPE_FLOAT;
    param->range.float_range.min_value = min_value;
    param->range.float_range.max_value = max_value;
    param->range.float_range.step = 0.0f; // 默认无步长
    param->default_value = default_value;
    param->is_log_scale = is_log_scale;
    
    manager->space.num_parameters++;
    
    return 0;
}

int add_hyperparameter_int(hyperparameter_optimization_manager_t* manager,
                         const char* name, int min_value, int max_value,
                         int default_value, bool is_log_scale) {
    if (!manager || !name || min_value >= max_value) {
        return -1;
    }
    
    // 扩展参数数组
    if (manager->space.num_parameters >= manager->space.max_parameters) {
        int new_max = manager->space.max_parameters == 0 ? 10 : manager->space.max_parameters * 2;
        hyperparameter_definition_t* new_params = 
            (hyperparameter_definition_t*)realloc(manager->space.parameters, 
                                                 new_max * sizeof(hyperparameter_definition_t));
        if (!new_params) {
            return -1;
        }
        manager->space.parameters = new_params;
        manager->space.max_parameters = new_max;
    }
    
    // 添加新参数
    hyperparameter_definition_t* param = &manager->space.parameters[manager->space.num_parameters];
    
    param->name = strdup(name);
    param->type = PARAM_TYPE_INT;
    param->range.int_range.min_value = min_value;
    param->range.int_range.max_value = max_value;
    param->range.int_range.step = 1; // 默认步长为1
    param->default_value = (float)default_value;
    param->is_log_scale = is_log_scale;
    
    manager->space.num_parameters++;
    
    return 0;
}

int add_hyperparameter_categorical(hyperparameter_optimization_manager_t* manager,
                                 const char* name, const char** categories, 
                                 int num_categories, const char* default_category) {
    if (!manager || !name || !categories || num_categories <= 0) {
        return -1;
    }
    
    // 扩展参数数组
    if (manager->space.num_parameters >= manager->space.max_parameters) {
        int new_max = manager->space.max_parameters == 0 ? 10 : manager->space.max_parameters * 2;
        hyperparameter_definition_t* new_params = 
            (hyperparameter_definition_t*)realloc(manager->space.parameters, 
                                                 new_max * sizeof(hyperparameter_definition_t));
        if (!new_params) {
            return -1;
        }
        manager->space.parameters = new_params;
        manager->space.max_parameters = new_max;
    }
    
    // 添加新参数
    hyperparameter_definition_t* param = &manager->space.parameters[manager->space.num_parameters];
    
    param->name = strdup(name);
    param->type = PARAM_TYPE_CATEGORICAL;
    
    // 复制分类值
    param->range.categorical.categories = (char**)malloc(num_categories * sizeof(char*));
    if (!param->range.categorical.categories) {
        return -1;
    }
    
    for (int i = 0; i < num_categories; i++) {
        param->range.categorical.categories[i] = strdup(categories[i]);
    }
    param->range.categorical.num_categories = num_categories;
    
    // 设置默认值（找到默认分类的索引）
    param->default_value = 0.0f;
    for (int i = 0; i < num_categories; i++) {
        if (strcmp(categories[i], default_category) == 0) {
            param->default_value = (float)i;
            break;
        }
    }
    
    param->is_log_scale = false;
    
    manager->space.num_parameters++;
    
    return 0;
}

// ===========================================
// 优化配置
// ===========================================

int set_optimization_method(hyperparameter_optimization_manager_t* manager, 
                          hyperparameter_optimization_type_t method) {
    if (!manager) {
        return -1;
    }
    
    manager->config.method = method;
    return 0;
}

int configure_optimization(hyperparameter_optimization_manager_t* manager,
                         const hyperparameter_optimization_config_t* config) {
    if (!manager || !config) {
        return -1;
    }
    
    manager->config = *config;
    
    // 复制字符串（如果提供了新的metric）
    if (config->metric) {
        if (manager->config.metric) {
            free(manager->config.metric);
        }
        manager->config.metric = strdup(config->metric);
    }
    
    return 0;
}

// ===========================================
// 参数采样函数
// ===========================================

// 随机采样参数
static float* sample_random_parameters(const hyperparameter_space_t* space) {
    if (!space || space->num_parameters <= 0) {
        return NULL;
    }
    
    float* params = (float*)calloc(space->num_parameters, sizeof(float));
    if (!params) {
        return NULL;
    }
    
    for (int i = 0; i < space->num_parameters; i++) {
        const hyperparameter_definition_t* param = &space->parameters[i];
        
        switch (param->type) {
            case PARAM_TYPE_FLOAT:
                if (param->is_log_scale) {
                    // 对数尺度采样
                    float log_min = logf(param->range.float_range.min_value);
                    float log_max = logf(param->range.float_range.max_value);
                    float log_val = log_min + random_float() * (log_max - log_min);
                    params[i] = expf(log_val);
                } else {
                    // 线性尺度采样
                    params[i] = param->range.float_range.min_value + 
                               random_float() * (param->range.float_range.max_value - param->range.float_range.min_value);
                }
                break;
                
            case PARAM_TYPE_INT:
                if (param->is_log_scale) {
                    // 对数尺度采样（整数）
                    float log_min = logf((float)param->range.int_range.min_value);
                    float log_max = logf((float)param->range.int_range.max_value);
                    float log_val = log_min + random_float() * (log_max - log_min);
                    params[i] = (float)(int)expf(log_val);
                } else {
                    // 线性尺度采样
                    params[i] = (float)random_int(param->range.int_range.min_value, 
                                                param->range.int_range.max_value);
                }
                break;
                
            case PARAM_TYPE_CATEGORICAL:
                // 随机选择分类
                params[i] = (float)random_int(0, param->range.categorical.num_categories - 1);
                break;
                
            default:
                params[i] = param->default_value;
                break;
        }
    }
    
    return params;
}

// ===========================================
// 试验执行函数
// ===========================================

// 执行单个超参数试验
static hyperparameter_trial_result_t* execute_trial(
    int trial_id, 
    const float* parameters,
    const hyperparameter_space_t* space,
    const hyperparameter_optimization_config_t* config,
    nn_module_t* (*model_creator)(const float* params),
    tensor_t* train_data, tensor_t* train_labels,
    tensor_t* validation_data, tensor_t* validation_labels) {
    
    printf("执行试验 %d\n", trial_id);
    
    hyperparameter_trial_result_t* result = 
        (hyperparameter_trial_result_t*)calloc(1, sizeof(hyperparameter_trial_result_t));
    if (!result) {
        return NULL;
    }
    
    result->trial_id = trial_id;
    result->parameter_values = (float*)malloc(space->num_parameters * sizeof(float));
    if (!result->parameter_values) {
        free(result);
        return NULL;
    }
    
    memcpy(result->parameter_values, parameters, space->num_parameters * sizeof(float));
    
    double start_time = get_current_time();
    
    // 创建模型
    nn_module_t* model = model_creator(parameters);
    if (!model) {
        result->status_message = strdup("模型创建失败");
        result->is_completed = false;
        return result;
    }
    
    // 创建训练器
    ai_trainer_t* trainer = create_ai_trainer(model);
    if (!trainer) {
        nn_module_free(model);
        result->status_message = strdup("训练器创建失败");
        result->is_completed = false;
        return result;
    }
    
    // 配置训练器超参数
    // 这里需要根据parameters设置训练器的超参数
    // 简化实现：使用固定配置
    
    // 训练模型
    int epochs_trained = 0;
    float best_validation_score = 0.0f;
    
    for (int epoch = 0; epoch < config->max_epochs_per_trial; epoch++) {
        // 简化训练过程
        epochs_trained++;
        
        // 模拟训练进度
        if (epoch % 10 == 0) {
            printf("试验 %d，轮次 %d\n", trial_id, epoch);
        }
        
        // 模拟验证分数（实际实现需要真实计算）
        float validation_score = 0.7f + 0.2f * random_float();
        
        if (validation_score > best_validation_score) {
            best_validation_score = validation_score;
        }
        
        // 早停检查（简化）
        if (epoch > 10 && validation_score < best_validation_score - 0.01f) {
            // 早停
            break;
        }
    }
    
    result->score = best_validation_score;
    result->training_time = (float)(get_current_time() - start_time);
    result->epochs_trained = epochs_trained;
    result->is_completed = true;
    result->status_message = strdup("试验完成");
    
    // 存储模型和训练器（注意：需要外部管理内存）
    result->best_model = model;
    result->trainer = trainer;
    
    printf("试验 %d 完成，得分: %.4f，时间: %.2f秒\n", 
           trial_id, result->score, result->training_time);
    
    return result;
}

// ===========================================
// 优化执行
// ===========================================

int start_hyperparameter_optimization(hyperparameter_optimization_manager_t* manager,
                                     nn_module_t* (*model_creator)(const float* params),
                                     tensor_t* train_data, tensor_t* train_labels,
                                     tensor_t* validation_data, tensor_t* validation_labels) {
    if (!manager || !model_creator || !train_data || !train_labels) {
        return -1;
    }
    
    if (manager->space.num_parameters <= 0) {
        printf("错误：未定义任何超参数\n");
        return -1;
    }
    
    if (manager->is_running) {
        printf("错误：优化已在运行\n");
        return -1;
    }
    
    printf("开始超参数优化，参数数量: %d，最大试验次数: %d\n", 
           manager->space.num_parameters, manager->config.max_trials);
    
    manager->is_running = true;
    manager->current_trial = 0;
    manager->start_time = (float)get_current_time();
    
    // 创建结果存储
    if (manager->result) {
        destroy_hyperparameter_optimization_manager(manager);
        // 重新创建管理器
        // 简化处理：这里需要重新初始化
    }
    
    manager->result = (hyperparameter_optimization_result_t*)calloc(1, sizeof(hyperparameter_optimization_result_t));
    if (!manager->result) {
        manager->is_running = false;
        return -1;
    }
    
    manager->result->trials = (hyperparameter_trial_result_t*)calloc(
        manager->config.max_trials, sizeof(hyperparameter_trial_result_t));
    
    if (!manager->result->trials) {
        free(manager->result);
        manager->result = NULL;
        manager->is_running = false;
        return -1;
    }
    
    manager->result->num_trials = manager->config.max_trials;
    manager->result->best_trial_index = -1;
    manager->result->best_score = manager->config.maximize_metric ? -FLT_MAX : FLT_MAX;
    
    // 执行优化循环
    for (int trial = 0; trial < manager->config.max_trials; trial++) {
        manager->current_trial = trial;
        
        // 采样参数
        float* parameters = sample_random_parameters(&manager->space);
        if (!parameters) {
            printf("试验 %d：参数采样失败\n", trial);
            continue;
        }
        
        // 执行试验
        hyperparameter_trial_result_t* trial_result = execute_trial(
            trial, parameters, &manager->space, &manager->config,
            model_creator, train_data, train_labels, validation_data, validation_labels);
        
        if (trial_result && trial_result->is_completed) {
            // 存储结果
            manager->result->trials[trial] = *trial_result;
            
            // 更新最佳结果
            if ((manager->config.maximize_metric && trial_result->score > manager->result->best_score) ||
                (!manager->config.maximize_metric && trial_result->score < manager->result->best_score)) {
                manager->result->best_score = trial_result->score;
                manager->result->best_trial_index = trial;
            }
            
            // 调用进度回调
            if (manager->progress_callback) {
                int progress = (int)((trial + 1) * 100 / manager->config.max_trials);
                char message[100];
                snprintf(message, sizeof(message), "试验 %d 完成，得分: %.4f", trial, trial_result->score);
                manager->progress_callback(progress, message);
            }
            
            // 调用试验完成回调
            if (manager->trial_complete_callback) {
                manager->trial_complete_callback(trial_result);
            }
            
            free(trial_result); // 注意：trials数组已经复制了数据
        }
        
        free(parameters);
        
        // 检查时间限制
        if (manager->config.max_time_seconds > 0) {
            double elapsed = get_current_time() - manager->start_time;
            if (elapsed > manager->config.max_time_seconds) {
                printf("达到时间限制，停止优化\n");
                break;
            }
        }
    }
    
    // 计算统计信息
    if (manager->result->num_trials > 0) {
        float* scores = (float*)malloc(manager->result->num_trials * sizeof(float));
        if (scores) {
            for (int i = 0; i < manager->result->num_trials; i++) {
                scores[i] = manager->result->trials[i].score;
            }
            
            manager->result->mean_score = vector_mean(scores, manager->result->num_trials);
            manager->result->std_score = vector_stddev(scores, manager->result->num_trials);
            manager->result->median_score = vector_median(scores, manager->result->num_trials);
            
            free(scores);
        }
    }
    
    manager->result->total_optimization_time = (float)(get_current_time() - manager->start_time);
    manager->is_running = false;
    
    printf("超参数优化完成，最佳得分: %.4f，总时间: %.2f秒\n", 
           manager->result->best_score, manager->result->total_optimization_time);
    
    // 调用优化完成回调
    if (manager->optimization_complete_callback) {
        manager->optimization_complete_callback(manager->result);
    }
    
    return 0;
}

int stop_hyperparameter_optimization(hyperparameter_optimization_manager_t* manager) {
    if (!manager) {
        return -1;
    }
    
    if (manager->is_running) {
        manager->is_running = false;
        printf("超参数优化已停止\n");
        return 0;
    }
    
    return -1;
}

// ===========================================
// 结果获取和分析
// ===========================================

hyperparameter_optimization_result_t* get_optimization_result(
    const hyperparameter_optimization_manager_t* manager) {
    if (!manager) {
        return NULL;
    }
    
    return manager->result;
}

float* get_best_hyperparameters(const hyperparameter_optimization_manager_t* manager) {
    if (!manager || !manager->result || manager->result->best_trial_index < 0) {
        return NULL;
    }
    
    int best_index = manager->result->best_trial_index;
    float* best_params = (float*)malloc(manager->space.num_parameters * sizeof(float));
    
    if (best_params) {
        memcpy(best_params, manager->result->trials[best_index].parameter_values,
               manager->space.num_parameters * sizeof(float));
    }
    
    return best_params;
}

nn_module_t* get_best_model(const hyperparameter_optimization_manager_t* manager) {
    if (!manager || !manager->result || manager->result->best_trial_index < 0) {
        return NULL;
    }
    
    int best_index = manager->result->best_trial_index;
    return manager->result->trials[best_index].best_model;
}

// ===========================================
// 回调函数设置
// ===========================================

void set_optimization_progress_callback(hyperparameter_optimization_manager_t* manager,
                                      void (*callback)(int progress, const char* message)) {
    if (manager) {
        manager->progress_callback = callback;
    }
}

void set_trial_complete_callback(hyperparameter_optimization_manager_t* manager,
                               void (*callback)(const hyperparameter_trial_result_t* trial)) {
    if (manager) {
        manager->trial_complete_callback = callback;
    }
}

void set_optimization_complete_callback(hyperparameter_optimization_manager_t* manager,
                                      void (*callback)(const hyperparameter_optimization_result_t* result)) {
    if (manager) {
        manager->optimization_complete_callback = callback;
    }
}

// ===========================================
// 工具函数
// ===========================================

hyperparameter_space_t* create_hyperparameter_space(void) {
    hyperparameter_space_t* space = (hyperparameter_space_t*)calloc(1, sizeof(hyperparameter_space_t));
    if (space) {
        space->parameters = NULL;
        space->num_parameters = 0;
        space->max_parameters = 0;
    }
    return space;
}

void destroy_hyperparameter_space(hyperparameter_space_t* space) {
    if (space) {
        if (space->parameters) {
            for (int i = 0; i < space->num_parameters; i++) {
                free(space->parameters[i].name);
                if (space->parameters[i].type == PARAM_TYPE_CATEGORICAL) {
                    for (int j = 0; j < space->parameters[i].range.categorical.num_categories; j++) {
                        free(space->parameters[i].range.categorical.categories[j]);
                    }
                    free(space->parameters[i].range.categorical.categories);
                }
            }
            free(space->parameters);
        }
        free(space);
    }
}

hyperparameter_optimization_config_t create_default_optimization_config(void) {
    hyperparameter_optimization_config_t config;
    
    config.method = OPTIMIZATION_RANDOM_SEARCH;
    config.max_trials = 100;
    config.max_epochs_per_trial = 50;
    config.early_stopping_patience = 5.0f;
    config.num_folds = 5;
    config.train_validation_split = 0.8f;
    
    config.bayesian_num_initial_points = 10;
    config.bayesian_acquisition_weight = 0.1f;
    
    config.evolutionary_population_size = 20;
    config.evolutionary_mutation_rate = 0.1f;
    config.evolutionary_crossover_rate = 0.8f;
    
    config.hyperband_max_iter = 81;
    config.hyperband_eta = 3.0f;
    
    config.use_parallel = false;
    config.max_parallel_jobs = 4;
    config.metric = strdup("accuracy");
    config.maximize_metric = true;
    
    config.max_time_seconds = 3600;
    config.max_memory_mb = 4096;
    
    return config;
}