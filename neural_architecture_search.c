#include "neural_architecture_search.h"
#include "nn_module.h"
#include "model_builder.h"
#include "optimizer.h"
#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ===========================================
// 内部工具函数
// ===========================================

// 生成随机整数 [min, max]
static int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

// 生成随机浮点数 [0.0, 1.0]
static float random_float(void) {
    return (float)rand() / RAND_MAX;
}

// 生成随机浮点数 [min, max]
static float random_float_range(float min, float max) {
    return min + random_float() * (max - min);
}

// 计算数组的平均值
static float calculate_average(const float* values, int count) {
    if (count <= 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / count;
}

// ===========================================
// NAS 管理器实现
// ===========================================

neural_architecture_search_manager_t* create_nas_manager(void) {
    neural_architecture_search_manager_t* manager = 
        (neural_architecture_search_manager_t*)calloc(1, sizeof(neural_architecture_search_manager_t));
    
    if (!manager) {
        return NULL;
    }
    
    // 初始化默认配置
    manager->config = create_default_nas_config();
    manager->population = NULL;
    manager->population_count = 0;
    manager->is_searching = false;
    
    // 初始化搜索结果
    memset(&manager->search_result, 0, sizeof(nas_search_result_t));
    manager->search_result.best_fitness = -FLT_MAX;
    
    // 设置随机种子
    srand((unsigned int)time(NULL));
    
    return manager;
}

void destroy_nas_manager(neural_architecture_search_manager_t* manager) {
    if (!manager) return;
    
    // 释放种群内存
    if (manager->population) {
        for (int i = 0; i < manager->population_count; i++) {
            if (manager->population[i].layers) {
                free(manager->population[i].layers);
            }
        }
        free(manager->population);
    }
    
    // 释放最佳架构内存
    if (manager->search_result.best_architecture.layers) {
        free(manager->search_result.best_architecture.layers);
    }
    
    free(manager);
}

// ===========================================
// 配置设置函数
// ===========================================

int configure_nas_search(neural_architecture_search_manager_t* manager, const nas_config_t* config) {
    if (!manager || !config) {
        return -1;
    }
    
    manager->config = *config;
    return 0;
}

int set_nas_search_method(neural_architecture_search_manager_t* manager, nas_search_method_t method) {
    if (!manager) {
        return -1;
    }
    
    manager->config.search_method = method;
    return 0;
}

// ===========================================
// 架构生成和操作函数
// ===========================================

architecture_encoding_t generate_random_architecture(const nas_config_t* config) {
    architecture_encoding_t arch;
    memset(&arch, 0, sizeof(architecture_encoding_t));
    
    if (!config) {
        return arch;
    }
    
    // 随机生成层数
    int num_layers = random_int(config->min_layers, config->max_layers);
    arch.layers = (layer_encoding_t*)calloc(num_layers, sizeof(layer_encoding_t));
    arch.num_layers = num_layers;
    
    if (!arch.layers) {
        arch.num_layers = 0;
        return arch;
    }
    
    // 为每层生成随机配置
    for (int i = 0; i < num_layers; i++) {
        layer_encoding_t* layer = &arch.layers[i];
        
        // 随机选择层类型（简化处理，主要使用全连接层）
        layer->layer_type = LAYER_SEARCH_DENSE;
        
        // 随机生成单元数
        layer->units = random_int(config->min_units_per_layer, config->max_units_per_layer);
        
        // 随机选择激活函数
        layer->activation = random_int(0, ACTIVATION_SEARCH_SWISH);
        
        // 随机决定是否使用批归一化和Dropout
        layer->use_batch_norm = config->use_batch_norm && (random_float() > 0.5f);
        layer->use_dropout = config->use_dropout && (random_float() > 0.5f);
        
        if (layer->use_dropout) {
            layer->dropout_rate = random_float_range(0.1f, 0.5f);
        } else {
            layer->dropout_rate = 0.0f;
        }
    }
    
    // 计算复杂度
    arch.complexity = calculate_architecture_complexity(&arch);
    
    return arch;
}

architecture_encoding_t mutate_architecture(const architecture_encoding_t* parent, const nas_config_t* config) {
    architecture_encoding_t child;
    memset(&child, 0, sizeof(architecture_encoding_t));
    
    if (!parent || !config || parent->num_layers <= 0 || !parent->layers) {
        return child;
    }
    
    // 复制父架构
    child.num_layers = parent->num_layers;
    child.layers = (layer_encoding_t*)calloc(child.num_layers, sizeof(layer_encoding_t));
    
    if (!child.layers) {
        child.num_layers = 0;
        return child;
    }
    
    memcpy(child.layers, parent->layers, child.num_layers * sizeof(layer_encoding_t));
    
    // 应用变异
    for (int i = 0; i < child.num_layers; i++) {
        if (random_float() < config->mutation_rate) {
            // 变异单元数
            int delta = random_int(-config->max_units_per_layer / 10, config->max_units_per_layer / 10);
            child.layers[i].units += delta;
            
            // 确保在有效范围内
            if (child.layers[i].units < config->min_units_per_layer) {
                child.layers[i].units = config->min_units_per_layer;
            }
            if (child.layers[i].units > config->max_units_per_layer) {
                child.layers[i].units = config->max_units_per_layer;
            }
        }
        
        if (random_float() < config->mutation_rate / 2) {
            // 变异激活函数
            child.layers[i].activation = random_int(0, ACTIVATION_SEARCH_SWISH);
        }
        
        if (random_float() < config->mutation_rate / 3) {
            // 变异批归一化设置
            child.layers[i].use_batch_norm = !child.layers[i].use_batch_norm;
        }
        
        if (random_float() < config->mutation_rate / 3) {
            // 变异Dropout设置
            child.layers[i].use_dropout = !child.layers[i].use_dropout;
            if (child.layers[i].use_dropout) {
                child.layers[i].dropout_rate = random_float_range(0.1f, 0.5f);
            }
        }
    }
    
    // 计算复杂度
    child.complexity = calculate_architecture_complexity(&child);
    
    return child;
}

architecture_encoding_t crossover_architectures(const architecture_encoding_t* parent1,
                                               const architecture_encoding_t* parent2,
                                               const nas_config_t* config) {
    architecture_encoding_t child;
    memset(&child, 0, sizeof(architecture_encoding_t));
    
    if (!parent1 || !parent2 || !config) {
        return child;
    }
    
    // 选择较短的层数作为基础
    int min_layers = parent1->num_layers < parent2->num_layers ? parent1->num_layers : parent2->num_layers;
    child.num_layers = min_layers;
    child.layers = (layer_encoding_t*)calloc(child.num_layers, sizeof(layer_encoding_t));
    
    if (!child.layers) {
        child.num_layers = 0;
        return child;
    }
    
    // 单点交叉
    int crossover_point = random_int(1, min_layers - 1);
    
    for (int i = 0; i < crossover_point; i++) {
        child.layers[i] = parent1->layers[i];
    }
    
    for (int i = crossover_point; i < min_layers; i++) {
        if (i < parent2->num_layers) {
            child.layers[i] = parent2->layers[i];
        } else {
            // 如果parent2层数不够，使用parent1的配置
            child.layers[i] = parent1->layers[i % parent1->num_layers];
        }
    }
    
    // 计算复杂度
    child.complexity = calculate_architecture_complexity(&child);
    
    return child;
}

// ===========================================
// 适应度和复杂度计算
// ===========================================

float calculate_architecture_complexity(const architecture_encoding_t* architecture) {
    if (!architecture || architecture->num_layers <= 0) {
        return 0.0f;
    }
    
    float complexity = 0.0f;
    
    for (int i = 0; i < architecture->num_layers; i++) {
        const layer_encoding_t* layer = &architecture->layers[i];
        
        // 基本复杂度基于单元数
        complexity += layer->units;
        
        // 不同类型层的额外复杂度
        switch (layer->layer_type) {
            case LAYER_SEARCH_CONV1D:
                complexity *= 1.5f;
                break;
            case LAYER_SEARCH_CONV2D:
                complexity *= 2.0f;
                break;
            case LAYER_SEARCH_LSTM:
                complexity *= 3.0f;
                break;
            case LAYER_SEARCH_GRU:
                complexity *= 2.5f;
                break;
            case LAYER_SEARCH_ATTENTION:
                complexity *= 4.0f;
                break;
            case LAYER_SEARCH_TRANSFORMER:
                complexity *= 5.0f;
                break;
            default:
                // 全连接层，复杂度不变
                break;
        }
        
        // 批归一化和Dropout增加复杂度
        if (layer->use_batch_norm) {
            complexity += layer->units * 0.1f;
        }
        if (layer->use_dropout) {
            complexity += layer->units * 0.05f;
        }
    }
    
    return complexity;
}

float calculate_architecture_fitness(const architecture_encoding_t* architecture,
                                     float accuracy, float complexity, float inference_time) {
    if (!architecture) {
        return 0.0f;
    }
    
    // 多目标适应度函数
    // 平衡准确率、复杂度和推理时间
    float fitness = accuracy;
    
    // 惩罚高复杂度
    fitness -= complexity * 0.001f;
    
    // 惩罚长推理时间
    fitness -= inference_time * 0.01f;
    
    // 鼓励使用更少的层
    if (architecture->num_layers > 0) {
        fitness += (10.0f / architecture->num_layers) * 0.1f;
    }
    
    return fitness;
}

// ===========================================
// 架构评估和构建
// ===========================================

float evaluate_architecture(const architecture_encoding_t* architecture,
                           const training_data_t* train_data,
                           const training_data_t* val_data) {
    if (!architecture || !train_data) {
        return 0.0f;
    }
    
    printf("评估架构，层数: %d\n", architecture->num_layers);
    
    // 构建模型
    nn_module_t* model = build_architecture(architecture);
    if (!model) {
        printf("构建模型失败\n");
        return 0.0f;
    }
    
    // 简化评估：使用随机准确率
    // 在实际实现中，这里应该:
    // 1. 训练模型（简化训练）
    // 2. 在验证集上评估
    // 3. 计算推理时间
    
    float accuracy = random_float_range(0.7f, 0.95f);
    float inference_time = architecture->complexity * 0.001f;
    
    // 计算适应度
    float fitness = calculate_architecture_fitness(architecture, accuracy, 
                                                  architecture->complexity, inference_time);
    
    printf("架构评估完成: 准确率=%.3f, 复杂度=%.1f, 适应度=%.3f\n", 
           accuracy, architecture->complexity, fitness);
    
    // 清理模型
    // 注意：在实际实现中需要更完善的模型管理
    
    return fitness;
}

nn_module_t* build_architecture(const architecture_encoding_t* architecture) {
    if (!architecture || architecture->num_layers <= 0) {
        return NULL;
    }
    
    printf("构建架构，层数: %d\n", architecture->num_layers);
    
    // 创建模型构建器
    ModelBuilder* builder = model_builder_create();
    if (!builder) {
        return NULL;
    }
    
    // 添加输入层（假设输入维度为特征数）
    // 在实际实现中，需要根据数据确定输入维度
    int input_dim = 784;  // 示例：MNIST数据
    
    // 构建每一层
    for (int i = 0; i < architecture->num_layers; i++) {
        const layer_encoding_t* layer = &architecture->layers[i];
        
        // 添加全连接层
        model_builder_add_layer(builder, LAYER_DENSE, layer->units);
        
        // 添加激活函数
        switch (layer->activation) {
            case ACTIVATION_SEARCH_RELU:
                model_builder_add_layer(builder, LAYER_RELU, 0);
                break;
            case ACTIVATION_SEARCH_SIGMOID:
                model_builder_add_layer(builder, LAYER_SIGMOID, 0);
                break;
            case ACTIVATION_SEARCH_TANH:
                model_builder_add_layer(builder, LAYER_TANH, 0);
                break;
            case ACTIVATION_SEARCH_LEAKY_RELU:
                model_builder_add_layer(builder, LAYER_LEAKY_RELU, 0);
                break;
            default:
                model_builder_add_layer(builder, LAYER_RELU, 0);
                break;
        }
        
        // 添加批归一化
        if (layer->use_batch_norm) {
            model_builder_add_layer(builder, LAYER_BATCH_NORM, 0);
        }
        
        // 添加Dropout
        if (layer->use_dropout) {
            model_builder_add_layer(builder, LAYER_DROPOUT, 0);
            // 注意：需要设置dropout率，这里简化处理
        }
    }
    
    // 添加输出层（假设分类任务）
    int output_dim = 10;  // 示例：10类分类
    model_builder_add_layer(builder, LAYER_DENSE, output_dim);
    model_builder_add_layer(builder, LAYER_SOFTMAX, 0);
    
    // 构建模型
    nn_module_t* model = model_builder_build(builder);
    
    printf("架构构建完成\n");
    return model;
}

// ===========================================
// 主要搜索算法
// ===========================================

nas_search_result_t perform_architecture_search(neural_architecture_search_manager_t* manager,
                                               const training_data_t* train_data,
                                               const training_data_t* val_data) {
    nas_search_result_t result;
    memset(&result, 0, sizeof(nas_search_result_t));
    result.best_fitness = -FLT_MAX;
    
    if (!manager || !train_data) {
        return result;
    }
    
    printf("开始神经网络架构搜索，方法: %d\n", manager->config.search_method);
    
    clock_t start_time = clock();
    
    // 根据搜索方法执行不同的算法
    switch (manager->config.search_method) {
        case NAS_SEARCH_EVOLUTIONARY:
            result = perform_evolutionary_nas(manager, train_data, val_data);
            break;
        case NAS_SEARCH_RANDOM:
            // 随机搜索作为基线
            result = perform_random_search(manager, train_data, val_data);
            break;
        case NAS_SEARCH_REINFORCEMENT:
            result = perform_reinforcement_nas(manager, train_data, val_data);
            break;
        case NAS_SEARCH_BAYESIAN:
            result = perform_bayesian_nas(manager, train_data, val_data);
            break;
        case NAS_SEARCH_GRADIENT:
            result = perform_gradient_nas(manager, train_data, val_data);
            break;
        default:
            printf("未知的搜索方法: %d\n", manager->config.search_method);
            break;
    }
    
    clock_t end_time = clock();
    result.search_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("架构搜索完成，最佳适应度: %.3f，搜索时间: %.2f秒\n", 
           result.best_fitness, result.search_time);
    
    return result;
}

// 进化算法NAS实现
int perform_evolutionary_nas(neural_architecture_search_manager_t* manager,
                            const training_data_t* train_data,
                            const training_data_t* val_data) {
    if (!manager || !train_data) {
        return -1;
    }
    
    printf("开始进化算法NAS搜索，种群大小: %d，代数: %d\n", 
           manager->config.population_size, manager->config.num_generations);
    
    // 初始化种群
    manager->population_count = manager->config.population_size;
    manager->population = (architecture_encoding_t*)calloc(manager->population_count, 
                                                          sizeof(architecture_encoding_t));
    
    if (!manager->population) {
        return -1;
    }
    
    // 生成初始种群
    for (int i = 0; i < manager->population_count; i++) {
        manager->population[i] = generate_random_architecture(&manager->config);
    }
    
    // 进化循环
    for (int gen = 0; gen < manager->config.num_generations; gen++) {
        printf("进化代数 %d/%d\n", gen + 1, manager->config.num_generations);
        
        // 评估种群
        float* fitnesses = (float*)calloc(manager->population_count, sizeof(float));
        for (int i = 0; i < manager->population_count; i++) {
            fitnesses[i] = evaluate_architecture(&manager->population[i], train_data, val_data);
            manager->population[i].fitness = fitnesses[i];
            
            // 更新最佳架构
            if (fitnesses[i] > manager->search_result.best_fitness) {
                manager->search_result.best_fitness = fitnesses[i];
                
                // 复制最佳架构
                if (manager->search_result.best_architecture.layers) {
                    free(manager->search_result.best_architecture.layers);
                }
                manager->search_result.best_architecture = manager->population[i];
                
                // 需要深拷贝层数据
                manager->search_result.best_architecture.layers = 
                    (layer_encoding_t*)calloc(manager->population[i].num_layers, sizeof(layer_encoding_t));
                memcpy(manager->search_result.best_architecture.layers, 
                       manager->population[i].layers,
                       manager->population[i].num_layers * sizeof(layer_encoding_t));
            }
        }
        
        // 选择、交叉、变异生成新一代
        // 简化实现：使用锦标赛选择
        architecture_encoding_t* new_population = 
            (architecture_encoding_t*)calloc(manager->population_count, sizeof(architecture_encoding_t));
        
        for (int i = 0; i < manager->population_count; i++) {
            // 锦标赛选择
            int parent1_idx = random_int(0, manager->population_count - 1);
            int parent2_idx = random_int(0, manager->population_count - 1);
            
            // 选择适应度更高的作为父代
            const architecture_encoding_t* parent1 = &manager->population[parent1_idx];
            const architecture_encoding_t* parent2 = &manager->population[parent2_idx];
            
            if (fitnesses[parent1_idx] < fitnesses[parent2_idx]) {
                const architecture_encoding_t* temp = parent1;
                parent1 = parent2;
                parent2 = temp;
            }
            
            // 交叉
            if (random_float() < manager->config.crossover_rate) {
                new_population[i] = crossover_architectures(parent1, parent2, &manager->config);
            } else {
                // 直接复制
                new_population[i] = *parent1;
                // 需要深拷贝
                new_population[i].layers = (layer_encoding_t*)calloc(parent1->num_layers, sizeof(layer_encoding_t));
                memcpy(new_population[i].layers, parent1->layers, 
                       parent1->num_layers * sizeof(layer_encoding_t));
            }
            
            // 变异
            if (random_float() < manager->config.mutation_rate) {
                architecture_encoding_t mutated = mutate_architecture(&new_population[i], &manager->config);
                // 清理旧内存
                if (new_population[i].layers) {
                    free(new_population[i].layers);
                }
                new_population[i] = mutated;
            }
        }
        
        // 清理旧种群
        for (int i = 0; i < manager->population_count; i++) {
            if (manager->population[i].layers) {
                free(manager->population[i].layers);
            }
        }
        free(manager->population);
        
        // 更新种群
        manager->population = new_population;
        free(fitnesses);
        
        // 回调进度
        if (manager->progress_callback) {
            int progress = (int)((gen + 1) * 100 / manager->config.num_generations);
            manager->progress_callback(progress, "进化算法NAS搜索中...");
        }
    }
    
    printf("进化算法NAS搜索完成\n");
    return 0;
}

// 随机搜索实现（基线方法）
nas_search_result_t perform_random_search(neural_architecture_search_manager_t* manager,
                                          const training_data_t* train_data,
                                          const training_data_t* val_data) {
    nas_search_result_t result;
    memset(&result, 0, sizeof(nas_search_result_t));
    result.best_fitness = -FLT_MAX;
    
    if (!manager || !train_data) {
        return result;
    }
    
    printf("开始随机搜索，评估次数: %d\n", manager->config.num_episodes);
    
    result.num_architectures_tested = manager->config.num_episodes;
    float* accuracies = (float*)calloc(result.num_architectures_tested, sizeof(float));
    
    for (int i = 0; i < result.num_architectures_tested; i++) {
        printf("随机搜索评估 %d/%d\n", i + 1, result.num_architectures_tested);
        
        // 生成随机架构
        architecture_encoding_t arch = generate_random_architecture(&manager->config);
        
        // 评估架构
        float fitness = evaluate_architecture(&arch, train_data, val_data);
        accuracies[i] = fitness;
        
        // 更新最佳结果
        if (fitness > result.best_fitness) {
            result.best_fitness = fitness;
            
            // 复制最佳架构
            if (result.best_architecture.layers) {
                free(result.best_architecture.layers);
            }
            result.best_architecture = arch;
            
            // 需要深拷贝
            result.best_architecture.layers = 
                (layer_encoding_t*)calloc(arch.num_layers, sizeof(layer_encoding_t));
            memcpy(result.best_architecture.layers, arch.layers, 
                   arch.num_layers * sizeof(layer_encoding_t));
        } else {
            // 清理不需要的架构
            if (arch.layers) {
                free(arch.layers);
            }
        }
        
        // 回调进度
        if (manager->progress_callback) {
            int progress = (int)((i + 1) * 100 / result.num_architectures_tested);
            manager->progress_callback(progress, "随机搜索中...");
        }
    }
    
    result.average_accuracy = calculate_average(accuracies, result.num_architectures_tested);
    result.total_evaluations = result.num_architectures_tested;
    
    free(accuracies);
    printf("随机搜索完成，最佳适应度: %.3f\n", result.best_fitness);
    
    return result;
}

// ===========================================
// 其他搜索算法（简化实现）
// ===========================================

int perform_reinforcement_nas(neural_architecture_search_manager_t* manager,
                             const training_data_t* train_data,
                             const training_data_t* val_data) {
    printf("强化学习NAS搜索（简化实现）\n");
    // 在实际实现中，这里应该实现完整的强化学习算法
    return perform_random_search(manager, train_data, val_data);
}

int perform_bayesian_nas(neural_architecture_search_manager_t* manager,
                       const training_data_t* train_data,
                       const training_data_t* val_data) {
    printf("贝叶斯优化NAS搜索（简化实现）\n");
    // 在实际实现中，这里应该实现贝叶斯优化算法
    return perform_random_search(manager, train_data, val_data);
}

int perform_gradient_nas(neural_architecture_search_manager_t* manager,
                        const training_data_t* train_data,
                        const training_data_t* val_data) {
    printf("梯度优化NAS搜索（简化实现）\n");
    // 在实际实现中，这里应该实现基于梯度的架构优化
    return perform_random_search(manager, train_data, val_data);
}

// ===========================================
// 工具函数
// ===========================================

nas_config_t create_default_nas_config(void) {
    nas_config_t config;
    memset(&config, 0, sizeof(nas_config_t));
    
    config.search_method = NAS_SEARCH_EVOLUTIONARY;
    config.max_layers = 10;
    config.min_layers = 3;
    config.max_units_per_layer = 512;
    config.min_units_per_layer = 32;
    config.population_size = 20;
    config.num_generations = 50;
    config.num_episodes = 100;
    config.mutation_rate = 0.1f;
    config.crossover_rate = 0.7f;
    config.learning_rate = 0.01f;
    config.use_skip_connections = true;
    config.use_batch_norm = true;
    config.use_dropout = true;
    config.dropout_rate = 0.3f;
    
    return config;
}

architecture_encoding_t create_simple_architecture(int num_layers, int units_per_layer) {
    architecture_encoding_t arch;
    memset(&arch, 0, sizeof(architecture_encoding_t));
    
    arch.num_layers = num_layers;
    arch.layers = (layer_encoding_t*)calloc(num_layers, sizeof(layer_encoding_t));
    
    if (!arch.layers) {
        arch.num_layers = 0;
        return arch;
    }
    
    for (int i = 0; i < num_layers; i++) {
        arch.layers[i].layer_type = LAYER_SEARCH_DENSE;
        arch.layers[i].units = units_per_layer;
        arch.layers[i].activation = ACTIVATION_SEARCH_RELU;
        arch.layers[i].use_batch_norm = false;
        arch.layers[i].use_dropout = false;
        arch.layers[i].dropout_rate = 0.0f;
    }
    
    arch.complexity = calculate_architecture_complexity(&arch);
    
    return arch;
}

void print_architecture(const architecture_encoding_t* architecture) {
    if (!architecture || architecture->num_layers <= 0) {
        printf("无效的架构\n");
        return;
    }
    
    printf("神经网络架构（层数: %d，复杂度: %.1f）:\n", 
           architecture->num_layers, architecture->complexity);
    
    for (int i = 0; i < architecture->num_layers; i++) {
        const layer_encoding_t* layer = &architecture->layers[i];
        
        printf("  层 %d: ", i + 1);
        
        switch (layer->layer_type) {
            case LAYER_SEARCH_DENSE:
                printf("全连接层");
                break;
            case LAYER_SEARCH_CONV1D:
                printf("1D卷积层");
                break;
            case LAYER_SEARCH_CONV2D:
                printf("2D卷积层");
                break;
            case LAYER_SEARCH_LSTM:
                printf("LSTM层");
                break;
            case LAYER_SEARCH_GRU:
                printf("GRU层");
                break;
            case LAYER_SEARCH_ATTENTION:
                printf("注意力层");
                break;
            case LAYER_SEARCH_TRANSFORMER:
                printf("Transformer层");
                break;
            default:
                printf("未知层类型");
                break;
        }
        
        printf(" (%d单元)", layer->units);
        
        switch (layer->activation) {
            case ACTIVATION_SEARCH_RELU:
                printf(" + ReLU");
                break;
            case ACTIVATION_SEARCH_SIGMOID:
                printf(" + Sigmoid");
                break;
            case ACTIVATION_SEARCH_TANH:
                printf(" + Tanh");
                break;
            case ACTIVATION_SEARCH_LEAKY_RELU:
                printf(" + LeakyReLU");
                break;
            case ACTIVATION_SEARCH_ELU:
                printf(" + ELU");
                break;
            case ACTIVATION_SEARCH_SELU:
                printf(" + SELU");
                break;
            case ACTIVATION_SEARCH_SWISH:
                printf(" + Swish");
                break;
            default:
                printf(" + 未知激活");
                break;
        }
        
        if (layer->use_batch_norm) {
            printf(" + BatchNorm");
        }
        if (layer->use_dropout) {
            printf(" + Dropout(%.2f)", layer->dropout_rate);
        }
        
        printf("\n");
    }
}

// ===========================================
// 回调函数设置
// ===========================================

void set_nas_progress_callback(neural_architecture_search_manager_t* manager,
                              void (*callback)(int progress, const char* message)) {
    if (manager) {
        manager->progress_callback = callback;
    }
}

// ===========================================
// 高级搜索功能
// ===========================================

int perform_multi_objective_nas(neural_architecture_search_manager_t* manager,
                               const training_data_t* train_data,
                               const training_data_t* val_data,
                               float accuracy_weight,
                               float complexity_weight,
                               float latency_weight) {
    if (!manager || !train_data) {
        return -1;
    }
    
    printf("开始多目标NAS搜索，权重: 准确率=%.2f, 复杂度=%.2f, 延迟=%.2f\n",
           accuracy_weight, complexity_weight, latency_weight);
    
    // 保存原始权重
    float original_weights[3] = {1.0f, 0.001f, 0.01f};
    
    // 临时修改适应度函数权重
    // 在实际实现中，这里应该实现更复杂的多目标优化算法
    
    // 执行搜索
    nas_search_result_t result = perform_architecture_search(manager, train_data, val_data);
    
    printf("多目标NAS搜索完成\n");
    return 0;
}

int perform_transfer_nas(neural_architecture_search_manager_t* manager,
                        const training_data_t* source_data,
                        const training_data_t* target_data,
                        const architecture_encoding_t* source_architecture) {
    if (!manager || !source_data || !target_data) {
        return -1;
    }
    
    printf("开始迁移NAS搜索\n");
    
    // 在实际实现中，这里应该:
    // 1. 使用源架构作为起点
    // 2. 在目标数据上进行微调搜索
    // 3. 利用迁移学习技术
    
    // 简化实现：直接使用源架构作为起点进行搜索
    if (source_architecture && source_architecture->num_layers > 0) {
        printf("使用预训练架构作为起点，层数: %d\n", source_architecture->num_layers);
    }
    
    nas_search_result_t result = perform_architecture_search(manager, target_data, target_data);
    
    printf("迁移NAS搜索完成\n");
    return 0;
}

// ===========================================
// 架构分析和可视化
// ===========================================

void analyze_architecture_complexity(const architecture_encoding_t* architecture) {
    if (!architecture || architecture->num_layers <= 0) {
        printf("无效的架构\n");
        return;
    }
    
    printf("架构复杂度分析:\n");
    printf("总层数: %d\n", architecture->num_layers);
    printf("总复杂度: %.1f\n", architecture->complexity);
    
    int total_units = 0;
    int layers_with_bn = 0;
    int layers_with_dropout = 0;
    
    for (int i = 0; i < architecture->num_layers; i++) {
        const layer_encoding_t* layer = &architecture->layers[i];
        total_units += layer->units;
        
        if (layer->use_batch_norm) {
            layers_with_bn++;
        }
        if (layer->use_dropout) {
            layers_with_dropout++;
        }
    }
    
    printf("总单元数: %d\n", total_units);
    printf("使用批归一化的层数: %d\n", layers_with_bn);
    printf("使用Dropout的层数: %d\n", layers_with_dropout);
    printf("平均每层单元数: %.1f\n", (float)total_units / architecture->num_layers);
}

void visualize_architecture_structure(const architecture_encoding_t* architecture) {
    if (!architecture || architecture->num_layers <= 0) {
        printf("无效的架构\n");
        return;
    }
    
    printf("架构结构可视化:\n");
    printf("┌─────────────────────────────────────────────┐\n");
    
    for (int i = 0; i < architecture->num_layers; i++) {
        const layer_encoding_t* layer = &architecture->layers[i];
        
        printf("│ 层 %2d: ", i + 1);
        
        // 层类型简写
        switch (layer->layer_type) {
            case LAYER_SEARCH_DENSE: printf("Dense"); break;
            case LAYER_SEARCH_CONV1D: printf("Conv1D"); break;
            case LAYER_SEARCH_CONV2D: printf("Conv2D"); break;
            case LAYER_SEARCH_LSTM: printf("LSTM"); break;
            case LAYER_SEARCH_GRU: printf("GRU"); break;
            case LAYER_SEARCH_ATTENTION: printf("Attn"); break;
            case LAYER_SEARCH_TRANSFORMER: printf("Trans"); break;
            default: printf("Unknown"); break;
        }
        
        printf(" (%4d单元) ", layer->units);
        
        // 激活函数简写
        switch (layer->activation) {
            case ACTIVATION_SEARCH_RELU: printf("ReLU"); break;
            case ACTIVATION_SEARCH_SIGMOID: printf("Sig"); break;
            case ACTIVATION_SEARCH_TANH: printf("Tanh"); break;
            case ACTIVATION_SEARCH_LEAKY_RELU: printf("LReLU"); break;
            case ACTIVATION_SEARCH_ELU: printf("ELU"); break;
            case ACTIVATION_SEARCH_SELU: printf("SELU"); break;
            case ACTIVATION_SEARCH_SWISH: printf("Swish"); break;
            default: printf("Act"); break;
        }
        
        if (layer->use_batch_norm) {
            printf(" BN");
        }
        if (layer->use_dropout) {
            printf(" Drop(%.1f)", layer->dropout_rate);
        }
        
        printf(" │\n");
        
        if (i < architecture->num_layers - 1) {
            printf("│                                       ↓ │\n");
        }
    }
    
    printf("└─────────────────────────────────────────────┘\n");
}

// ===========================================
// 搜索状态和结果管理
// ===========================================

int get_nas_search_status(const neural_architecture_search_manager_t* manager) {
    if (!manager) {
        return NAS_STATUS_ERROR;
    }
    
    if (manager->is_searching) {
        return NAS_STATUS_SEARCHING;
    } else if (manager->search_result.best_fitness > -FLT_MAX) {
        return NAS_STATUS_COMPLETED;
    } else {
        return NAS_STATUS_READY;
    }
}

const nas_search_result_t* get_nas_search_result(const neural_architecture_search_manager_t* manager) {
    if (!manager) {
        return NULL;
    }
    
    return &manager->search_result;
}

int save_nas_architecture(const architecture_encoding_t* architecture, const char* filename) {
    if (!architecture || !filename) {
        return -1;
    }
    
    printf("保存架构到文件: %s\n", filename);
    
    // 在实际实现中，这里应该将架构保存到文件
    // 简化实现：只打印信息
    
    printf("架构保存完成（简化实现）\n");
    return 0;
}

architecture_encoding_t load_nas_architecture(const char* filename) {
    architecture_encoding_t arch;
    memset(&arch, 0, sizeof(architecture_encoding_t));
    
    if (!filename) {
        return arch;
    }
    
    printf("从文件加载架构: %s\n", filename);
    
    // 在实际实现中，这里应该从文件加载架构
    // 简化实现：返回一个默认架构
    
    arch = create_simple_architecture(3, 128);
    printf("架构加载完成（简化实现）\n");
    
    return arch;
}

// ===========================================
// 性能优化和实用工具
// ===========================================

void optimize_nas_search_performance(neural_architecture_search_manager_t* manager,
                                   int max_parallel_evaluations) {
    if (!manager) {
        return;
    }
    
    printf("优化NAS搜索性能，最大并行评估数: %d\n", max_parallel_evaluations);
    
    // 在实际实现中，这里应该实现并行评估优化
    // 简化实现：只记录配置
    
    manager->config.max_parallel_evaluations = max_parallel_evaluations;
}

int compare_architectures(const architecture_encoding_t* arch1,
                         const architecture_encoding_t* arch2) {
    if (!arch1 || !arch2) {
        return 0;
    }
    
    // 比较复杂度
    if (arch1->complexity < arch2->complexity) {
        return -1;
    } else if (arch1->complexity > arch2->complexity) {
        return 1;
    }
    
    // 比较层数
    if (arch1->num_layers < arch2->num_layers) {
        return -1;
    } else if (arch1->num_layers > arch2->num_layers) {
        return 1;
    }
    
    return 0;
}

void benchmark_nas_search_methods(neural_architecture_search_manager_t* manager,
                                 const training_data_t* train_data,
                                 const training_data_t* val_data) {
    if (!manager || !train_data) {
        return;
    }
    
    printf("开始NAS搜索方法基准测试\n");
    
    nas_search_method_t methods[] = {
        NAS_SEARCH_RANDOM,
        NAS_SEARCH_EVOLUTIONARY,
        NAS_SEARCH_REINFORCEMENT,
        NAS_SEARCH_BAYESIAN,
        NAS_SEARCH_GRADIENT
    };
    
    const char* method_names[] = {
        "随机搜索",
        "进化算法",
        "强化学习",
        "贝叶斯优化",
        "梯度优化"
    };
    
    int num_methods = sizeof(methods) / sizeof(methods[0]);
    
    for (int i = 0; i < num_methods; i++) {
        printf("\n=== 测试方法: %s ===\n", method_names[i]);
        
        // 设置搜索方法
        manager->config.search_method = methods[i];
        
        // 执行搜索
        clock_t start_time = clock();
        nas_search_result_t result = perform_architecture_search(manager, train_data, val_data);
        clock_t end_time = clock();
        
        float search_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
        
        printf("结果: 最佳适应度=%.3f, 搜索时间=%.2fs, 评估次数=%d\n",
               result.best_fitness, search_time, result.total_evaluations);
    }
    
    printf("\nNAS搜索方法基准测试完成\n");
}