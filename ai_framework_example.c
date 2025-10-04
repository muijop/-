#include "ai_framework_main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ===========================================
// 示例数据生成函数
// ===========================================

static float* generate_sample_data(size_t num_samples, size_t input_dim, size_t output_dim) {
    float* data = (float*)malloc(num_samples * input_dim * sizeof(float));
    if (!data) return NULL;
    
    // 生成简单的线性可分数据
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < input_dim; j++) {
            data[i * input_dim + j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    
    return data;
}

static float* generate_sample_labels(const float* input_data, 
                                    size_t num_samples, 
                                    size_t input_dim, 
                                    size_t output_dim) {
    float* labels = (float*)malloc(num_samples * output_dim * sizeof(float));
    if (!labels) return NULL;
    
    // 生成简单的线性分类标签
    for (size_t i = 0; i < num_samples; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < input_dim; j++) {
            sum += input_data[i * input_dim + j];
        }
        
        // 二分类问题
        labels[i * output_dim] = sum > 0 ? 1.0f : 0.0f;
        if (output_dim > 1) {
            labels[i * output_dim + 1] = sum <= 0 ? 1.0f : 0.0f;
        }
    }
    
    return labels;
}

// ===========================================
// 示例1：标准训练演示
// ===========================================

static void example_standard_training(void) {
    printf("=== 示例1：标准训练演示 ===\n");
    
    // 创建AI框架管理器
    ai_framework_manager_t* manager = create_ai_framework_manager();
    if (!manager) {
        printf("错误：无法创建AI框架管理器\n");
        return;
    }
    
    // 生成示例数据
    size_t num_samples = 1000;
    size_t input_dim = 10;
    size_t output_dim = 2;
    
    float* input_data = generate_sample_data(num_samples, input_dim, output_dim);
    float* target_data = generate_sample_labels(input_data, num_samples, input_dim, output_dim);
    
    if (!input_data || !target_data) {
        printf("错误：无法生成示例数据\n");
        destroy_ai_framework_manager(manager);
        return;
    }
    
    // 设置训练配置
    training_config_t config;
    config.training_mode = TRAINING_MODE_STANDARD;
    config.model_type = MODEL_TYPE_FEEDFORWARD;
    config.optimizer_type = OPTIMIZER_ADAM;
    config.learning_rate = 0.001f;
    config.batch_size = 32;
    config.epochs = 10;
    config.use_early_stopping = true;
    config.patience = 5;
    config.validation_split = 0.2f;
    
    // 设置训练数据
    training_data_t data;
    data.input_data = input_data;
    data.target_data = target_data;
    data.data_size = num_samples;
    data.input_dim = input_dim;
    data.output_dim = output_dim;
    
    // 执行训练
    training_result_t* result = train_model(manager, &config, &data);
    
    if (result && result->success) {
        printf("训练成功完成！\n");
        printf("最终损失: %.4f\n", result->final_loss);
        printf("最终准确率: %.2f%%\n", result->final_accuracy * 100);
        printf("训练时间: %zu ms\n", result->training_time_ms);
    } else {
        printf("训练失败: %s\n", result ? result->error_message : "未知错误");
    }
    
    // 清理资源
    if (result) destroy_training_result(result);
    free(input_data);
    free(target_data);
    destroy_ai_framework_manager(manager);
    
    printf("示例1完成\n\n");
}

// ===========================================
// 示例2：超参数优化训练演示
// ===========================================

static void example_hyperparameter_optimization(void) {
    printf("=== 示例2：超参数优化训练演示 ===\n");
    
    // 创建AI框架管理器
    ai_framework_manager_t* manager = create_ai_framework_manager();
    if (!manager) {
        printf("错误：无法创建AI框架管理器\n");
        return;
    }
    
    // 生成示例数据
    size_t num_samples = 1000;
    size_t input_dim = 10;
    size_t output_dim = 2;
    
    float* input_data = generate_sample_data(num_samples, input_dim, output_dim);
    float* target_data = generate_sample_labels(input_data, num_samples, input_dim, output_dim);
    
    if (!input_data || !target_data) {
        printf("错误：无法生成示例数据\n");
        destroy_ai_framework_manager(manager);
        return;
    }
    
    // 设置训练配置（使用超参数优化）
    training_config_t config;
    config.training_mode = TRAINING_MODE_HYPERPARAM_OPTIMIZATION;
    config.model_type = MODEL_TYPE_FEEDFORWARD;
    config.optimizer_type = OPTIMIZER_ADAM; // 这个会被优化覆盖
    config.learning_rate = 0.001f; // 这个会被优化覆盖
    config.batch_size = 32; // 这个会被优化覆盖
    config.epochs = 10;
    config.use_early_stopping = true;
    config.patience = 5;
    config.validation_split = 0.2f;
    
    // 设置训练数据
    training_data_t data;
    data.input_data = input_data;
    data.target_data = target_data;
    data.data_size = num_samples;
    data.input_dim = input_dim;
    data.output_dim = output_dim;
    
    // 执行训练
    training_result_t* result = train_model(manager, &config, &data);
    
    if (result && result->success) {
        printf("超参数优化训练成功完成！\n");
        printf("最终损失: %.4f\n", result->final_loss);
        printf("最终准确率: %.2f%%\n", result->final_accuracy * 100);
        printf("训练时间: %zu ms\n", result->training_time_ms);
        
        // 显示超参数优化信息
        if (result->hyperparameter_optimization_info) {
            printf("使用了超参数优化技术\n");
        }
    } else {
        printf("训练失败: %s\n", result ? result->error_message : "未知错误");
    }
    
    // 清理资源
    if (result) destroy_training_result(result);
    free(input_data);
    free(target_data);
    destroy_ai_framework_manager(manager);
    
    printf("示例2完成\n\n");
}

// ===========================================
// 示例3：模型部署演示
// ===========================================

static void example_model_deployment(void) {
    printf("=== 示例3：模型部署演示 ===\n");
    
    // 创建AI框架管理器
    ai_framework_manager_t* manager = create_ai_framework_manager();
    if (!manager) {
        printf("错误：无法创建AI框架管理器\n");
        return;
    }
    
    // 生成示例数据
    size_t num_samples = 100;
    size_t input_dim = 10;
    size_t output_dim = 2;
    
    float* input_data = generate_sample_data(num_samples, input_dim, output_dim);
    float* target_data = generate_sample_labels(input_data, num_samples, input_dim, output_dim);
    
    if (!input_data || !target_data) {
        printf("错误：无法生成示例数据\n");
        destroy_ai_framework_manager(manager);
        return;
    }
    
    // 首先训练一个简单模型
    training_config_t config;
    config.training_mode = TRAINING_MODE_STANDARD;
    config.model_type = MODEL_TYPE_FEEDFORWARD;
    config.optimizer_type = OPTIMIZER_ADAM;
    config.learning_rate = 0.001f;
    config.batch_size = 32;
    config.epochs = 5;
    config.use_early_stopping = false;
    config.patience = 5;
    config.validation_split = 0.2f;
    
    training_data_t data;
    data.input_data = input_data;
    data.target_data = target_data;
    data.data_size = num_samples;
    data.input_dim = input_dim;
    data.output_dim = output_dim;
    
    training_result_t* result = train_model(manager, &config, &data);
    
    if (!result || !result->success) {
        printf("错误：模型训练失败\n");
        if (result) destroy_training_result(result);
        free(input_data);
        free(target_data);
        destroy_ai_framework_manager(manager);
        return;
    }
    
    printf("模型训练成功，开始部署...\n");
    
    // 部署模型到云端环境
    deployment_config_t deploy_config;
    deploy_config.environment = DEPLOYMENT_CLOUD;
    deploy_config.max_memory_mb = 1024;
    deploy_config.max_storage_mb = 512;
    deploy_config.max_computation_ms = 1000;
    deploy_config.power_consumption_limit = 10.0f;
    deploy_config.use_quantization = true;
    deploy_config.use_pruning = false;
    deploy_config.use_compression = true;
    deploy_config.use_caching = true;
    deploy_config.specific.cloud.instance_type = 1;
    deploy_config.specific.cloud.auto_scaling = true;
    deploy_config.specific.cloud.min_instances = 1;
    deploy_config.specific.cloud.max_instances = 10;
    deploy_config.specific.cloud.cpu_threshold = 80.0f;
    deploy_config.specific.cloud.memory_threshold = 85.0f;
    deploy_config.use_model_optimization = true;
    deploy_config.use_hardware_acceleration = true;
    deploy_config.use_dynamic_loading = false;
    deploy_config.use_progressive_loading = true;
    deploy_config.max_concurrent_requests = 100;
    deploy_config.request_timeout_ms = 5000;
    deploy_config.retry_attempts = 3;
    deploy_config.use_fault_tolerance = true;
    deploy_config.use_monitoring = true;
    deploy_config.use_logging = true;
    
    deployment_result_t* deploy_result = deploy_model_with_framework(
        manager, DEPLOYMENT_CLOUD, &deploy_config);
    
    if (deploy_result && deploy_result->success) {
        printf("模型部署成功！\n");
        printf("部署时间: %zu ms\n", deploy_result->deployment_time_ms);
        printf("优化后模型大小: %zu bytes\n", deploy_result->optimized_model_size_bytes);
        printf("压缩比: %.2f\n", deploy_result->compression_ratio);
        printf("平均延迟: %.2f ms\n", deploy_result->average_latency_ms);
        printf("最大吞吐量: %.2f rps\n", deploy_result->max_throughput_rps);
        printf("部署成本: $%.2f\n", deploy_result->deployment_cost);
        printf("运营成本: $%.2f/小时\n", deploy_result->operation_cost_per_hour);
    } else {
        printf("模型部署失败: %s\n", 
               deploy_result ? deploy_result->error_message : "未知错误");
    }
    
    // 清理资源
    if (deploy_result) destroy_deployment_result(deploy_result);
    destroy_training_result(result);
    free(input_data);
    free(target_data);
    destroy_ai_framework_manager(manager);
    
    printf("示例3完成\n\n");
}

// ===========================================
// 示例4：模型推理演示
// ===========================================

static void example_model_inference(void) {
    printf("=== 示例4：模型推理演示 ===\n");
    
    // 创建AI框架管理器
    ai_framework_manager_t* manager = create_ai_framework_manager();
    if (!manager) {
        printf("错误：无法创建AI框架管理器\n");
        return;
    }
    
    // 生成示例数据
    size_t num_samples = 10; // 少量样本用于推理演示
    size_t input_dim = 10;
    size_t output_dim = 2;
    
    float* input_data = generate_sample_data(num_samples, input_dim, output_dim);
    
    if (!input_data) {
        printf("错误：无法生成示例数据\n");
        destroy_ai_framework_manager(manager);
        return;
    }
    
    // 创建推理请求
    inference_request_t* request = 
        (inference_request_t*)calloc(1, sizeof(inference_request_t));
    if (!request) {
        printf("错误：无法创建推理请求\n");
        free(input_data);
        destroy_ai_framework_manager(manager);
        return;
    }
    
    request->request_id = 1;
    request->input_data = input_data;
    request->input_size = num_samples * input_dim;
    request->output_buffer = malloc(num_samples * output_dim * sizeof(float));
    request->output_size = num_samples * output_dim;
    request->timeout_ms = 5000;
    request->submission_time = (double)time(NULL);
    
    // 执行推理
    inference_result_t* result = perform_inference_with_framework(manager, request);
    
    if (result && result->success) {
        printf("推理成功完成！\n");
        printf("推理时间: %.2f ms\n", result->inference_time_ms);
        printf("置信度: %.2f\n", result->confidence);
        printf("预处理时间: %.2f ms\n", result->preprocessing_time_ms);
        printf("后处理时间: %.2f ms\n", result->postprocessing_time_ms);
        printf("内存使用: %zu bytes\n", result->memory_used_bytes);
    } else {
        printf("推理失败: %s\n", result ? result->error_message : "未知错误");
    }
    
    // 清理资源
    if (result) destroy_inference_result(result);
    destroy_inference_request(request);
    free(input_data);
    destroy_ai_framework_manager(manager);
    
    printf("示例4完成\n\n");
}

// ===========================================
// 示例5：框架状态监控演示
// ===========================================

static void example_framework_monitoring(void) {
    printf("=== 示例5：框架状态监控演示 ===\n");
    
    // 创建AI框架管理器
    ai_framework_manager_t* manager = create_ai_framework_manager();
    if (!manager) {
        printf("错误：无法创建AI框架管理器\n");
        return;
    }
    
    // 设置自定义配置
    ai_framework_config_t custom_config = get_default_framework_config();
    custom_config.framework_mode = FRAMEWORK_MODE_HIGH_PERFORMANCE;
    custom_config.use_gpu_acceleration = true;
    custom_config.use_distributed_training = true;
    custom_config.max_memory_usage_mb = 8192;
    custom_config.max_computation_time_ms = 120000;
    
    // 应用配置
    configure_ai_framework(manager, &custom_config);
    
    // 显示框架状态
    print_framework_status(manager);
    
    // 清理资源
    destroy_ai_framework_manager(manager);
    
    printf("示例5完成\n\n");
}

// ===========================================
// 示例6：多模式训练演示
// ===========================================

static void example_multi_mode_training(void) {
    printf("=== 示例6：多模式训练演示 ===\n");
    
    // 创建AI框架管理器
    ai_framework_manager_t* manager = create_ai_framework_manager();
    if (!manager) {
        printf("错误：无法创建AI框架管理器\n");
        return;
    }
    
    // 生成示例数据
    size_t num_samples = 500;
    size_t input_dim = 10;
    size_t output_dim = 2;
    
    float* input_data = generate_sample_data(num_samples, input_dim, output_dim);
    float* target_data = generate_sample_labels(input_data, num_samples, input_dim, output_dim);
    
    if (!input_data || !target_data) {
        printf("错误：无法生成示例数据\n");
        destroy_ai_framework_manager(manager);
        return;
    }
    
    training_data_t data;
    data.input_data = input_data;
    data.target_data = target_data;
    data.data_size = num_samples;
    data.input_dim = input_dim;
    data.output_dim = output_dim;
    
    // 测试不同的训练模式
    training_mode_t modes[] = {
        TRAINING_MODE_STANDARD,
        TRAINING_MODE_HYPERPARAM_OPTIMIZATION,
        TRAINING_MODE_DISTRIBUTED
    };
    
    const char* mode_names[] = {
        "标准训练",
        "超参数优化训练", 
        "分布式训练"
    };
    
    for (int i = 0; i < 3; i++) {
        printf("--- 测试模式: %s ---\n", mode_names[i]);
        
        training_config_t config;
        config.training_mode = modes[i];
        config.model_type = MODEL_TYPE_FEEDFORWARD;
        config.optimizer_type = OPTIMIZER_ADAM;
        config.learning_rate = 0.001f;
        config.batch_size = 32;
        config.epochs = 5;
        config.use_early_stopping = true;
        config.patience = 3;
        config.validation_split = 0.2f;
        
        training_result_t* result = train_model(manager, &config, &data);
        
        if (result && result->success) {
            printf("%s成功！损失: %.4f, 准确率: %.2f%%, 时间: %zu ms\n",
                   mode_names[i], result->final_loss, result->final_accuracy * 100,
                   result->training_time_ms);
        } else {
            printf("%s失败: %s\n", mode_names[i], 
                   result ? result->error_message : "未知错误");
        }
        
        if (result) destroy_training_result(result);
    }
    
    // 清理资源
    free(input_data);
    free(target_data);
    destroy_ai_framework_manager(manager);
    
    printf("示例6完成\n\n");
}

// ===========================================
// 主函数
// ===========================================

int main(void) {
    printf("=== AI框架综合演示程序 ===\n\n");
    
    // 设置随机种子
    srand((unsigned int)time(NULL));
    
    // 运行各个示例
    example_standard_training();
    example_hyperparameter_optimization();
    example_model_deployment();
    example_model_inference();
    example_framework_monitoring();
    example_multi_mode_training();
    
    printf("=== 所有演示完成 ===\n");
    printf("这个演示展示了AI框架的核心功能：\n");
    printf("1. 多种训练模式（标准、超参数优化、分布式等）\n");
    printf("2. 模型部署到不同环境（云端、边缘、移动端等）\n");
    printf("3. 高性能推理服务\n");
    printf("4. 框架状态监控和配置管理\n");
    printf("5. 多模式训练能力\n");
    printf("\n框架支持的功能模块包括：\n");
    printf("- 超参数优化\n");
    printf("- 元学习\n");
    printf("- 强化学习\n");
    printf("- 图神经网络\n");
    printf("- 时间序列分析\n");
    printf("- 联邦学习\n");
    printf("- 全场景部署\n");
    printf("- 模型压缩\n");
    printf("- 模型解释性\n");
    printf("- 分布式训练\n");
    printf("- 可视化工具\n");
    
    return 0;
}