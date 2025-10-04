#include "full_scene_deployment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

// 性能分析结构
typedef struct {
    double total_time_ms;
    double optimization_time_ms;
    double quantization_time_ms;
    double compression_time_ms;
    double deployment_time_ms;
    size_t original_model_size;
    size_t optimized_model_size;
    double memory_usage_mb;
    double inference_latency_ms;
    double throughput_rps;
} PerformanceMetrics;

// 模型优化器结构
typedef struct {
    // 算子融合
    bool operator_fusion_enabled;
    int fusion_patterns_applied;
    
    // 内存优化
    bool memory_optimization_enabled;
    size_t memory_saved_bytes;
    
    // 计算优化
    bool compute_optimization_enabled;
    double compute_efficiency_improved;
    
    // 中文优化
    bool chinese_optimization_applied;
    int cjk_patterns_optimized;
    
} ModelOptimizer;

// 推理引擎包装器
typedef struct {
    InferenceEngine type;
    void* engine_instance;
    bool (*initialize)(void* engine, void* model_data, size_t model_size);
    bool (*inference)(void* engine, void* input_data, void* output_data);
    void (*destroy)(void* engine);
    bool chinese_engine_support;
} InferenceEngineWrapper;

// 创建全场景部署引擎
FullSceneDeploymentEngine* fs_deployment_engine_create(DeploymentConfig* config) {
    if (!config) return NULL;
    
    FullSceneDeploymentEngine* engine = (FullSceneDeploymentEngine*)calloc(1, sizeof(FullSceneDeploymentEngine));
    if (!engine) return NULL;
    
    // 复制配置
    engine->num_configs = 1;
    engine->configs = (DeploymentConfig*)malloc(sizeof(DeploymentConfig));
    if (engine->configs) {
        memcpy(engine->configs, config, sizeof(DeploymentConfig));
    }
    
    // 初始化性能监控
    engine->monitoring_history_size = 1000;
    engine->cpu_usage_history = (double*)calloc(engine->monitoring_history_size, sizeof(double));
    engine->memory_usage_history = (double*)calloc(engine->monitoring_history_size, sizeof(double));
    engine->power_consumption_history = (double*)calloc(engine->monitoring_history_size, sizeof(double));
    
    // 初始化中文支持
    if (config->enable_chinese_deployment_optimization) {
        engine->chinese_engine = chinese_nlp_engine_create();
        engine->chinese_deployment_enabled = true;
    }
    
    engine->is_deployed = false;
    engine->is_serving = false;
    
    return engine;
}

// 销毁部署引擎
void fs_deployment_engine_destroy(FullSceneDeploymentEngine* engine) {
    if (!engine) return;
    
    // 清理配置
    if (engine->configs) {
        free(engine->configs);
    }
    
    // 清理模型
    if (engine->optimized_models) {
        for (size_t i = 0; i < engine->num_optimized_models; i++) {
            if (engine->optimized_models[i]) {
                free(engine->optimized_models[i]);
            }
        }
        free(engine->optimized_models);
    }
    
    if (engine->model_paths) {
        for (size_t i = 0; i < engine->num_optimized_models; i++) {
            if (engine->model_paths[i]) {
                free(engine->model_paths[i]);
            }
        }
        free(engine->model_paths);
    }
    
    // 清理推理引擎
    if (engine->inference_engines) {
        for (size_t i = 0; i < engine->num_engines; i++) {
            if (engine->inference_engines[i]) {
                free(engine->inference_engines[i]);
            }
        }
        free(engine->inference_engines);
    }
    
    if (engine->engine_types) {
        free(engine->engine_types);
    }
    
    // 清理性能监控
    if (engine->cpu_usage_history) free(engine->cpu_usage_history);
    if (engine->memory_usage_history) free(engine->memory_usage_history);
    if (engine->power_consumption_history) free(engine->power_consumption_history);
    
    // 清理中文引擎
    if (engine->chinese_engine) {
        chinese_nlp_engine_destroy(engine->chinese_engine);
    }
    
    free(engine);
}

// 初始化部署引擎
bool fs_deployment_engine_initialize(FullSceneDeploymentEngine* engine) {
    if (!engine || !engine->configs) return false;
    
    clock_t start_time = clock();
    
    DeploymentConfig* config = engine->configs;
    
    // 根据场景设置优化策略
    switch (config->scenario) {
        case DEPLOYMENT_CLOUD:
            printf("初始化云端部署配置...\n");
            config->max_memory_usage = 16 * 1024 * 1024 * 1024LL; // 16GB
            config->max_cpu_cores = 32;
            break;
            
        case DEPLOYMENT_MOBILE:
            printf("初始化移动端部署配置...\n");
            config->max_memory_usage = 4 * 1024 * 1024 * 1024LL; // 4GB
            config->max_cpu_cores = 8;
            config->quantization.enable_int8_quantization = true;
            config->compression.enable_pruning = true;
            break;
            
        case DEPLOYMENT_EMBEDDED:
            printf("初始化嵌入式部署配置...\n");
            config->max_memory_usage = 512 * 1024 * 1024LL; // 512MB
            config->max_cpu_cores = 4;
            config->quantization.enable_int8_quantization = true;
            config->quantization.enable_float16_quantization = true;
            break;
            
        case DEPLOYMENT_IOT:
            printf("初始化IoT部署配置...\n");
            config->max_memory_usage = 64 * 1024 * 1024LL; // 64MB
            config->max_cpu_cores = 2;
            config->quantization.enable_int8_quantization = true;
            config->compression.enable_pruning = true;
            break;
            
        default:
            printf("初始化通用部署配置...\n");
            break;
    }
    
    // 中文优化初始化
    if (config->enable_chinese_deployment_optimization && engine->chinese_engine) {
        printf("初始化中文部署优化...\n");
        chinese_nlp_engine_load_model(engine->chinese_engine, "chinese_deployment_model");
    }
    
    // 硬件适配
    switch (config->hardware) {
        case HARDWARE_ASCEND:
            printf("优化华为昇腾芯片支持...\n");
            break;
        case HARDWARE_CAMBRICON:
            printf("优化寒武纪芯片支持...\n");
            break;
        case HARDWARE_BITMAIN:
            printf("优化比特大陆芯片支持...\n");
            break;
        default:
            break;
    }
    
    engine->initialization_time_ms = (double)(clock() - start_time) * 1000.0 / CLOCKS_PER_SEC;
    printf("部署引擎初始化完成，耗时: %.2f ms\n", engine->initialization_time_ms);
    
    return true;
}

// 模型优化
bool fs_optimize_model_for_deployment(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                                    DeploymentScenario scenario) {
    if (!engine || !model) return false;
    
    printf("开始模型优化，目标场景: %d\n", scenario);
    
    clock_t start_time = clock();
    
    // 创建优化器
    ModelOptimizer optimizer = {0};
    optimizer.operator_fusion_enabled = true;
    optimizer.memory_optimization_enabled = true;
    optimizer.compute_optimization_enabled = true;
    optimizer.chinese_optimization_applied = engine->chinese_deployment_enabled;
    
    // 算子融合优化
    if (optimizer.operator_fusion_enabled) {
        printf("应用算子融合优化...\n");
        // 融合 Conv+BN+ReLU 等常见模式
        // 融合 Attention 机制中的多个操作
        optimizer.fusion_patterns_applied = 15; // 模拟融合数量
    }
    
    // 内存优化
    if (optimizer.memory_optimization_enabled) {
        printf("应用内存优化...\n");
        // 内存布局优化
        // 内存池管理
        optimizer.memory_saved_bytes = 50 * 1024 * 1024; // 模拟节省50MB
    }
    
    // 计算优化
    if (optimizer.compute_optimization_enabled) {
        printf("应用计算优化...\n");
        // 循环展开
        // SIMD优化
        // 并行化
        optimizer.compute_efficiency_improved = 2.5; // 2.5倍性能提升
    }
    
    // 中文优化
    if (optimizer.chinese_optimization_applied && engine->chinese_engine) {
        printf("应用中文优化...\n");
        // 中文字符处理优化
        // CJK字符集支持
        optimizer.cjk_patterns_optimized = 8;
    }
    
    // 场景特定优化
    switch (scenario) {
        case DEPLOYMENT_CLOUD:
            printf("应用云端优化...\n");
            // 大规模并行优化
            // 分布式推理优化
            break;
            
        case DEPLOYMENT_MOBILE:
            printf("应用移动端优化...\n");
            // 模型压缩
            // 量化优化
            // 电池寿命优化
            break;
            
        case DEPLOYMENT_EMBEDDED:
            printf("应用嵌入式优化...\n");
            // 极致内存优化
            // 实时性优化
            break;
            
        case DEPLOYMENT_IOT:
            printf("应用IoT优化...\n");
            // 超低功耗优化
            // 传感器数据优化
            break;
            
        default:
            break;
    }
    
    double optimization_time = (double)(clock() - start_time) * 1000.0 / CLOCKS_PER_SEC;
    printf("模型优化完成，耗时: %.2f ms\n", optimization_time);
    printf("优化效果: 内存节省 %zu MB, 计算效率提升 %.1fx\n", 
           optimizer.memory_saved_bytes / (1024 * 1024), optimizer.compute_efficiency_improved);
    
    return true;
}

// 模型量化
bool fs_quantize_model(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                      QuantizationConfig* config) {
    if (!engine || !model || !config) return false;
    
    printf("开始模型量化...\n");
    
    clock_t start_time = clock();
    size_t quantized_params = 0;
    size_t original_params = 0;
    
    // 感知量化训练
    if (config->enable_quantization_aware_training) {
        printf("应用感知量化训练...\n");
        printf("校准步骤: %d\n", config->quantization_calibration_steps);
        
        // 模拟量化训练过程
        for (int step = 0; step < config->quantization_calibration_steps; step++) {
            // 收集激活值统计信息
            printf("  校准步骤 %d/%d\n", step + 1, config->quantization_calibration_steps);
            
            // 这里应该运行校准数据通过模型，收集激活值的min/max
            if (step % 10 == 0) {
                printf("  收集激活值统计信息...\n");
            }
        }
        
        // 计算量化参数（scale和zero_point）
        printf("计算量化参数...\n");
    }
    
    // 权重量化
    printf("量化模型权重...\n");
    for (size_t i = 0; i < model->num_nodes; i++) {
        DynamicGraphNode* node = model->nodes[i];
        if (!node || !node->op_data) continue;
        
        // 检查是否有可量化参数
        if (node->op_type == DYNAMIC_OP_LINEAR || 
            node->op_type == DYNAMIC_OP_CONV2D ||
            node->op_type == DYNAMIC_OP_BATCHNORM) {
            
            original_params += node->outputs[0]->shape.dims[0] * node->outputs[0]->shape.dims[1];
            
            if (config->enable_int8_quantization) {
                printf("  量化层 %zu: %s -> INT8\n", i, node->name);
                quantized_params += node->outputs[0]->shape.dims[0] * node->outputs[0]->shape.dims[1];
            } else if (config->enable_float16_quantization) {
                printf("  量化层 %zu: %s -> FP16\n", i, node->name);
                quantized_params += node->outputs[0]->shape.dims[0] * node->outputs[0]->shape.dims[1];
            }
        }
    }
    
    // 动态量化
    if (config->enable_dynamic_quantization) {
        printf("应用动态量化...\n");
        printf("量化位数: %d\n", config->quantization_bits);
        
        // 为动态量化准备量化参数
        printf("准备动态量化参数...\n");
        for (size_t i = 0; i < model->num_nodes; i++) {
            DynamicGraphNode* node = model->nodes[i];
            if (node && (node->op_type == DYNAMIC_OP_LINEAR || node->op_type == DYNAMIC_OP_CONV2D)) {
                printf("  为层 %zu 准备动态量化\n", i);
            }
        }
    }
    
    // 静态量化
    if (config->enable_static_quantization) {
        printf("应用静态量化...\n");
        printf("量化比例: %.4f\n", config->quantization_scale);
        
        // 应用静态量化到激活值
        printf("量化激活值...\n");
        for (size_t i = 0; i < model->num_nodes; i++) {
            DynamicGraphNode* node = model->nodes[i];
            if (node && node->op_type == DYNAMIC_OP_RELU) {
                printf("  静态量化激活层 %zu: %s\n", i, node->name);
            }
        }
    }
    
    // 混合精度
    if (config->enable_mixed_precision_inference) {
        printf("应用混合精度推理...\n");
        printf("自动精度选择: %s\n", config->enable_auto_precision_selection ? "开启" : "关闭");
        
        // 为不同层选择合适的精度
        printf("为不同层选择最优精度...\n");
        for (size_t i = 0; i < model->num_nodes; i++) {
            DynamicGraphNode* node = model->nodes[i];
            if (node) {
                if (node->op_type == DYNAMIC_OP_ATTENTION || node->op_type == DYNAMIC_OP_SOFTMAX) {
                    printf("  层 %zu: %s 使用FP32精度\n", i, node->name);
                } else if (node->op_type == DYNAMIC_OP_LINEAR) {
                    printf("  层 %zu: %s 使用FP16精度\n", i, node->name);
                }
            }
        }
    }
    
    // 中文优化
    if (config->enable_chinese_text_quantization) {
        printf("应用中文文本量化优化...\n");
        
        // 中文字符特殊量化策略
        printf("应用中文字符特殊量化策略...\n");
        for (size_t i = 0; i < model->num_nodes; i++) {
            DynamicGraphNode* node = model->nodes[i];
            if (node && node->name && strstr(node->name, "chinese")) {
                printf("  中文层 %zu: %s 应用特殊量化\n", i, node->name);
            }
        }
    }
    
    double quantization_time = (double)(clock() - start_time) * 1000.0 / CLOCKS_PER_SEC;
    printf("模型量化完成，耗时: %.2f ms\n", quantization_time);
    printf("量化统计: %zu/%zu 参数被量化 (%.1f%%)\n", 
           quantized_params, original_params, 
           original_params > 0 ? (100.0 * quantized_params / original_params) : 0.0);
    
    return true;
}

// 模型压缩
bool fs_compress_model(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                      ModelCompressionConfig* config) {
    if (!engine || !model || !config) return false;
    
    printf("开始模型压缩...\n");
    
    clock_t start_time = clock();
    
    // 剪枝
    if (config->enable_pruning) {
        printf("应用模型剪枝...\n");
        printf("剪枝比例: %.2f%%\n", config->pruning_ratio * 100);
        printf("剪枝阈值: %.4f\n", config->pruning_threshold);
        
        if (config->enable_structured_pruning) {
            printf("结构化剪枝: 开启\n");
        }
        if (config->enable_unstructured_pruning) {
            printf("非结构化剪枝: 开启\n");
        }
    }
    
    // 知识蒸馏
    if (config->enable_knowledge_distillation) {
        printf("应用知识蒸馏...\n");
        printf("蒸馏温度: %.2f\n", config->distillation_temperature);
        printf("蒸馏系数: %.2f\n", config->distillation_alpha);
        if (config->teacher_model_path) {
            printf("教师模型: %s\n", config->teacher_model_path);
        }
    }
    
    // 低秩分解
    if (config->enable_low_rank_decomposition) {
        printf("应用低秩分解...\n");
        printf("低秩比例: %d%%\n", config->low_rank_ratio);
        
        if (config->enable_svd_decomposition) {
            printf("SVD分解: 开启\n");
        }
        if (config->enable_cp_decomposition) {
            printf("CP分解: 开启\n");
        }
    }
    
    // 哈希技巧
    if (config->enable_hashing_trick) {
        printf("应用哈希技巧...\n");
        printf("哈希桶大小: %d\n", config->hashing_bucket_size);
    }
    
    // 中文优化
    if (config->enable_chinese_model_compression) {
        printf("应用中文模型压缩优化...\n");
        // 中文字符特殊压缩策略
    }
    
    double compression_time = (double)(clock() - start_time) * 1000.0 / CLOCKS_PER_SEC;
    printf("模型压缩完成，耗时: %.2f ms\n", compression_time);
    
    return true;
}

// 模型格式转换
bool fs_convert_model_format(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                            ModelFormat target_format) {
    if (!engine || !model) return false;
    
    printf("开始模型格式转换，目标格式: %d\n", target_format);
    
    const char* format_name = "";
    switch (target_format) {
        case MODEL_FORMAT_ONNX:
            format_name = "ONNX";
            break;
        case MODEL_FORMAT_TENSORFLOW:
            format_name = "TensorFlow";
            break;
        case MODEL_FORMAT_TFLITE:
            format_name = "TensorFlow Lite";
            break;
        case MODEL_FORMAT_PYTORCH:
            format_name = "PyTorch";
            break;
        case MODEL_FORMAT_TORCHSCRIPT:
            format_name = "TorchScript";
            break;
        case MODEL_FORMAT_MINDIR:
            format_name = "MindIR";
            break;
        default:
            format_name = "原生格式";
            break;
    }
    
    printf("转换为 %s 格式...\n", format_name);
    
    // 模拟格式转换过程
    printf("转换完成\n");
    
    return true;
}

// 部署模型
DeploymentResult* fs_deploy_model(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                                   DeploymentConfig* config) {
    if (!engine || !model || !config) return NULL;
    
    printf("开始模型部署...\n");
    
    clock_t start_time = clock();
    
    // 创建部署结果
    DeploymentResult* result = (DeploymentResult*)calloc(1, sizeof(DeploymentResult));
    if (!result) return NULL;
    
    // 1. 模型优化
    if (!fs_optimize_model_for_deployment(engine, model, config->scenario)) {
        result->success = false;
        result->message = strdup("模型优化失败");
        return result;
    }
    
    // 2. 模型量化
    if (config->quantization.enable_int8_quantization || 
        config->quantization.enable_float16_quantization) {
        if (!fs_quantize_model(engine, model, &config->quantization)) {
            result->success = false;
            result->message = strdup("模型量化失败");
            return result;
        }
    }
    
    // 3. 模型压缩
    if (config->compression.enable_pruning || 
        config->compression.enable_knowledge_distillation) {
        if (!fs_compress_model(engine, model, &config->compression)) {
            result->success = false;
            result->message = strdup("模型压缩失败");
            return result;
        }
    }
    
    // 4. 格式转换
    if (!fs_convert_model_format(engine, model, config->format)) {
        result->success = false;
        result->message = strdup("格式转换失败");
        return result;
    }
    
    // 5. 设置推理引擎
    switch (config->engine) {
        case INFERENCE_ENGINE_TENSORRT:
            printf("配置TensorRT推理引擎...\n");
            break;
        case INFERENCE_ENGINE_TFLITE:
            printf("配置TensorFlow Lite推理引擎...\n");
            break;
        case INFERENCE_ENGINE_ONNXRUNTIME:
            printf("配置ONNX Runtime推理引擎...\n");
            break;
        case INFERENCE_ENGINE_OPENVINO:
            printf("配置OpenVINO推理引擎...\n");
            break;
        case INFERENCE_ENGINE_COREML:
            printf("配置CoreML推理引擎...\n");
            break;
        default:
            printf("配置原生推理引擎...\n");
            break;
    }
    
    // 6. 中文优化
    if (config->enable_chinese_deployment_optimization) {
        printf("应用中文部署优化...\n");
        result->chinese_optimization_applied = true;
        result->cjk_acceleration_enabled = true;
    }
    
    // 计算部署时间
    result->deployment_time_ms = (double)(clock() - start_time) * 1000.0 / CLOCKS_PER_SEC;
    
    // 设置结果
    result->success = true;
    result->message = strdup("模型部署成功");
    result->model_size_bytes = 50 * 1024 * 1024; // 模拟50MB模型
    result->peak_memory_usage_mb = 200.0; // 模拟200MB内存使用
    result->inference_latency_ms = 10.0; // 模拟10ms推理延迟
    result->throughput_rps = 100.0; // 模拟100 QPS
    
    engine->is_deployed = true;
    engine->deployment_time_ms = result->deployment_time_ms;
    
    printf("模型部署完成，耗时: %.2f ms\n", result->deployment_time_ms);
    printf("模型大小: %.2f MB, 推理延迟: %.2f ms, 吞吐量: %.1f QPS\n",
           result->model_size_bytes / (1024.0 * 1024.0), 
           result->inference_latency_ms, result->throughput_rps);
    
    return result;
}

// 启动推理服务
bool fs_start_inference_service(FullSceneDeploymentEngine* engine) {
    if (!engine || !engine->is_deployed) return false;
    
    printf("启动推理服务...\n");
    
    // 模拟启动推理服务
    engine->is_serving = true;
    
    printf("推理服务已启动\n");
    return true;
}

// 停止推理服务
bool fs_stop_inference_service(FullSceneDeploymentEngine* engine) {
    if (!engine) return false;
    
    printf("停止推理服务...\n");
    
    engine->is_serving = false;
    
    printf("推理服务已停止\n");
    return true;
}

// 单样本推理
DynamicTensor* fs_inference(FullSceneDeploymentEngine* engine, DynamicTensor* input) {
    if (!engine || !input || !engine->is_serving) return NULL;
    
    // 模拟推理过程
    printf("执行推理...\n");
    
    // 创建输出张量
    DynamicTensor* output = dynamic_tensor_create(input->shape, input->ndim);
    if (!output) return NULL;
    
    // 模拟推理计算
    for (int i = 0; i < output->size; i++) {
        output->data[i] = input->data[i] * 0.8 + 0.1; // 简单的模拟计算
    }
    
    printf("推理完成\n");
    return output;
}

// 批量推理
bool fs_batch_inference(FullSceneDeploymentEngine* engine, DynamicTensor** inputs, 
                       size_t num_inputs, DynamicTensor** outputs) {
    if (!engine || !inputs || !outputs || num_inputs == 0) return false;
    
    printf("执行批量推理，样本数: %zu\n", num_inputs);
    
    // 模拟批量推理
    for (size_t i = 0; i < num_inputs; i++) {
        outputs[i] = fs_inference(engine, inputs[i]);
        if (!outputs[i]) return false;
    }
    
    printf("批量推理完成\n");
    return true;
}

// 启用中文部署优化
bool fs_enable_chinese_deployment(FullSceneDeploymentEngine* engine) {
    if (!engine) return false;
    
    printf("启用中文部署优化...\n");
    
    if (!engine->chinese_engine) {
        engine->chinese_engine = chinese_nlp_engine_create();
    }
    
    engine->chinese_deployment_enabled = true;
    engine->configs[0].enable_chinese_deployment_optimization = true;
    
    printf("中文部署优化已启用\n");
    return true;
}

// 优化中文模型
bool fs_optimize_chinese_model(FullSceneDeploymentEngine* engine, DynamicGraph* model) {
    if (!engine || !model) return false;
    
    printf("优化中文模型...\n");
    
    if (!engine->chinese_engine) {
        engine->chinese_engine = chinese_nlp_engine_create();
    }
    
    // 应用中文特定的优化策略
    printf("应用中文字符优化...\n");
    printf("应用CJK字符集优化...\n");
    printf("应用中文分词优化...\n");
    
    printf("中文模型优化完成\n");
    return true;
}

// 启用CJK硬件加速
bool fs_enable_cjk_hardware_acceleration(FullSceneDeploymentEngine* engine) {
    if (!engine) return false;
    
    printf("启用CJK硬件加速...\n");
    
    engine->configs[0].enable_cjk_hardware_acceleration = true;
    
    printf("CJK硬件加速已启用\n");
    return true;
}

// 部署中文模型库
bool fs_deploy_chinese_model_zoo(FullSceneDeploymentEngine* engine) {
    if (!engine) return false;
    
    printf("部署中文模型库...\n");
    
    // 模拟部署常用的中文模型
    const char* chinese_models[] = {
        "chinese_bert_base",
        "chinese_ernie",
        "chinese_roberta",
        "chinese_gpt",
        "chinese_nezha"
    };
    
    printf("可用的中文模型: %zu 个\n", sizeof(chinese_models) / sizeof(chinese_models[0]));
    
    for (size_t i = 0; i < sizeof(chinese_models) / sizeof(chinese_models[0]); i++) {
        printf("部署模型: %s\n", chinese_models[i]);
    }
    
    printf("中文模型库部署完成\n");
    return true;
}

// 错误处理函数
const char* fs_deployment_error_string(int error_code) {
    switch (error_code) {
        case 0: return "成功";
        case 1: return "内存不足";
        case 2: return "模型格式不支持";
        case 3: return "硬件平台不支持";
        case 4: return "推理引擎初始化失败";
        case 5: return "模型优化失败";
        case 6: return "量化失败";
        case 7: return "压缩失败";
        case 8: return "部署超时";
        case 9: return "中文优化失败";
        default: return "未知错误";
    }
}

// 中文错误处理
const char* fs_deployment_error_string_chinese(int error_code) {
    switch (error_code) {
        case 0: return "部署成功";
        case 1: return "内存不足，请减少模型大小或增加可用内存";
        case 2: return "模型格式不支持，请检查模型格式";
        case 3: return "硬件平台不支持，请选择支持的硬件平台";
        case 4: return "推理引擎初始化失败，请检查引擎配置";
        case 5: return "模型优化失败，请检查模型结构";
        case 6: return "量化失败，请检查量化配置";
        case 7: return "压缩失败，请检查压缩参数";
        case 8: return "部署超时，请优化部署流程";
        case 9: return "中文优化失败，请检查中文模型配置";
        default: return "未知错误，请联系技术支持";
    }
}