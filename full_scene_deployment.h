#ifndef FULL_SCENE_DEPLOYMENT_H
#define FULL_SCENE_DEPLOYMENT_H

#include "dynamic_graph.h"
#include "static_graph_optimizer.h"
#include "distributed_training.h"
#include "high_performance_kernels.h"
#include "chinese_nlp_support.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// 部署场景类型
typedef enum {
    DEPLOYMENT_CLOUD = 0,           // 云端部署
    DEPLOYMENT_EDGE,                // 边缘部署
    DEPLOYMENT_MOBILE,              // 移动端部署
    DEPLOYMENT_EMBEDDED,            // 嵌入式部署
    DEPLOYMENT_IOT,                 // IoT设备部署
    DEPLOYMENT_WEB,                 // Web部署
    DEPLOYMENT_DESKTOP,             // 桌面应用部署
    DEPLOYMENT_HPC,                 // 高性能计算集群
    DEPLOYMENT_HYBRID               // 混合部署
} DeploymentScenario;

// 目标硬件平台
typedef enum {
    HARDWARE_CPU_X86 = 0,         // x86 CPU
    HARDWARE_CPU_ARM,               // ARM CPU
    HARDWARE_CPU_RISCV,             // RISC-V CPU
    HARDWARE_GPU_CUDA,            // NVIDIA GPU
    HARDWARE_GPU_ROCM,              // AMD GPU
    HARDWARE_GPU_INTEL,             // Intel GPU
    HARDWARE_ASCEND,                // 华为昇腾
    HARDWARE_CAMBRICON,             // 寒武纪
    HARDWARE_BITMAIN,               // 比特大陆
    HARDWARE_FPGA,                  // FPGA
    HARDWARE_DSP,                     // DSP
    HARDWARE_MICROCONTROLLER        // 微控制器
} HardwarePlatform;

// 操作系统类型
typedef enum {
    OS_LINUX = 0,                   // Linux
    OS_WINDOWS,                     // Windows
    OS_MACOS,                       // macOS
    OS_ANDROID,                     // Android
    OS_IOS,                         // iOS
    OS_RTOS,                        // 实时操作系统
    OS_EMBEDDED,                    // 嵌入式OS
    OS_WEBASSEMBLY,                 // WebAssembly
    OS_DOCKER,                      // Docker容器
    OS_KUBERNETES                   // Kubernetes
} OperatingSystem;

// 推理引擎类型
typedef enum {
    INFERENCE_ENGINE_NATIVE = 0,    // 原生引擎
    INFERENCE_ENGINE_TENSORRT,      // TensorRT
    INFERENCE_ENGINE_TFLITE,        // TensorFlow Lite
    INFERENCE_ENGINE_ONNXRUNTIME,   // ONNX Runtime
    INFERENCE_ENGINE_OPENVINO,      // OpenVINO
    INFERENCE_ENGINE_COREML,        // CoreML
    INFERENCE_ENGINE_NNAPI,         // NNAPI
    INFERENCE_ENGINE_MINDSPORE_LITE, // MindSpore Lite
    INFERENCE_ENGINE_TVM,           // TVM
    INFERENCE_ENGINE_XLA,           // XLA
    INFERENCE_ENGINE_CUSTOM         // 自定义引擎
} InferenceEngine;

// 模型格式
typedef enum {
    MODEL_FORMAT_NATIVE = 0,        // 原生格式
    MODEL_FORMAT_ONNX,              // ONNX
    MODEL_FORMAT_TENSORFLOW,      // TensorFlow
    MODEL_FORMAT_TFLITE,          // TensorFlow Lite
    MODEL_FORMAT_PYTORCH,         // PyTorch
    MODEL_FORMAT_TORCHSCRIPT,     // TorchScript
    MODEL_FORMAT_MINDIR,            // MindIR
    MODEL_FORMAT_CAMBRICON,         // 寒武纪格式
    MODEL_FORMAT_BITMAIN,           // 比特大陆格式
    MODEL_FORMAT_FPGA_BITSTREAM,    // FPGA比特流
    MODEL_FORMAT_WEBASSEMBLY        // WebAssembly
} ModelFormat;

// 量化配置
typedef struct {
    // 量化类型
    bool enable_int8_quantization;
    bool enable_int16_quantization;
    bool enable_float16_quantization;
    bool enable_bfloat16_quantization;
    bool enable_dynamic_quantization;
    bool enable_static_quantization;
    
    // 量化参数
    int quantization_bits;
    double quantization_scale;
    int quantization_zero_point;
    bool enable_symmetric_quantization;
    bool enable_asymmetric_quantization;
    
    // 感知量化训练
    bool enable_quantization_aware_training;
    int quantization_calibration_steps;
    double quantization_error_threshold;
    
    // 混合精度
    bool enable_mixed_precision_inference;
    bool enable_auto_precision_selection;
    
    // 中文优化
    bool enable_chinese_text_quantization;
    bool enable_cjk_character_quantization;
    
} QuantizationConfig;

// 模型压缩配置
typedef struct {
    // 剪枝配置
    bool enable_pruning;
    double pruning_ratio;
    double pruning_threshold;
    bool enable_structured_pruning;
    bool enable_unstructured_pruning;
    bool enable_magnitude_pruning;
    bool enable_gradient_pruning;
    
    // 知识蒸馏
    bool enable_knowledge_distillation;
    double distillation_temperature;
    double distillation_alpha;
    char* teacher_model_path;
    
    // 低秩分解
    bool enable_low_rank_decomposition;
    int low_rank_ratio;
    bool enable_svd_decomposition;
    bool enable_cp_decomposition;
    
    // 哈希技巧
    bool enable_hashing_trick;
    int hashing_bucket_size;
    bool enable_feature_hashing;
    
    // 中文优化
    bool enable_chinese_model_compression;
    bool enable_cjk_character_compression;
    
} ModelCompressionConfig;

// 安全与隐私配置
typedef struct {
    // 差分隐私
    bool enable_differential_privacy;
    double privacy_epsilon;
    double privacy_delta;
    int privacy_max_grad_norm;
    
    // 联邦学习
    bool enable_federated_learning;
    int federated_num_clients;
    int federated_rounds;
    double federated_learning_rate;
    
    // 同态加密
    bool enable_homomorphic_encryption;
    int encryption_key_size;
    char* encryption_scheme;
    
    // 安全多方计算
    bool enable_secure_multi_party_computation;
    int smpc_num_parties;
    char* smpc_protocol;
    
    // 模型水印
    bool enable_model_watermarking;
    char* watermark_key;
    double watermark_strength;
    
    // 中文隐私保护
    bool enable_chinese_text_privacy_protection;
    bool enable_cjk_character_privacy_protection;
    
} SecurityPrivacyConfig;

// 性能优化配置
typedef struct {
    // 内存优化
    bool enable_memory_pooling;
    size_t memory_pool_size;
    bool enable_memory_defragmentation;
    int memory_defragmentation_interval;
    
    // 计算优化
    bool enable_operator_fusion;
    bool enable_kernel_auto_tuning;
    bool enable_parallel_execution;
    int max_parallel_threads;
    
    // I/O优化
    bool enable_async_io;
    bool enable_prefetching;
    int io_buffer_size;
    int prefetch_queue_size;
    
    // 缓存优化
    bool enable_result_caching;
    size_t cache_size;
    int cache_eviction_policy;
    double cache_hit_ratio;
    
    // 网络优化
    bool enable_network_compression;
    bool enable_protocol_optimization;
    int network_timeout;
    
    // 中文优化
    bool enable_chinese_text_performance_opt;
    bool enable_cjk_character_performance_opt;
    
} PerformanceOptimizationConfig;

// 部署配置
typedef struct {
    // 基础配置
    DeploymentScenario scenario;
    HardwarePlatform hardware;
    OperatingSystem os;
    InferenceEngine engine;
    ModelFormat format;
    
    // 优化配置
    QuantizationConfig quantization;
    ModelCompressionConfig compression;
    SecurityPrivacyConfig security;
    PerformanceOptimizationConfig performance;
    
    // 资源限制
    size_t max_memory_usage;
    int max_cpu_cores;
    double max_power_consumption;
    int max_inference_latency_ms;
    double min_throughput_rps;
    
    // 中文配置
    bool enable_chinese_deployment_optimization;
    bool enable_cjk_hardware_acceleration;
    bool enable_chinese_model_zoo;
    
    // 监控配置
    bool enable_performance_monitoring;
    bool enable_resource_monitoring;
    bool enable_error_reporting;
    int monitoring_interval_ms;
    
} DeploymentConfig;

// 全场景部署引擎
typedef struct FullSceneDeploymentEngine {
    // 部署配置
    DeploymentConfig* configs;
    size_t num_configs;
    
    // 模型管理
    DynamicGraph* source_model;
    void** optimized_models;      // 针对不同场景的优化模型
    char** model_paths;
    size_t num_optimized_models;
    
    // 推理引擎
    void** inference_engines;   // 不同推理引擎实例
    InferenceEngine* engine_types;
    size_t num_engines;
    
    // 性能监控
    PerformanceProfile** performance_profiles;
    size_t num_profiles;
    
    // 资源监控
    double* cpu_usage_history;
    double* memory_usage_history;
    double* power_consumption_history;
    size_t monitoring_history_size;
    
    // 中文支持
    ChineseNLPEngine* chinese_engine;
    bool chinese_deployment_enabled;
    
    // 安全模块
    void* security_module;
    void* privacy_module;
    void* watermark_module;
    
    // 部署状态
    bool is_deployed;
    bool is_serving;
    double deployment_time_ms;
    double initialization_time_ms;
    
    // 错误处理
    char error_message[1024];
    int error_code;
    
} FullSceneDeploymentEngine;

// 部署结果
typedef struct {
    bool success;
    char* message;
    double deployment_time_ms;
    size_t model_size_bytes;
    double peak_memory_usage_mb;
    double inference_latency_ms;
    double throughput_rps;
    
    // 中文部署信息
    bool chinese_optimization_applied;
    bool cjk_acceleration_enabled;
    
} DeploymentResult;

// 核心API函数

// 全场景部署引擎管理
FullSceneDeploymentEngine* fs_deployment_engine_create(DeploymentConfig* config);
void fs_deployment_engine_destroy(FullSceneDeploymentEngine* engine);
bool fs_deployment_engine_initialize(FullSceneDeploymentEngine* engine);

// 模型优化与转换
bool fs_optimize_model_for_deployment(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                                    DeploymentScenario scenario);
bool fs_quantize_model(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                      QuantizationConfig* config);
bool fs_compress_model(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                      ModelCompressionConfig* config);
bool fs_convert_model_format(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                            ModelFormat target_format);

// 部署执行
DeploymentResult* fs_deploy_model(FullSceneDeploymentEngine* engine, DynamicGraph* model, 
                                 DeploymentConfig* config);
bool fs_undeploy_model(FullSceneDeploymentEngine* engine);
bool fs_update_deployment(FullSceneDeploymentEngine* engine, DeploymentConfig* new_config);

// 推理服务
bool fs_start_inference_service(FullSceneDeploymentEngine* engine);
bool fs_stop_inference_service(FullSceneDeploymentEngine* engine);
DynamicTensor* fs_inference(FullSceneDeploymentEngine* engine, DynamicTensor* input);
bool fs_batch_inference(FullSceneDeploymentEngine* engine, DynamicTensor** inputs, 
                       size_t num_inputs, DynamicTensor** outputs);

// 性能优化
bool fs_enable_auto_optimization(FullSceneDeploymentEngine* engine);
bool fs_tune_for_latency(FullSceneDeploymentEngine* engine, double target_latency_ms);
bool fs_tune_for_throughput(FullSceneDeploymentEngine* engine, double target_throughput_rps);
bool fs_tune_for_memory(FullSceneDeploymentEngine* engine, size_t max_memory_mb);

// 中文部署优化
bool fs_enable_chinese_deployment(FullSceneDeploymentEngine* engine);
bool fs_optimize_chinese_model(FullSceneDeploymentEngine* engine, DynamicGraph* model);
bool fs_enable_cjk_hardware_acceleration(FullSceneDeploymentEngine* engine);
bool fs_deploy_chinese_model_zoo(FullSceneDeploymentEngine* engine);

// 安全与隐私
bool fs_enable_security_features(FullSceneDeploymentEngine* engine, SecurityPrivacyConfig* config);
bool fs_add_model_watermark(FullSceneDeploymentEngine* engine, const char* watermark_key);
bool fs_verify_model_integrity(FullSceneDeploymentEngine* engine);
bool fs_enable_privacy_protection(FullSceneDeploymentEngine* engine);

// 监控与诊断
bool fs_enable_monitoring(FullSceneDeploymentEngine* engine);
bool fs_collect_performance_metrics(FullSceneDeploymentEngine* engine);
bool fs_export_deployment_report(FullSceneDeploymentEngine* engine, const char* filename);
bool fs_generate_deployment_diagnostic(FullSceneDeploymentEngine* engine);

// 多场景支持
bool fs_deploy_to_cloud(FullSceneDeploymentEngine* engine, const char* cloud_provider);
bool fs_deploy_to_edge(FullSceneDeploymentEngine* engine, const char* edge_device);
bool fs_deploy_to_mobile(FullSceneDeploymentEngine* engine, MobileConfig* mobile_config);
bool fs_deploy_to_embedded(FullSceneDeploymentEngine* engine, EmbeddedConfig* embedded_config);
bool fs_deploy_to_iot(FullSceneDeploymentEngine* engine, IoTConfig* iot_config);
bool fs_deploy_to_web(FullSceneDeploymentEngine* engine, WebConfig* web_config);

// 硬件适配
bool fs_optimize_for_ascend(FullSceneDeploymentEngine* engine);
bool fs_optimize_for_cambricon(FullSceneDeploymentEngine* engine);
bool fs_optimize_for_bitmain(FullSceneDeploymentEngine* engine);
bool fs_optimize_for_fpga(FullSceneDeploymentEngine* engine);
bool fs_optimize_for_arm(FullSceneDeploymentEngine* engine);
bool fs_optimize_for_riscv(FullSceneDeploymentEngine* engine);

// 推理引擎管理
bool fs_setup_tensorrt(FullSceneDeploymentEngine* engine);
bool fs_setup_tflite(FullSceneDeploymentEngine* engine);
bool fs_setup_onnxruntime(FullSceneDeploymentEngine* engine);
bool fs_setup_openvino(FullSceneDeploymentEngine* engine);
bool fs_setup_coreml(FullSceneDeploymentEngine* engine);
bool fs_setup_mindspore_lite(FullSceneDeploymentEngine* engine);

// 错误处理（中文支持）
const char* fs_deployment_error_string(int error_code);
const char* fs_deployment_error_string_chinese(int error_code);
bool fs_deployment_set_error_handler(FullSceneDeploymentEngine* engine, void (*handler)(int, const char*));

// 高级功能
bool fs_enable_auto_scaling(FullSceneDeploymentEngine* engine);
bool fs_enable_load_balancing(FullSceneDeploymentEngine* engine);
bool fs_enable_fault_tolerance(FullSceneDeploymentEngine* engine);
bool fs_enable_rolling_updates(FullSceneDeploymentEngine* engine);

// 兼容性支持
bool fs_pytorch_compatible_deployment(FullSceneDeploymentEngine* engine);
bool fs_tensorflow_compatible_deployment(FullSceneDeploymentEngine* engine);
bool fs_onnx_compatible_deployment(FullSceneDeploymentEngine* engine);
bool fs_mindspore_compatible_deployment(FullSceneDeploymentEngine* engine);

// 调试支持
bool fs_enable_deployment_debug(FullSceneDeploymentEngine* engine);
bool fs_dump_deployment_config(FullSceneDeploymentEngine* engine, const char* filename);
bool fs_dump_optimized_model(FullSceneDeploymentEngine* engine, const char* filename);
bool fs_visualize_deployment_pipeline(FullSceneDeploymentEngine* engine, const char* filename);

// 移动端配置
typedef struct {
    bool enable_mobile_optimization;
    bool enable_quantization;
    bool enable_pruning;
    bool enable_knowledge_distillation;
    size_t max_model_size;
    int max_inference_time_ms;
    bool enable_hardware_acceleration;
    bool enable_chinese_mobile_support;
} MobileConfig;

// 嵌入式配置
typedef struct {
    bool enable_embedded_optimization;
    size_t max_memory_usage;
    int max_cpu_frequency;
    bool enable_low_power_mode;
    bool enable_real_time_inference;
    int inference_priority;
    bool enable_chinese_embedded_support;
} EmbeddedConfig;

// IoT配置
typedef struct {
    bool enable_iot_optimization;
    size_t max_flash_size;
    size_t max_ram_size;
    bool enable_sensor_integration;
    bool enable_edge_computing;
    bool enable_federated_learning;
    bool enable_chinese_iot_support;
} IoTConfig;

// Web配置
typedef struct {
    bool enable_web_optimization;
    bool enable_webassembly;
    bool enable_webgl_acceleration;
    bool enable_web_worker;
    bool enable_service_worker;
    bool enable_pwa_support;
    bool enable_chinese_web_support;
    size_t max_bundle_size;
} WebConfig;

#endif // FULL_SCENE_DEPLOYMENT_H