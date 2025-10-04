#ifndef STATIC_GRAPH_OPTIMIZER_H
#define STATIC_GRAPH_OPTIMIZER_H

#include "static_graph_types.h"
#include "dynamic_graph.h"
#include "tensor_autograd.h"

// 图优化级别
typedef enum {
    OPT_LEVEL_NONE = 0,
    OPT_LEVEL_BASIC = 1,      // 基础优化：死代码消除、常量折叠
    OPT_LEVEL_ADVANCED = 2,   // 高级优化：算子融合、内存布局优化
    OPT_LEVEL_AGGRESSIVE = 3  // 激进优化：循环展开、向量化、并行化
} OptimizationLevel;

// 优化器配置
typedef struct {
    OptimizationLevel level;
    bool enable_fusion;           // 算子融合
    bool enable_memory_opt;       // 内存优化
    bool enable_parallelization;  // 自动并行化
    bool enable_vectorization;    // 向量化
    bool enable_loop_unrolling;  // 循环展开
    bool enable_constant_folding; // 常量折叠
    bool enable_dead_code_elim;  // 死代码消除
    int max_fusion_size;         // 最大融合大小
    int vectorization_width;     // 向量化宽度
    int parallel_threshold;      // 并行化阈值
} GraphOptimizerConfig;

// 静态图编译器（类似TensorFlow XLA）
typedef struct {
    StaticGraph* graph;
    GraphOptimizerConfig config;
    void* compiled_code;       // 编译后的代码
    size_t code_size;
    void* memory_pool;          // 内存池
    size_t pool_size;
    bool is_compiled;
} GraphCompiler;

// 算子融合模式
typedef struct {
    const char* pattern_name;
    OperationType* pattern_ops;
    size_t pattern_length;
    OperationType fused_op;
    bool (*can_fuse)(GraphNode** nodes, size_t num_nodes);
    void (*fuse)(GraphNode** nodes, size_t num_nodes, GraphNode** fused_node);
} FusionPattern;

// 内存布局优化
typedef struct {
    bool enable_nhwc;           // NHWC格式优化（卷积）
    bool enable_nchw;           // NCHW格式优化
    bool enable_blocked;        // 分块布局
    int block_size;             // 分块大小
    bool enable_packing;        // 数据打包
    int pack_size;              // 打包大小
} MemoryLayoutConfig;

// 并行化策略
typedef enum {
    PARALLEL_NONE = 0,
    PARALLEL_DATA,      // 数据并行
    PARALLEL_MODEL,     // 模型并行
    PARALLEL_PIPELINE,  // 流水线并行
    PARALLEL_HYBRID    // 混合并行
} ParallelizationStrategy;

// 编译优化（类似JAX的JIT）
typedef struct {
    bool enable_jit;                    // 启用JIT编译
    bool enable_aot;                    // 启用AOT编译
    bool enable_gpu_codegen;            // GPU代码生成
    bool enable_cpu_codegen;            // CPU代码生成
    int optimization_level;               // 优化级别
    bool enable_autotuning;              // 自动调优
    int num_warmup_runs;                 // 预热运行次数
    bool enable_profile_guided_opt;      // 性能分析引导优化
} JITConfig;

// 静态图优化器
GraphCompiler* graph_compiler_create(DynamicGraph* dynamic_graph, GraphOptimizerConfig* config);
void graph_compiler_destroy(GraphCompiler* compiler);

// 核心优化功能
bool graph_compiler_optimize(GraphCompiler* compiler);
bool graph_compiler_fuse_operators(GraphCompiler* compiler);
bool graph_compiler_optimize_memory_layout(GraphCompiler* compiler, MemoryLayoutConfig* layout_config);
bool graph_compiler_parallelize(GraphCompiler* compiler, ParallelizationStrategy strategy);

// JIT编译（类似JAX）
bool graph_compiler_jit_compile(GraphCompiler* compiler, JITConfig* jit_config);
bool graph_compiler_aot_compile(GraphCompiler* compiler, const char* output_file);

// 高级优化技术
bool graph_compiler_constant_folding(GraphCompiler* compiler);
bool graph_compiler_dead_code_elimination(GraphCompiler* compiler);
bool graph_compiler_common_subexpression_elimination(GraphCompiler* compiler);
bool graph_compiler_loop_optimization(GraphCompiler* compiler);
bool graph_compiler_vectorization(GraphCompiler* compiler);

// 内存优化
bool graph_compiler_memory_planning(GraphCompiler* compiler);
bool graph_compiler_buffer_reuse(GraphCompiler* compiler);
size_t graph_compiler_estimate_memory_usage(GraphCompiler* compiler);

// 性能分析
typedef struct {
    double optimization_time_ms;
    size_t original_nodes;
    size_t optimized_nodes;
    size_t fused_operators;
    size_t memory_saved_bytes;
    double estimated_speedup;
    double compilation_time_ms;
    size_t code_size_bytes;
} OptimizationStats;

OptimizationStats graph_compiler_get_stats(GraphCompiler* compiler);
void graph_compiler_print_stats(GraphCompiler* compiler);

// 融合模式库
FusionPattern* graph_compiler_get_fusion_patterns(size_t* num_patterns);
bool graph_compiler_register_fusion_pattern(FusionPattern* pattern);

// 常见融合模式
bool graph_compiler_fuse_conv_bn_relu(GraphCompiler* compiler);
bool graph_compiler_fuse_matmul_add(GraphCompiler* compiler);
bool graph_compiler_fuse_dropout_relu(GraphCompiler* compiler);
bool graph_compiler_fuse_layernorm_activation(GraphCompiler* compiler);

// 分布式优化（类似TensorFlow分布式）
typedef struct {
    int num_devices;
    int device_id;
    bool enable_nccl;           // NCCL通信优化
    bool enable_mpi;            // MPI通信
    bool enable_gradient_compression;  // 梯度压缩
    float compression_ratio;
    bool enable_allreduce_opt;  // AllReduce优化
} DistributedConfig;

bool graph_compiler_distributed_optimize(GraphCompiler* compiler, DistributedConfig* dist_config);

// 混合精度支持（类似MindSpore）
typedef struct {
    bool enable_fp16;           // FP16支持
    bool enable_bf16;           // BF16支持
    bool enable_mixed_precision; // 混合精度
    float loss_scale;           // 损失缩放
    bool enable_auto_loss_scale;  // 自动损失缩放
    bool enable_stochastic_rounding; // 随机舍入
} MixedPrecisionConfig;

bool graph_compiler_mixed_precision_optimize(GraphCompiler* compiler, MixedPrecisionConfig* mp_config);

// 部署优化（类似TensorFlow Lite/TensorRT）
typedef struct {
    bool enable_quantization;   // 量化
    int quantization_bits;      // 量化位数
    bool enable_pruning;        // 剪枝
    float pruning_ratio;        // 剪枝比例
    bool enable_knowledge_distillation; // 知识蒸馏
    bool enable_mobile_opt;     // 移动端优化
} DeploymentConfig;

bool graph_compiler_deployment_optimize(GraphCompiler* compiler, DeploymentConfig* deploy_config);

// 自动调优（类似AutoTVM）
typedef struct {
    bool enable_autotuning;     // 启用自动调优
    int num_trials;             // 试验次数
    int timeout_seconds;        // 超时时间
    bool enable_hardware_aware;  // 硬件感知
    const char* target_device;  // 目标设备
    bool enable_profile_guided;  // 性能分析引导
} AutoTuningConfig;

bool graph_compiler_autotune(GraphCompiler* compiler, AutoTuningConfig* tuning_config);

// 中文错误信息和文档（类似PaddlePaddle）
typedef enum {
    OPTIMIZER_SUCCESS = 0,
    OPTIMIZER_ERROR_INVALID_GRAPH,
    OPTIMIZER_ERROR_UNSUPPORTED_OPERATION,
    OPTIMIZER_ERROR_MEMORY_ALLOCATION_FAILED,
    OPTIMIZER_ERROR_COMPILATION_FAILED,
    OPTIMIZER_ERROR_PARALLELIZATION_FAILED,
    OPTIMIZER_ERROR_DISTRIBUTED_INIT_FAILED,
    OPTIMIZER_ERROR_AUTOTUNING_FAILED
} OptimizerError;

const char* graph_optimizer_error_string(OptimizerError error);
const char* graph_optimizer_error_string_chinese(OptimizerError error);

// 性能分析器（集成TensorBoard风格）
typedef struct {
    double total_time_ms;
    double optimization_time_ms;
    double compilation_time_ms;
    size_t memory_peak_bytes;
    size_t memory_avg_bytes;
    double throughput_gflops;
    double efficiency_percent;
    int num_operations;
    int num_fused_operations;
} GraphPerformanceProfile;

GraphPerformanceProfile graph_compiler_profile(GraphCompiler* compiler, int warmup_runs, int benchmark_runs);
bool graph_compiler_export_profile(GraphCompiler* compiler, const char* filename);

// 多后端支持（类似MXNet）
typedef enum {
    BACKEND_CPU = 0,
    BACKEND_CUDA,
    BACKEND_ROCM,
    BACKEND_OPENCL,
    BACKEND_METAL,
    BACKEND_TPU,
    BACKEND_ACL,      // ARM Compute Library
    BACKEND_MKLDNN,   // Intel MKL-DNN
    BACKEND_CUDNN,    // NVIDIA cuDNN
    BACKEND_TENSORRT  // NVIDIA TensorRT
} BackendType;

bool graph_compiler_set_backend(GraphCompiler* compiler, BackendType backend);
BackendType graph_compiler_get_optimal_backend(GraphCompiler* compiler);

// 高级API（类似Keras）
typedef struct {
    GraphCompiler* compiler;
    const char* name;
    void* layers;  // Keras风格层列表
    size_t num_layers;
    bool is_compiled;
    void* optimizer;
    void* loss_function;
    void* metrics;
} KerasLikeModel;

KerasLikeModel* keras_model_create(const char* name);
bool keras_model_add_layer(KerasLikeModel* model, const char* layer_type, void* layer_params);
bool keras_model_compile(KerasLikeModel* model, const char* optimizer, const char* loss, const char* metrics[]);
bool keras_model_fit(KerasLikeModel* model, void* x_train, void* y_train, int epochs, int batch_size);
bool keras_model_predict(KerasLikeModel* model, void* x_test, void* y_pred);
bool keras_model_save(KerasLikeModel* model, const char* filepath);
KerasLikeModel* keras_model_load(const char* filepath);

#endif // STATIC_GRAPH_OPTIMIZER_H