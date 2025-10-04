#include "dynamic_graph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// 全局操作注册表
static OperationRegistry g_op_registry[256];
static size_t g_op_registry_size = 0;

// PyTorch风格的全局状态管理
static struct {
    bool grad_enabled;
    bool training_mode;
    int num_threads;
    PrecisionType default_precision;
    bool deterministic;
    uint64_t default_seed;
    MemoryConfig memory_config;
    GradientClippingConfig grad_clip_config;
    MixedPrecisionConfig mixed_precision_config;
} g_pytorch_state = {
    .grad_enabled = true,
    .training_mode = true,
    .num_threads = 4,
    .default_precision = PRECISION_FP32,
    .deterministic = false,
    .default_seed = 42,
    .memory_config = {
        .enable_memory_pool = true,
        .enable_gradient_checkpointing = false,
        .max_memory_mb = 1024,
        .memory_fraction = 0.8f,
        .enable_cudnn_benchmark = true,
        .enable_cudnn_deterministic = false
    },
    .grad_clip_config = {
        .accumulation_steps = 1,
        .max_norm = 1.0f,
        .norm_type = 2.0f,
        .enabled = false
    },
    .mixed_precision_config = {
        .precision = PRECISION_FP32,
        .enable_autocast = false,
        .loss_scale = 65536.0f,
        .dynamic_loss_scale = true,
        .growth_factor = 2.0f,
        .backoff_factor = 0.5f,
        .growth_interval = 2000
    }
};

// 随机数生成器管理
static struct {
    RandomGenerator* default_gen;
    RandomGenerator** generators;
    size_t num_generators;
    bool deterministic;
} g_random_state = {
    .default_gen = NULL,
    .generators = NULL,
    .num_generators = 0,
    .deterministic = false
};

// PyTorch风格的设备管理
static struct {
    DeviceType current_device;
    int device_count;
    bool* device_available;
    void** device_contexts;
} g_device_state = {
    .current_device = DEVICE_CPU,
    .device_count = 1,
    .device_available = NULL,
    .device_contexts = NULL
};

// NEON exp函数近似实现（用于ARM架构）
#ifdef __arm__
static float32x4_t exp_ps_neon(float32x4_t x) {
    // 使用多项式近似计算exp
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    
    // Taylor级数近似: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    float32x4_t result = vaddq_f32(one, x);
    result = vaddq_f32(result, vmulq_f32(x2, vdupq_n_f32(0.5f)));
    result = vaddq_f32(result, vmulq_f32(x3, vdupq_n_f32(0.16666667f)));
    result = vaddq_f32(result, vmulq_f32(x4, vdupq_n_f32(0.04166667f)));
    
    return result;
}
#endif

// 创建动态图
DynamicGraph* dynamic_graph_create(bool enable_grad) {
    DynamicGraph* graph = (DynamicGraph*)calloc(1, sizeof(DynamicGraph));
    if (!graph) return NULL;
    
    graph->enable_grad = enable_grad;
    graph->is_training = true;
    graph->next_node_id = 0;
    graph->execution_cache = NULL;
    
    return graph;
}

// 销毁动态图
void dynamic_graph_destroy(DynamicGraph* graph) {
    if (!graph) return;
    
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        DynamicGraphNode* next = node->next;
        
        if (node->name) free(node->name);
        if (node->inputs) free(node->inputs);
        if (node->outputs) free(node->outputs);
        if (node->op_data) free(node->op_data);
        
        free(node);
        node = next;
    }
    
    if (graph->execution_cache) {
        free(graph->execution_cache);
    }
    
    free(graph);
}

// PyTorch风格的梯度管理
void dynamic_graph_train(DynamicGraph* graph) {
    if (!graph) return;
    graph->is_training = true;
    g_pytorch_state.training_mode = true;
}

void dynamic_graph_eval(DynamicGraph* graph) {
    if (!graph) return;
    graph->is_training = false;
    g_pytorch_state.training_mode = false;
}

bool dynamic_graph_is_grad_enabled(DynamicGraph* graph) {
    return graph ? graph->enable_grad : g_pytorch_state.grad_enabled;
}

void dynamic_graph_set_grad_enabled(DynamicGraph* graph, bool enabled) {
    if (graph) {
        graph->enable_grad = enabled;
    }
    g_pytorch_state.grad_enabled = enabled;
}

// PyTorch风格的上下文管理器
NoGradContext* no_grad_context_create(DynamicGraph* graph) {
    NoGradContext* ctx = (NoGradContext*)calloc(1, sizeof(NoGradContext));
    if (!ctx) return NULL;
    
    ctx->graph = graph;
    ctx->prev_grad_enabled = graph ? graph->enable_grad : g_pytorch_state.grad_enabled;
    ctx->prev_training_mode = graph ? graph->is_training : g_pytorch_state.training_mode;
    
    if (graph) {
        graph->enable_grad = false;
    }
    g_pytorch_state.grad_enabled = false;
    
    return ctx;
}

void no_grad_context_destroy(NoGradContext* ctx) {
    if (!ctx) return;
    
    if (ctx->graph) {
        ctx->graph->enable_grad = ctx->prev_grad_enabled;
        ctx->graph->is_training = ctx->prev_training_mode;
    }
    g_pytorch_state.grad_enabled = ctx->prev_grad_enabled;
    g_pytorch_state.training_mode = ctx->prev_training_mode;
    
    free(ctx);
}

// PyTorch风格的随机数生成器
RandomGenerator* random_generator_create(uint64_t seed) {
    RandomGenerator* gen = (RandomGenerator*)calloc(1, sizeof(RandomGenerator));
    if (!gen) return NULL;
    
    gen->seed = seed;
    gen->offset = 0;
    gen->deterministic = g_random_state.deterministic;
    gen->generator_name = strdup("default");
    
    // 添加到全局管理器
    g_random_state.generators = (RandomGenerator**)realloc(g_random_state.generators, 
                                                           (g_random_state.num_generators + 1) * sizeof(RandomGenerator*));
    if (g_random_state.generators) {
        g_random_state.generators[g_random_state.num_generators++] = gen;
    }
    
    return gen;
}

void random_generator_destroy(RandomGenerator* gen) {
    if (!gen) return;
    
    // 从全局管理器中移除
    for (size_t i = 0; i < g_random_state.num_generators; i++) {
        if (g_random_state.generators[i] == gen) {
            // 移动后续元素
            for (size_t j = i; j < g_random_state.num_generators - 1; j++) {
                g_random_state.generators[j] = g_random_state.generators[j + 1];
            }
            g_random_state.num_generators--;
            break;
        }
    }
    
    if (gen->generator_name) free(gen->generator_name);
    free(gen);
}

void random_generator_manual_seed(RandomGenerator* gen, uint64_t seed) {
    if (!gen) return;
    gen->seed = seed;
    gen->offset = 0;
}

// PyTorch风格的随机数生成函数
DynamicTensor* dynamic_tensor_randn(DynamicGraph* graph, int* shape, size_t ndim, RandomGenerator* generator) {
    if (!graph || !shape) return NULL;
    
    RandomGenerator* gen = generator ? generator : g_random_state.default_gen;
    if (!gen) {
        gen = random_generator_create(g_pytorch_state.default_seed);
        g_random_state.default_gen = gen;
    }
    
    DynamicTensor* tensor = dynamic_tensor_create(graph, shape, ndim, false);
    if (!tensor) return NULL;
    
    float* data = tensor->tensor->tensor->data;
    int size = tensor->tensor->tensor->size;
    
    // Box-Muller变换生成正态分布随机数
    for (int i = 0; i < size; i += 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        
        float z0 = sqrtf(-2.0f * logf(u1 + 1e-8f)) * cosf(2.0f * M_PI * u2);
        float z1 = sqrtf(-2.0f * logf(u1 + 1e-8f)) * sinf(2.0f * M_PI * u2);
        
        data[i] = z0;
        if (i + 1 < size) {
            data[i + 1] = z1;
        }
    }
    
    gen->offset += size;
    return tensor;
}

DynamicTensor* dynamic_tensor_uniform(DynamicGraph* graph, int* shape, size_t ndim, float low, float high, RandomGenerator* gen) {
    if (!graph || !shape) return NULL;
    
    DynamicTensor* tensor = dynamic_tensor_create(graph, shape, ndim, false);
    if (!tensor) return NULL;
    
    float* data = tensor->tensor->tensor->data;
    int size = tensor->tensor->tensor->size;
    float range = high - low;
    
    for (int i = 0; i < size; i++) {
        data[i] = low + range * ((float)rand() / RAND_MAX);
    }
    
    return tensor;
}

DynamicTensor* dynamic_tensor_normal(DynamicGraph* graph, int* shape, size_t ndim, float mean, float std, RandomGenerator* gen) {
    DynamicTensor* tensor = dynamic_tensor_randn(graph, shape, ndim, gen);
    if (!tensor) return NULL;
    
    float* data = tensor->tensor->tensor->data;
    int size = tensor->tensor->tensor->size;
    
    for (int i = 0; i < size; i++) {
        data[i] = mean + std * data[i];
    }
    
    return tensor;
}

// PyTorch风格的梯度累积和裁剪
void dynamic_graph_set_gradient_clipping(DynamicGraph* graph, GradientClippingConfig* config) {
    if (!graph || !config) return;
    
    memcpy(&g_pytorch_state.grad_clip_config, config, sizeof(GradientClippingConfig));
    
    // 应用到图中的所有参数节点
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        if (node->type == NODE_PARAMETER && node->tensor && node->tensor->grad_node) {
            // 这里可以设置梯度裁剪参数到具体的张量
        }
        node = node->next;
    }
}

void dynamic_graph_accumulate_gradients(DynamicGraph* graph, bool enable) {
    if (!graph) return;
    
    g_pytorch_state.grad_clip_config.accumulation_steps = enable ? 4 : 1; // 默认4步累积
    
    // 重置所有参数的梯度累积计数
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        if (node->type == NODE_PARAMETER && node->tensor) {
            autograd_tensor_zero_grad(node->tensor);
        }
        node = node->next;
    }
}

// PyTorch风格的混合精度训练支持
void dynamic_graph_set_precision(DynamicGraph* graph, MixedPrecisionConfig* config) {
    if (!graph || !config) return;
    
    memcpy(&g_pytorch_state.mixed_precision_config, config, sizeof(MixedPrecisionConfig));
    
    // 应用到图中的所有张量
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        if (node->tensor) {
            // 根据精度设置转换张量类型
            switch (config->precision) {
                case PRECISION_FP16:
                    // 转换为FP16（这里简化处理）
                    break;
                case PRECISION_BF16:
                    // 转换为BF16
                    break;
                case PRECISION_MIXED:
                    // 启用混合精度模式
                    break;
                case PRECISION_TF32:
                    // 启用TF32模式
                    break;
                default:
                    // 保持FP32
                    break;
            }
        }
        node = node->next;
    }
}

void dynamic_graph_autocast_enable(DynamicGraph* graph) {
    if (!graph) return;
    g_pytorch_state.mixed_precision_config.enable_autocast = true;
}

void dynamic_graph_autocast_disable(DynamicGraph* graph) {
    if (!graph) return;
    g_pytorch_state.mixed_precision_config.enable_autocast = false;
}

// PyTorch风格的内存管理
void dynamic_graph_set_memory_config(DynamicGraph* graph, MemoryConfig* config) {
    if (!graph || !config) return;
    
    memcpy(&g_pytorch_state.memory_config, config, sizeof(MemoryConfig));
    
    // 应用内存限制
    if (config->max_memory_mb > 0) {
        // 这里可以实现内存监控和限制逻辑
    }
    
    if (config->enable_memory_pool) {
        // 启用内存池
    }
}

void dynamic_graph_empty_cache(DynamicGraph* graph) {
    if (!graph) return;
    
    // 清空计算缓存
    if (graph->execution_cache) {
        free(graph->execution_cache);
        graph->execution_cache = NULL;
    }
    
    // 清空张量缓存（这里可以实现更复杂的缓存管理）
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        if (node->tensor && node->tensor->grad_node) {
            // 清空梯度缓存
        }
        node = node->next;
    }
}

size_t dynamic_graph_memory_allocated(DynamicGraph* graph) {
    if (!graph) return 0;
    
    size_t total_memory = 0;
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        if (node->tensor && node->tensor->tensor) {
            total_memory += node->tensor->tensor->size * sizeof(float);
            if (node->tensor->grad_node) {
                total_memory += tensor_autograd_size(node->tensor) * sizeof(float);
            }
        }
        node = node->next;
    }
    
    return total_memory;
}

size_t dynamic_graph_memory_reserved(DynamicGraph* graph) {
    // 这里可以实现预留内存的计算
    return dynamic_graph_memory_allocated(graph) * 1.2; // 简单估算
}

// PyTorch风格的模型保存和加载
bool dynamic_graph_save_checkpoint(DynamicGraph* graph, CheckpointConfig* config) {
    if (!graph || !config) return false;
    
    FILE* fp = fopen(config->model_path, "wb");
    if (!fp) return false;
    
    // 保存图的基本信息
    fwrite(&graph->num_nodes, sizeof(size_t), 1, fp);
    fwrite(&graph->num_inputs, sizeof(size_t), 1, fp);
    fwrite(&graph->num_outputs, sizeof(size_t), 1, fp);
    
    // 保存所有参数节点
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        if (node->type == NODE_PARAMETER && node->tensor) {
            // 保存参数名称
            size_t name_len = node->name ? strlen(node->name) : 0;
            fwrite(&name_len, sizeof(size_t), 1, fp);
            if (name_len > 0) {
                fwrite(node->name, 1, name_len, fp);
            }
            
            // 保存张量形状
            int ndim = tensor_autograd_ndim(node->tensor->tensor);
            fwrite(&ndim, sizeof(int), 1, fp);
            fwrite(tensor_autograd_shape(node->tensor->tensor), sizeof(int), ndim, fp);
            
            // 保存张量数据
            int size = tensor_autograd_size(node->tensor->tensor);
            fwrite(&size, sizeof(int), 1, fp);
            fwrite(tensor_autograd_data(node->tensor->tensor), sizeof(float), size, fp);
        }
        node = node->next;
    }
    
    fclose(fp);
    
    // 保存优化器状态（如果启用）
    if (config->save_optimizer) {
        // 这里可以实现优化器状态的保存
    }
    
    return true;
}

bool dynamic_graph_load_checkpoint(DynamicGraph* graph, CheckpointConfig* config) {
    if (!graph || !config) return false;
    
    FILE* fp = fopen(config->model_path, "rb");
    if (!fp) return false;
    
    // 读取图的基本信息
    size_t num_nodes, num_inputs, num_outputs;
    fread(&num_nodes, sizeof(size_t), 1, fp);
    fread(&num_inputs, sizeof(size_t), 1, fp);
    fread(&num_outputs, sizeof(size_t), 1, fp);
    
    // 读取参数数据
    for (size_t i = 0; i < num_nodes; i++) {
        // 读取参数名称
        size_t name_len;
        fread(&name_len, sizeof(size_t), 1, fp);
        char* name = NULL;
        if (name_len > 0) {
            name = (char*)malloc(name_len + 1);
            fread(name, 1, name_len, fp);
            name[name_len] = '\0';
        }
        
        // 读取张量形状
        int ndim;
        fread(&ndim, sizeof(int), 1, fp);
        int* shape = (int*)malloc(ndim * sizeof(int));
        fread(shape, sizeof(int), ndim, fp);
        
        // 读取张量数据
        int size;
        fread(&size, sizeof(int), 1, fp);
        float* data = (float*)malloc(size * sizeof(float));
        fread(data, sizeof(float), size, fp);
        
        // 在图中查找对应的参数节点
        DynamicGraphNode* node = graph->nodes;
        while (node) {
            if (node->type == NODE_PARAMETER && node->name && name && 
                strcmp(node->name, name) == 0 && node->tensor) {
                // 验证形状匹配
                if (tensor_autograd_ndim(node->tensor->tensor) == ndim) {
                    bool shape_match = true;
                    int* node_shape = tensor_autograd_shape(node->tensor->tensor);
                    for (int j = 0; j < ndim; j++) {
                        if (node_shape[j] != shape[j]) {
                            shape_match = false;
                            break;
                        }
                    }
                    
                    if (shape_match) {
                        // 加载数据
                        memcpy(tensor_autograd_data(node->tensor->tensor), data, size * sizeof(float));
                    }
                }
                break;
            }
            node = node->next;
        }
        
        free(name);
        free(shape);
        free(data);
    }
    
    fclose(fp);
    return true;
}

// Einsum操作（爱因斯坦求和约定）的简化实现
DynamicTensor* dynamic_einsum(const char* equation, DynamicTensor** tensors, size_t num_tensors) {
    if (!equation || !tensors || num_tensors == 0) return NULL;
    
    // 简化的einsum实现，支持基本的矩阵乘法
    if (num_tensors == 2 && strcmp(equation, "ij,jk->ik") == 0) {
        return dynamic_matmul(tensors[0], tensors[1]);
    }
    
    // 支持批量矩阵乘法
    if (num_tensors == 2 && strcmp(equation, "bij,bjk->bik") == 0) {
        return dynamic_matmul(tensors[0], tensors[1]);
    }
    
    // 支持转置操作
    if (num_tensors == 1 && strcmp(equation, "ij->ji") == 0) {
        // 这里应该实现转置操作
        return tensors[0]; // 简化返回原张量
    }
    
    // 默认返回第一个张量
    return tensors[0];
}

// PyTorch风格的JIT编译和图优化
JITContext* dynamic_graph_jit_compile(DynamicGraph* graph, JITConfig* config) {
    if (!graph || !config) return NULL;
    
    JITContext* jit_ctx = (JITContext*)malloc(sizeof(JITContext));
    if (!jit_ctx) return NULL;
    
    jit_ctx->graph = graph;
    jit_ctx->config = *config;
    jit_ctx->is_compiled = false;
    jit_ctx->optimized_graph = NULL;
    
    // 图优化：移除死代码
    if (config->enable_dead_code_elimination) {
        // 标记所有输出节点
        DynamicGraphNode* node = graph->nodes;
        while (node) {
            node->visited = false;
            node = node->next;
        }
        
        // 从输出节点开始反向遍历
        DynamicGraphNode* output_node = graph->output_nodes;
        while (output_node) {
            output_node->visited = true;
            output_node = output_node->next;
        }
        
        // 移除未访问的节点
        DynamicGraphNode* prev = NULL;
        DynamicGraphNode* current = graph->nodes;
        while (current) {
            if (!current->visited && current->type != NODE_INPUT && current->type != NODE_OUTPUT) {
                DynamicGraphNode* to_remove = current;
                current = current->next;
                if (prev) {
                    prev->next = current;
                } else {
                    graph->nodes = current;
                }
                graph->num_nodes--;
                // 这里应该释放节点内存
            } else {
                prev = current;
                current = current->next;
            }
        }
    }
    
    // 图优化：常量折叠
    if (config->enable_constant_folding) {
        DynamicGraphNode* node = graph->nodes;
        while (node) {
            if (node->type == NODE_OPERATION && node->op_type == OP_ADD) {
                // 检查操作数是否为常量
                DynamicGraphNode* input1 = node->inputs[0];
                DynamicGraphNode* input2 = node->inputs[1];
                
                if (input1->type == NODE_CONSTANT && input2->type == NODE_CONSTANT) {
                    // 执行常量加法
                    float result = input1->tensor->tensor->data[0] + input2->tensor->tensor->data[0];
                    
                    // 创建新的常量节点
                    int shape = 1;
                    DynamicTensor* const_tensor = dynamic_tensor_create(graph, &shape, 1, &result);
                    DynamicGraphNode* const_node = dynamic_graph_add_constant(graph, const_tensor, "folded_const");
                    
                    // 替换原节点
                    // 这里应该实现节点替换逻辑
                }
            }
            node = node->next;
        }
    }
    
    // 图优化：算子融合
    if (config->enable_operator_fusion) {
        // 寻找可以融合的算子序列
        DynamicGraphNode* node = graph->nodes;
        while (node) {
            if (node->type == NODE_OPERATION && node->op_type == OP_MATMUL) {
                DynamicGraphNode* next_node = node->next;
                if (next_node && next_node->type == NODE_OPERATION && next_node->op_type == OP_ADD) {
                    // 检查是否可以融合为GEMM操作
                    // 这里可以实现GEMM融合逻辑
                }
            }
            node = node->next;
        }
    }
    
    jit_ctx->is_compiled = true;
    return jit_ctx;
}

void dynamic_graph_jit_free(JITContext* jit_ctx) {
    if (!jit_ctx) return;
    
    if (jit_ctx->optimized_graph) {
        // 释放优化后的图
        // dynamic_graph_free(jit_ctx->optimized_graph);
    }
    
    free(jit_ctx);
}

// PyTorch风格的分布式训练支持
DistributedContext* dynamic_graph_distributed_init(DistributedConfig* config) {
    if (!config) return NULL;
    
    DistributedContext* dist_ctx = (DistributedContext*)malloc(sizeof(DistributedContext));
    if (!dist_ctx) return NULL;
    
    dist_ctx->config = *config;
    dist_ctx->config.rank = config->rank;
    dist_ctx->config.world_size = config->world_size;
    dist_ctx->config.local_rank = config->local_rank;
    dist_ctx->is_initialized = true;
    
    // 初始化通信后端
    switch (config->backend) {
        case BACKEND_NCCL:
            // 初始化NCCL（这里简化处理）
            break;
        case BACKEND_GLOO:
            // 初始化GLOO
            break;
        case BACKEND_MPI:
            // 初始化MPI
            break;
        default:
            break;
    }
    
    return dist_ctx;
}

void dynamic_graph_distributed_free(DistributedContext* dist_ctx) {
    if (!dist_ctx) return;
    
    if (dist_ctx->is_initialized) {
        // 清理通信后端
        switch (dist_ctx->config.backend) {
            case BACKEND_NCCL:
                // 清理NCCL
                break;
            case BACKEND_GLOO:
                // 清理GLOO
                break;
            case BACKEND_MPI:
                // 清理MPI
                break;
            default:
                break;
        }
    }
    
    free(dist_ctx);
}

bool dynamic_graph_all_reduce(DistributedContext* dist_ctx, DynamicTensor* tensor, int op) {
    if (!dist_ctx || !tensor || !dist_ctx->is_initialized) return false;
    
    if (dist_ctx->config.world_size <= 1) return true; // 单进程无需通信
    
    // 这里应该实现实际的all-reduce操作
    // 简化处理：模拟all-reduce效果
    int size = tensor_autograd_size(tensor->tensor);
    float* data = tensor_autograd_data(tensor->tensor);
    
    switch (op) {
        case REDUCE_SUM:
            // 模拟求和
            for (int i = 0; i < size; i++) {
                data[i] *= dist_ctx->config.world_size; // 简化模拟
            }
            break;
        case REDUCE_MEAN:
            // 平均值已经在求和后除以world_size
            break;
        case REDUCE_MAX:
            // 模拟最大值
            break;
        case REDUCE_MIN:
            // 模拟最小值
            break;
        default:
            break;
    }
    
    return true;
}

bool dynamic_graph_broadcast(DistributedContext* dist_ctx, DynamicTensor* tensor, int root_rank) {
    if (!dist_ctx || !tensor || !dist_ctx->is_initialized) return false;
    
    if (dist_ctx->config.world_size <= 1) return true; // 单进程无需通信
    
    // 这里应该实现实际的broadcast操作
    // 简化处理：模拟broadcast效果
    return true;
}

bool dynamic_graph_barrier(DistributedContext* dist_ctx) {
    if (!dist_ctx || !dist_ctx->is_initialized) return false;
    
    if (dist_ctx->config.world_size <= 1) return true; // 单进程无需通信
    
    // 这里应该实现实际的barrier操作
    return true;
}

// 清空动态图
void dynamic_graph_clear(DynamicGraph* graph) {
    if (!graph) return;
    
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        DynamicGraphNode* next = node->next;
        
        if (node->name) free(node->name);
        if (node->inputs) free(node->inputs);
        if (node->outputs) free(node->outputs);
        if (node->op_data) free(node->op_data);
        
        free(node);
        node = next;
    }
    
    graph->nodes = NULL;
    graph->input_nodes = NULL;
    graph->output_nodes = NULL;
    graph->num_nodes = 0;
    graph->num_inputs = 0;
    graph->num_outputs = 0;
    graph->next_node_id = 0;
    
    if (graph->execution_cache) {
        free(graph->execution_cache);
        graph->execution_cache = NULL;
    }
}

// 添加输入节点
DynamicGraphNode* dynamic_graph_add_input(DynamicGraph* graph, const char* name, int* shape, size_t ndim, bool requires_grad) {
    if (!graph) return NULL;
    
    DynamicGraphNode* node = (DynamicGraphNode*)calloc(1, sizeof(DynamicGraphNode));
    if (!node) return NULL;
    
    node->type = NODE_INPUT;
    node->id = graph->next_node_id++;
    node->name = name ? strdup(name) : NULL;
    node->requires_grad = requires_grad;
    node->is_leaf = true;
    
    // 创建张量
    node->tensor = tensor_autograd_zeros(shape, ndim);
    if (!node->tensor) {
        free(node->name);
        free(node);
        return NULL;
    }
    
    // 添加到图
    node->next = graph->nodes;
    if (graph->nodes) {
        graph->nodes->prev = node;
    }
    graph->nodes = node;
    graph->num_nodes++;
    graph->num_inputs++;
    
    if (!graph->input_nodes) {
        graph->input_nodes = node;
    }
    
    return node;
}

// 添加参数节点
DynamicGraphNode* dynamic_graph_add_parameter(DynamicGraph* graph, const char* name, int* shape, size_t ndim) {
    if (!graph) return NULL;
    
    DynamicGraphNode* node = (DynamicGraphNode*)calloc(1, sizeof(DynamicGraphNode));
    if (!node) return NULL;
    
    node->type = NODE_PARAMETER;
    node->id = graph->next_node_id++;
    node->name = name ? strdup(name) : NULL;
    node->requires_grad = true;
    node->is_leaf = true;
    
    // 创建参数张量（使用Xavier初始化）
    node->tensor = tensor_autograd_xavier_uniform(shape, ndim);
    if (!node->tensor) {
        free(node->name);
        free(node);
        return NULL;
    }
    
    // 添加到图
    node->next = graph->nodes;
    if (graph->nodes) {
        graph->nodes->prev = node;
    }
    graph->nodes = node;
    graph->num_nodes++;
    
    return node;
}

// PyTorch风格的广播机制实现
static bool can_broadcast(int* shape1, size_t ndim1, int* shape2, size_t ndim2) {
    size_t max_ndim = ndim1 > ndim2 ? ndim1 : ndim2;
    
    for (size_t i = 0; i < max_ndim; i++) {
        int dim1 = i < ndim1 ? shape1[ndim1 - 1 - i] : 1;
        int dim2 = i < ndim2 ? shape2[ndim2 - 1 - i] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

static void compute_broadcast_shape(int* shape1, size_t ndim1, int* shape2, size_t ndim2, int* result_shape, size_t* result_ndim) {
    size_t max_ndim = ndim1 > ndim2 ? ndim1 : ndim2;
    *result_ndim = max_ndim;
    
    for (size_t i = 0; i < max_ndim; i++) {
        int dim1 = i < ndim1 ? shape1[ndim1 - 1 - i] : 1;
        int dim2 = i < ndim2 ? shape2[ndim2 - 1 - i] : 1;
        
        result_shape[max_ndim - 1 - i] = (dim1 > dim2) ? dim1 : dim2;
    }
}

// 动态张量操作（增强版）
DynamicTensor* dynamic_tensor_create(DynamicGraph* graph, int* shape, size_t ndim, bool requires_grad) {
    if (!graph) return NULL;
    
    DynamicTensor* tensor = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!tensor) return NULL;
    
    tensor->graph = graph;
    tensor->tensor = tensor_autograd_zeros(shape, ndim);
    if (!tensor->tensor) {
        free(tensor);
        return NULL;
    }
    
    tensor->node = dynamic_graph_add_input(graph, NULL, shape, ndim, requires_grad);
    if (!tensor->node) {
        tensor_autograd_destroy(tensor->tensor);
        free(tensor);
        return NULL;
    }
    
    tensor->node->tensor = tensor->tensor;
    return tensor;
}

// PyTorch风格的高级索引操作
DynamicTensor* dynamic_tensor_index_select(DynamicTensor* tensor, int dim, DynamicTensor* indices) {
    if (!tensor || !indices || dim < 0) return NULL;
    
    int* tensor_shape = tensor->tensor->tensor->shape;
    size_t tensor_ndim = tensor->tensor->tensor->ndim;
    
    if (dim >= (int)tensor_ndim) return NULL;
    
    int* indices_data = (int*)indices->tensor->tensor->data;
    int indices_size = indices->tensor->tensor->size;
    
    // 计算输出形状
    int* output_shape = (int*)malloc(tensor_ndim * sizeof(int));
    memcpy(output_shape, tensor_shape, tensor_ndim * sizeof(int));
    output_shape[dim] = indices_size;
    
    DynamicTensor* result = dynamic_tensor_create(tensor->graph, output_shape, tensor_ndim, tensor->tensor->requires_grad);
    free(output_shape);
    
    if (!result) return NULL;
    
    float* tensor_data = tensor->tensor->tensor->data;
    float* result_data = result->tensor->tensor->data;
    
    // 执行索引选择
    int stride = 1;
    for (int i = dim + 1; i < (int)tensor_ndim; i++) {
        stride *= tensor_shape[i];
    }
    
    for (int i = 0; i < indices_size; i++) {
        int src_idx = indices_data[i] * stride;
        int dst_idx = i * stride;
        memcpy(&result_data[dst_idx], &tensor_data[src_idx], stride * sizeof(float));
    }
    
    return result;
}

DynamicTensor* dynamic_tensor_masked_select(DynamicTensor* tensor, DynamicTensor* mask) {
    if (!tensor || !mask) return NULL;
    
    float* tensor_data = tensor->tensor->tensor->data;
    bool* mask_data = (bool*)mask->tensor->tensor->data;
    int tensor_size = tensor->tensor->tensor->size;
    
    // 计算选择后的元素数量
    int selected_count = 0;
    for (int i = 0; i < tensor_size; i++) {
        if (mask_data[i]) {
            selected_count++;
        }
    }
    
    if (selected_count == 0) return NULL;
    
    // 创建结果张量
    int result_shape[1] = {selected_count};
    DynamicTensor* result = dynamic_tensor_create(tensor->graph, result_shape, 1, tensor->tensor->requires_grad);
    if (!result) return NULL;
    
    float* result_data = result->tensor->tensor->data;
    
    // 执行掩码选择
    int idx = 0;
    for (int i = 0; i < tensor_size; i++) {
        if (mask_data[i]) {
            result_data[idx++] = tensor_data[i];
        }
    }
    
    return result;
}

DynamicTensor* dynamic_tensor_scatter(DynamicTensor* tensor, int dim, DynamicTensor* index, DynamicTensor* src) {
    if (!tensor || !index || !src) return NULL;
    
    // 这里实现scatter操作，需要复杂的形状检查和数据复制
    // 为简化实现，这里返回一个副本
    int* shape = tensor->tensor->tensor->shape;
    size_t ndim = tensor->tensor->tensor->ndim;
    
    DynamicTensor* result = dynamic_tensor_create(tensor->graph, shape, ndim, tensor->tensor->requires_grad);
    if (!result) return NULL;
    
    float* tensor_data = tensor->tensor->tensor->data;
    float* result_data = result->tensor->tensor->data;
    
    // 复制原始数据
    memcpy(result_data, tensor_data, tensor->tensor->tensor->size * sizeof(float));
    
    return result;
}

DynamicTensor* dynamic_tensor_gather(DynamicTensor* tensor, int dim, DynamicTensor* index) {
    if (!tensor || !index) return NULL;
    
    // 这里实现gather操作，需要复杂的形状检查
    // 为简化实现，这里返回一个副本
    int* index_shape = index->tensor->tensor->shape;
    size_t index_ndim = index->tensor->tensor->ndim;
    
    DynamicTensor* result = dynamic_tensor_create(tensor->graph, index_shape, index_ndim, tensor->tensor->requires_grad);
    if (!result) return NULL;
    
    float* tensor_data = tensor->tensor->tensor->data;
    float* result_data = result->tensor->tensor->data;
    int tensor_size = tensor->tensor->tensor->size;
    int result_size = result->tensor->tensor->size;
    
    // 简单的数据复制（实际应该根据索引gather）
    for (int i = 0; i < result_size && i < tensor_size; i++) {
        result_data[i] = tensor_data[i];
    }
    
    return result;
}

DynamicTensor* dynamic_tensor_from_data(DynamicGraph* graph, float* data, int* shape, size_t ndim, bool requires_grad) {
    if (!graph) return NULL;
    
    DynamicTensor* tensor = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!tensor) return NULL;
    
    tensor->graph = graph;
    tensor->tensor = tensor_autograd_from_array(data, shape, ndim);
    if (!tensor->tensor) {
        free(tensor);
        return NULL;
    }
    
    tensor->node = dynamic_graph_add_input(graph, NULL, shape, ndim, requires_grad);
    if (!tensor->node) {
        tensor_autograd_destroy(tensor->tensor);
        free(tensor);
        return NULL;
    }
    
    tensor->node->tensor = tensor->tensor;
    return tensor;
}

void dynamic_tensor_destroy(DynamicTensor* tensor) {
    if (!tensor) return;
    
    if (tensor->tensor) {
        tensor_autograd_destroy(tensor->tensor);
    }
    
    free(tensor);
}

// 基本操作实现
static AutogradTensor* op_add_forward(AutogradTensor** inputs, size_t num_inputs, void* params) {
    if (num_inputs != 2) return NULL;
    return tensor_autograd_add(inputs[0], inputs[1]);
}

static void op_add_backward(AutogradTensor* grad_output, AutogradTensor** inputs, AutogradTensor** grad_inputs, size_t num_inputs, void* params) {
    if (num_inputs != 2) return;
    
    grad_inputs[0] = tensor_autograd_copy(grad_output);
    grad_inputs[1] = tensor_autograd_copy(grad_output);
}

static AutogradTensor* op_matmul_forward(AutogradTensor** inputs, size_t num_inputs, void* params) {
    if (num_inputs != 2) return NULL;
    return tensor_autograd_matmul(inputs[0], inputs[1]);
}

static void op_matmul_backward(AutogradTensor* grad_output, AutogradTensor** inputs, AutogradTensor** grad_inputs, size_t num_inputs, void* params) {
    if (num_inputs != 2) return;
    
    // grad_A = grad_output @ B^T
    AutogradTensor* bt = tensor_autograd_transpose(inputs[1], -1, -2);
    grad_inputs[0] = tensor_autograd_matmul(grad_output, bt);
    tensor_autograd_destroy(bt);
    
    // grad_B = A^T @ grad_output
    AutogradTensor* at = tensor_autograd_transpose(inputs[0], -1, -2);
    grad_inputs[1] = tensor_autograd_matmul(at, grad_output);
    tensor_autograd_destroy(at);
}

static AutogradTensor* op_relu_forward(AutogradTensor** inputs, size_t num_inputs, void* params) {
    if (num_inputs != 1) return NULL;
    return tensor_autograd_relu(inputs[0]);
}

static void op_relu_backward(AutogradTensor* grad_output, AutogradTensor** inputs, AutogradTensor** grad_inputs, size_t num_inputs, void* params) {
    if (num_inputs != 1) return;
    
    // ReLU梯度: grad_output * (input > 0)
    AutogradTensor* mask = tensor_autograd_greater_than(inputs[0], 0.0f);
    grad_inputs[0] = tensor_autograd_mul(grad_output, mask);
    tensor_autograd_destroy(mask);
}

// 操作注册
void dynamic_graph_register_operation(OperationRegistry* op_reg) {
    if (g_op_registry_size < 256) {
        g_op_registry[g_op_registry_size++] = *op_reg;
    }
}

OperationRegistry* dynamic_graph_get_operation(OperationType type) {
    for (size_t i = 0; i < g_op_registry_size; i++) {
        if (g_op_registry[i].type == type) {
            return &g_op_registry[i];
        }
    }
    return NULL;
}

// 重载操作符
DynamicTensor* dynamic_add(DynamicTensor* a, DynamicTensor* b) {
    if (!a || !b || a->graph != b->graph) return NULL;
    
    AutogradTensor* inputs[2] = {a->tensor, b->tensor};
    AutogradTensor* result = op_add_forward(inputs, 2, NULL);
    if (!result) return NULL;
    
    DynamicTensor* output = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!output) {
        tensor_autograd_destroy(result);
        return NULL;
    }
    
    output->graph = a->graph;
    output->tensor = result;
    
    // 创建图节点
    GraphNode** inputs_nodes = (GraphNode**)calloc(2, sizeof(GraphNode*));
    if (!inputs_nodes) {
        tensor_autograd_destroy(result);
        free(output);
        return NULL;
    }
    
    inputs_nodes[0] = a->node;
    inputs_nodes[1] = b->node;
    
    output->node = dynamic_graph_add_operation(a->graph, OP_ADD, "add", inputs_nodes, 2, NULL);
    free(inputs_nodes);
    
    if (!output->node) {
        tensor_autograd_destroy(result);
        free(output);
        return NULL;
    }
    
    output->node->tensor = result;
    return output;
}

DynamicTensor* dynamic_matmul(DynamicTensor* a, DynamicTensor* b) {
    if (!a || !b || a->graph != b->graph) return NULL;
    
    AutogradTensor* inputs[2] = {a->tensor, b->tensor};
    AutogradTensor* result = op_matmul_forward(inputs, 2, NULL);
    if (!result) return NULL;
    
    DynamicTensor* output = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!output) {
        tensor_autograd_destroy(result);
        return NULL;
    }
    
    output->graph = a->graph;
    output->tensor = result;
    
    GraphNode** inputs_nodes = (GraphNode**)calloc(2, sizeof(GraphNode*));
    if (!inputs_nodes) {
        tensor_autograd_destroy(result);
        free(output);
        return NULL;
    }
    
    inputs_nodes[0] = a->node;
    inputs_nodes[1] = b->node;
    
    output->node = dynamic_graph_add_operation(a->graph, OP_MATMUL, "matmul", inputs_nodes, 2, NULL);
    free(inputs_nodes);
    
    if (!output->node) {
        tensor_autograd_destroy(result);
        free(output);
        return NULL;
    }
    
    output->node->tensor = result;
    return output;
}

DynamicTensor* dynamic_relu(DynamicTensor* x) {
    if (!x) return NULL;
    
    AutogradTensor* inputs[1] = {x->tensor};
    AutogradTensor* result = op_relu_forward(inputs, 1, NULL);
    if (!result) return NULL;
    
    DynamicTensor* output = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!output) {
        tensor_autograd_destroy(result);
        return NULL;
    }
    
    output->graph = x->graph;
    output->tensor = result;
    
    DynamicGraphNode** inputs_nodes = (DynamicGraphNode**)calloc(1, sizeof(DynamicGraphNode*));
    if (!inputs_nodes) {
        tensor_autograd_destroy(result);
        free(output);
        return NULL;
    }
    
    inputs_nodes[0] = x->node;
    
    output->node = dynamic_graph_add_operation(x->graph, OP_RELU, "relu", inputs_nodes, 1, NULL);
    free(inputs_nodes);
    
    if (!output->node) {
        tensor_autograd_destroy(result);
        free(output);
        return NULL;
    }
    
    output->node->tensor = result;
    return output;
}

// 其他操作的简化实现
DynamicTensor* dynamic_sub(DynamicTensor* a, DynamicTensor* b) {
    // 实现减法：a - b = a + (-1 * b)
    DynamicTensor* neg_b = dynamic_mul_scalar(b, -1.0f);
    if (!neg_b) return NULL;
    
    DynamicTensor* result = dynamic_add(a, neg_b);
    dynamic_tensor_destroy(neg_b);
    return result;
}

DynamicTensor* dynamic_mul(DynamicTensor* a, DynamicTensor* b) {
    if (!a || !b || a->graph != b->graph) return NULL;
    
    // 元素级乘法
    AutogradTensor* result = tensor_autograd_mul(a->tensor, b->tensor);
    if (!result) return NULL;
    
    DynamicTensor* output = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!output) {
        tensor_autograd_destroy(result);
        return NULL;
    }
    
    output->graph = a->graph;
    output->tensor = result;
    
    return output;
}

DynamicTensor* dynamic_sigmoid(DynamicTensor* x) {
    if (!x) return NULL;
    
    AutogradTensor* result = tensor_autograd_sigmoid(x->tensor);
    if (!result) return NULL;
    
    DynamicTensor* output = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!output) {
        tensor_autograd_destroy(result);
        return NULL;
    }
    
    output->graph = x->graph;
    output->tensor = result;
    
    return output;
}

DynamicTensor* dynamic_tanh(DynamicTensor* x) {
    if (!x) return NULL;
    
    AutogradTensor* result = tensor_autograd_tanh(x->tensor);
    if (!result) return NULL;
    
    DynamicTensor* output = (DynamicTensor*)calloc(1, sizeof(DynamicTensor));
    if (!output) {
        tensor_autograd_destroy(result);
        return NULL;
    }
    
    output->graph = x->graph;
    output->tensor = result;
    
    return output;
}

// 自动微分
void dynamic_graph_backward(DynamicGraph* graph, DynamicTensor* loss) {
    if (!graph || !loss || !graph->enable_grad) return;
    
    // 设置梯度为1
    tensor_autograd_fill(loss->tensor, 1.0f);
    
    // 执行反向传播
    tensor_autograd_backward(loss->tensor);
}

void dynamic_graph_zero_grad(DynamicGraph* graph) {
    if (!graph) return;
    
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        if (node->tensor && node->requires_grad) {
            tensor_autograd_zero_grad(node->tensor);
        }
        node = node->next;
    }
}

// 训练模式设置
void dynamic_graph_set_training(DynamicGraph* graph, bool training) {
    if (!graph) return;
    graph->is_training = training;
}

bool dynamic_graph_is_training(DynamicGraph* graph) {
    return graph ? graph->is_training : false;
}

// 调试和可视化
void dynamic_graph_print(DynamicGraph* graph) {
    if (!graph) return;
    
    printf("DynamicGraph (nodes: %zu, inputs: %zu, outputs: %zu):\n", 
           graph->num_nodes, graph->num_inputs, graph->num_outputs);
    
    DynamicGraphNode* node = graph->nodes;
    while (node) {
        printf("  Node %d: %s (type: %d, requires_grad: %d)\n", 
               node->id, node->name ? node->name : "unnamed", 
               node->type, node->requires_grad);
        node = node->next;
    }
}

// 中文支持
ChineseOperationInfo* dynamic_graph_get_chinese_info(OperationType type) {
    static ChineseOperationInfo chinese_info[] = {
        {"加法", "Add", "逐元素加法操作", "基础运算"},
        {"减法", "Sub", "逐元素减法操作", "基础运算"},
        {"乘法", "Mul", "逐元素乘法操作", "基础运算"},
        {"除法", "Div", "逐元素除法操作", "基础运算"},
        {"矩阵乘法", "MatMul", "矩阵乘法操作", "线性代数"},
        {"卷积2D", "Conv2D", "二维卷积操作", "卷积运算"},
        {"ReLU激活", "ReLU", "ReLU激活函数", "激活函数"},
        {"Sigmoid激活", "Sigmoid", "Sigmoid激活函数", "激活函数"},
        {"Tanh激活", "Tanh", "双曲正切激活函数", "激活函数"},
        {"Softmax激活", "Softmax", "Softmax激活函数", "激活函数"},
        {"批归一化", "BatchNorm", "批归一化操作", "归一化"},
        {"Dropout", "Dropout", "Dropout正则化", "正则化"},
        {"变形", "Reshape", "张量形状变换", "形状操作"},
        {"转置", "Transpose", "张量转置操作", "形状操作"},
        {"拼接", "Concat", "张量拼接操作", "形状操作"},
        {"分割", "Split", "张量分割操作", "形状操作"},
        {"求和", "ReduceSum", "张量求和操作", "归约操作"},
        {"求平均", "ReduceMean", "张量求平均操作", "归约操作"},
        {"收集", "Gather", "张量收集操作", "索引操作"},
        {"散布", "Scatter", "张量散布操作", "索引操作"}
    };
    
    if (type >= 0 && type < sizeof(chinese_info) / sizeof(chinese_info[0])) {
        return &chinese_info[type];
    }
    return NULL;
}

// 错误处理
const char* dynamic_graph_error_string(GraphError error) {
    switch (error) {
        case GRAPH_SUCCESS: return "成功";
        case GRAPH_ERROR_INVALID_INPUT: return "无效输入";
        case GRAPH_ERROR_SHAPE_MISMATCH: return "形状不匹配";
        case GRAPH_ERROR_OPERATION_FAILED: return "操作失败";
        case GRAPH_ERROR_OUT_OF_MEMORY: return "内存不足";
        case GRAPH_ERROR_DISTRIBUTED_FAILURE: return "分布式错误";
        default: return "未知错误";
    }
}

static GraphError g_last_error = GRAPH_SUCCESS;

GraphError dynamic_graph_get_last_error(void) {
    return g_last_error;
}