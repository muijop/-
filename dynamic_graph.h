#ifndef DYNAMIC_GRAPH_H
#define DYNAMIC_GRAPH_H

#include "nn.h"
#include "unified_tensor.h"
#include "tensor_autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 动态图节点类型定义 ====================

typedef enum {
    DYNAMIC_NODE_INPUT = 0,      // 输入节点
    DYNAMIC_NODE_CONSTANT,       // 常量节点
    DYNAMIC_NODE_OPERATION,      // 操作节点
    DYNAMIC_NODE_PARAMETER,      // 参数节点
    DYNAMIC_NODE_OUTPUT,         // 输出节点
    DYNAMIC_NODE_LOSS,           // 损失节点
    DYNAMIC_NODE_GRADIENT,       // 梯度节点
    DYNAMIC_NODE_OPTIMIZER,      // 优化器节点
    DYNAMIC_NODE_LAYER,          // 层节点
    DYNAMIC_NODE_MODULE,         // 模块节点
    DYNAMIC_NODE_CUSTOM          // 自定义节点
} DynamicNodeType;

// ==================== 动态图操作类型定义 ====================

typedef enum {
    DYNAMIC_OP_ADD = 0,          // 加法
    DYNAMIC_OP_SUB,              // 减法
    DYNAMIC_OP_MUL,              // 乘法
    DYNAMIC_OP_DIV,              // 除法
    DYNAMIC_OP_MATMUL,           // 矩阵乘法
    DYNAMIC_OP_CONV,             // 卷积
    DYNAMIC_OP_POOL,             // 池化
    DYNAMIC_OP_RELU,             // ReLU激活
    DYNAMIC_OP_SIGMOID,          // Sigmoid激活
    DYNAMIC_OP_TANH,             // Tanh激活
    DYNAMIC_OP_SOFTMAX,          // Softmax
    DYNAMIC_OP_LOG_SOFTMAX,      // LogSoftmax
    DYNAMIC_OP_DROPOUT,          // Dropout
    DYNAMIC_OP_BATCH_NORM,       // 批归一化
    DYNAMIC_OP_LAYER_NORM,       // 层归一化
    DYNAMIC_OP_EMBEDDING,        // 嵌入层
    DYNAMIC_OP_LSTM,             // LSTM
    DYNAMIC_OP_GRU,              // GRU
    DYNAMIC_OP_ATTENTION,        // 注意力机制
    DYNAMIC_OP_TRANSPOSE,        // 转置
    DYNAMIC_OP_RESHAPE,          // 重塑
    DYNAMIC_OP_CONCAT,           // 拼接
    DYNAMIC_OP_SPLIT,            // 分割
    DYNAMIC_OP_SLICE,            // 切片
    DYNAMIC_OP_GATHER,           // 聚集
    DYNAMIC_OP_SCATTER,          // 分散
    DYNAMIC_OP_REDUCE_SUM,       // 求和归约
    DYNAMIC_OP_REDUCE_MEAN,      // 平均归约
    DYNAMIC_OP_REDUCE_MAX,       // 最大归约
    DYNAMIC_OP_REDUCE_MIN,       // 最小归约
    DYNAMIC_OP_EXP,              // 指数
    DYNAMIC_OP_LOG,              // 对数
    DYNAMIC_OP_SQRT,             // 平方根
    DYNAMIC_OP_POW,              // 幂运算
    DYNAMIC_OP_ABS,              // 绝对值
    DYNAMIC_OP_SIN,              // 正弦
    DYNAMIC_OP_COS,              // 余弦
    DYNAMIC_OP_TAN,              // 正切
    DYNAMIC_OP_CUSTOM            // 自定义操作
} DynamicOperationType;

// ==================== 动态图节点结构 ====================

typedef struct DynamicNode {
    char* name;                    // 节点名称
    DynamicNodeType type;          // 节点类型
    DynamicOperationType op_type;  // 操作类型（仅对操作节点有效）
    AutogradTensor* tensor;        // 节点张量
    struct DynamicNode** inputs;   // 输入节点数组
    int num_inputs;                // 输入节点数量
    struct DynamicNode** outputs;  // 输出节点数组
    int num_outputs;              // 输出节点数量
    void* user_data;              // 用户数据
    int requires_grad;             // 是否需要梯度
    int is_leaf;                   // 是否为叶子节点
    int visited;                   // 访问标记
    int id;                        // 节点ID
} DynamicNode;

// ==================== 动态图结构 ====================

typedef struct DynamicGraph {
    DynamicNode** nodes;           // 节点数组
    int num_nodes;                 // 节点数量
    DynamicNode** inputs;          // 输入节点数组
    int num_inputs;                // 输入节点数量
    DynamicNode** outputs;         // 输出节点数组
    int num_outputs;               // 输出节点数量
    DynamicNode** parameters;      // 参数节点数组
    int num_parameters;            // 参数节点数量
    char* name;                    // 图名称
    void* user_data;               // 用户数据
    int initialized;               // 是否已初始化
} DynamicGraph;

// ==================== 动态图配置结构 ====================

typedef struct DynamicGraphConfig {
    int enable_gradient_checkpointing;  // 是否启用梯度检查点
    int enable_memory_optimization;     // 是否启用内存优化
    int enable_parallel_execution;      // 是否启用并行执行
    int max_memory_usage;               // 最大内存使用量（MB）
    int max_computation_depth;          // 最大计算深度
    int enable_debug_mode;              // 是否启用调试模式
    int enable_profiling;               // 是否启用性能分析
    int enable_automatic_differentiation; // 是否启用自动微分
} DynamicGraphConfig;

// ==================== 动态图创建和销毁 ====================

// 创建动态图
DynamicGraph* dynamic_graph_create(const char* name);

// 销毁动态图
void dynamic_graph_destroy(DynamicGraph* graph);

// ==================== 节点管理 ====================

// 创建输入节点
DynamicNode* dynamic_graph_create_input(DynamicGraph* graph, const char* name, AutogradTensor* tensor);

// 创建常量节点
DynamicNode* dynamic_graph_create_constant(DynamicGraph* graph, const char* name, AutogradTensor* tensor);

// 创建参数节点
DynamicNode* dynamic_graph_create_parameter(DynamicGraph* graph, const char* name, AutogradTensor* tensor);

// 创建操作节点
DynamicNode* dynamic_graph_create_operation(DynamicGraph* graph, const char* name, DynamicOperationType op_type, DynamicNode** inputs, int num_inputs);

// 创建输出节点
DynamicNode* dynamic_graph_create_output(DynamicGraph* graph, const char* name, DynamicNode* input);

// 创建损失节点
DynamicNode* dynamic_graph_create_loss(DynamicGraph* graph, const char* name, DynamicNode* input, DynamicNode* target);

// 创建梯度节点
DynamicNode* dynamic_graph_create_gradient(DynamicGraph* graph, const char* name, DynamicNode* input);

// 删除节点
void dynamic_graph_remove_node(DynamicGraph* graph, DynamicNode* node);

// 查找节点
DynamicNode* dynamic_graph_find_node(DynamicGraph* graph, const char* name);

// 获取所有节点
DynamicNode** dynamic_graph_get_all_nodes(DynamicGraph* graph, int* num_nodes);

// 获取输入节点
DynamicNode** dynamic_graph_get_inputs(DynamicGraph* graph, int* num_inputs);

// 获取输出节点
DynamicNode** dynamic_graph_get_outputs(DynamicGraph* graph, int* num_outputs);

// 获取参数节点
DynamicNode** dynamic_graph_get_parameters(DynamicGraph* graph, int* num_parameters);

// ==================== 图操作 ====================

// 前向传播
AutogradTensor* dynamic_graph_forward(DynamicGraph* graph, DynamicNode* output_node);

// 反向传播
void dynamic_graph_backward(DynamicGraph* graph, DynamicNode* output_node);

// 计算梯度
AutogradTensor* dynamic_graph_compute_gradient(DynamicGraph* graph, DynamicNode* node);

// 清空梯度
void dynamic_graph_zero_grad(DynamicGraph* graph);

// 更新参数
void dynamic_graph_update_parameters(DynamicGraph* graph, float learning_rate);

// ==================== 图优化 ====================

// 优化图结构
void dynamic_graph_optimize(DynamicGraph* graph, const DynamicGraphConfig* config);

// 内存优化
void dynamic_graph_memory_optimize(DynamicGraph* graph);

// 梯度检查点
void dynamic_graph_gradient_checkpointing(DynamicGraph* graph);

// 并行执行
void dynamic_graph_parallel_execute(DynamicGraph* graph);

// ==================== 图序列化 ====================

// 序列化图
void dynamic_graph_serialize(DynamicGraph* graph, const char* filename);

// 反序列化图
DynamicGraph* dynamic_graph_deserialize(const char* filename);

// ==================== 图验证 ====================

// 验证图结构
int dynamic_graph_validate(DynamicGraph* graph);

// 检查图是否包含循环
int dynamic_graph_has_cycle(DynamicGraph* graph);

// 拓扑排序
DynamicNode** dynamic_graph_topological_sort(DynamicGraph* graph, int* num_nodes);

// ==================== 图统计 ====================

// 图统计信息
typedef struct DynamicGraphStats {
    int total_nodes;               // 总节点数
    int input_nodes;               // 输入节点数
    int output_nodes;              // 输出节点数
    int parameter_nodes;           // 参数节点数
    int operation_nodes;           // 操作节点数
    int leaf_nodes;                // 叶子节点数
    int max_depth;                 // 最大深度
    int min_depth;                 // 最小深度
    float average_depth;           // 平均深度
    int total_parameters;          // 总参数数量
    float total_memory_usage;      // 总内存使用量（MB）
    int forward_passes;            // 前向传播次数
    int backward_passes;           // 反向传播次数
} DynamicGraphStats;

// 获取图统计
DynamicGraphStats dynamic_graph_get_stats(DynamicGraph* graph);

// 重置图统计
void dynamic_graph_reset_stats(DynamicGraph* graph);

// ==================== 图调试 ====================

// 打印图信息
void dynamic_graph_print_info(DynamicGraph* graph);

// 打印图结构
void dynamic_graph_print_structure(DynamicGraph* graph);

// 打印节点信息
void dynamic_graph_print_node_info(DynamicNode* node);

// 启用调试模式
void dynamic_graph_set_debug_mode(DynamicGraph* graph, int debug);

// ==================== 节点操作 ====================

// 设置节点名称
void dynamic_node_set_name(DynamicNode* node, const char* name);

// 获取节点名称
const char* dynamic_node_get_name(DynamicNode* node);

// 设置节点张量
void dynamic_node_set_tensor(DynamicNode* node, AutogradTensor* tensor);

// 获取节点张量
AutogradTensor* dynamic_node_get_tensor(DynamicNode* node);

// 设置是否需要梯度
void dynamic_node_set_requires_grad(DynamicNode* node, int requires_grad);

// 获取是否需要梯度
int dynamic_node_get_requires_grad(DynamicNode* node);

// 添加输入节点
void dynamic_node_add_input(DynamicNode* node, DynamicNode* input);

// 移除输入节点
void dynamic_node_remove_input(DynamicNode* node, DynamicNode* input);

// 添加输出节点
void dynamic_node_add_output(DynamicNode* node, DynamicNode* output);

// 移除输出节点
void dynamic_node_remove_output(DynamicNode* node, DynamicNode* output);

// 获取输入节点
DynamicNode** dynamic_node_get_inputs(DynamicNode* node, int* num_inputs);

// 获取输出节点
DynamicNode** dynamic_node_get_outputs(DynamicNode* node, int* num_outputs);

// ==================== 操作节点函数 ====================

// 加法操作
DynamicNode* dynamic_graph_add(DynamicGraph* graph, DynamicNode* a, DynamicNode* b);

// 减法操作
DynamicNode* dynamic_graph_sub(DynamicGraph* graph, DynamicNode* a, DynamicNode* b);

// 乘法操作
DynamicNode* dynamic_graph_mul(DynamicGraph* graph, DynamicNode* a, DynamicNode* b);

// 除法操作
DynamicNode* dynamic_graph_div(DynamicGraph* graph, DynamicNode* a, DynamicNode* b);

// 矩阵乘法操作
DynamicNode* dynamic_graph_matmul(DynamicGraph* graph, DynamicNode* a, DynamicNode* b);

// ReLU激活操作
DynamicNode* dynamic_graph_relu(DynamicGraph* graph, DynamicNode* input);

// Sigmoid激活操作
DynamicNode* dynamic_graph_sigmoid(DynamicGraph* graph, DynamicNode* input);

// Tanh激活操作
DynamicNode* dynamic_graph_tanh(DynamicGraph* graph, DynamicNode* input);

// Softmax操作
DynamicNode* dynamic_graph_softmax(DynamicGraph* graph, DynamicNode* input, int dim);

// LogSoftmax操作
DynamicNode* dynamic_graph_log_softmax(DynamicGraph* graph, DynamicNode* input, int dim);

// Dropout操作
DynamicNode* dynamic_graph_dropout(DynamicGraph* graph, DynamicNode* input, float p);

// 批归一化操作
DynamicNode* dynamic_graph_batch_norm(DynamicGraph* graph, DynamicNode* input, DynamicNode* weight, DynamicNode* bias, DynamicNode* running_mean, DynamicNode* running_var, int training);

// 层归一化操作
DynamicNode* dynamic_graph_layer_norm(DynamicGraph* graph, DynamicNode* input, DynamicNode* weight, DynamicNode* bias);

// 转置操作
DynamicNode* dynamic_graph_transpose(DynamicGraph* graph, DynamicNode* input, int dim0, int dim1);

// 重塑操作
DynamicNode* dynamic_graph_reshape(DynamicGraph* graph, DynamicNode* input, int* shape, int num_dims);

// 拼接操作
DynamicNode* dynamic_graph_concat(DynamicGraph* graph, DynamicNode** inputs, int num_inputs, int dim);

// 求和归约操作
DynamicNode* dynamic_graph_reduce_sum(DynamicGraph* graph, DynamicNode* input, int* dims, int num_dims, int keepdim);

// 平均归约操作
DynamicNode* dynamic_graph_reduce_mean(DynamicGraph* graph, DynamicNode* input, int* dims, int num_dims, int keepdim);

// ==================== 高级图操作 ====================

// 创建子图
DynamicGraph* dynamic_graph_create_subgraph(DynamicGraph* graph, DynamicNode** inputs, int num_inputs, DynamicNode** outputs, int num_outputs);

// 合并图
DynamicGraph* dynamic_graph_merge(DynamicGraph* graph1, DynamicGraph* graph2, DynamicNode** connections, int num_connections);

// 克隆图
DynamicGraph* dynamic_graph_clone(DynamicGraph* graph);

// 图变换
DynamicGraph* dynamic_graph_transform(DynamicGraph* graph, void* transformer);

// ==================== 图工厂函数 ====================

// 创建线性图
DynamicGraph* dynamic_graph_factory_linear(int input_size, int output_size, int bias);

// 创建卷积图
DynamicGraph* dynamic_graph_factory_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int bias);

// 创建LSTM图
DynamicGraph* dynamic_graph_factory_lstm(int input_size, int hidden_size, int num_layers, int bidirectional);

// 创建Transformer图
DynamicGraph* dynamic_graph_factory_transformer(int d_model, int nhead, int num_layers, int dim_feedforward);

// ==================== 图回调函数 ====================

// 图回调函数类型
typedef void (*DynamicGraphCallback)(DynamicGraph* graph, int event_type, void* user_data);

// 设置图回调
void dynamic_graph_set_callback(DynamicGraph* graph, DynamicGraphCallback callback, void* user_data);

// ==================== 图性能分析 ====================

// 性能分析信息
typedef struct DynamicGraphProfile {
    float forward_time;            // 前向传播时间（ms）
    float backward_time;           // 反向传播时间（ms）
    float memory_usage;            // 内存使用量（MB）
    float computation_time;        // 计算时间（ms）
    float communication_time;      // 通信时间（ms）
    int num_operations;           // 操作数量
    int num_parameters;            // 参数数量
} DynamicGraphProfile;

// 性能分析
DynamicGraphProfile dynamic_graph_profile(DynamicGraph* graph);

// 开始性能分析
void dynamic_graph_start_profiling(DynamicGraph* graph);

// 停止性能分析
void dynamic_graph_stop_profiling(DynamicGraph* graph);

// ==================== 图工具函数 ====================

// 图初始化
void dynamic_graph_initialize(DynamicGraph* graph);

// 图验证
int dynamic_graph_validate_structure(DynamicGraph* graph);

// 图重置
void dynamic_graph_reset(DynamicGraph* graph);

// 图比较
int dynamic_graph_compare(DynamicGraph* graph1, DynamicGraph* graph2);

#ifdef __cplusplus
}
#endif

#endif // DYNAMIC_GRAPH_H