#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "unified_tensor.h"

// 计算图节点类型
typedef enum {
    OP_NONE,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MATMUL,
    OP_RELU,
    OP_SOFTMAX,
    OP_SIGMOID,
    OP_TANH,
    OP_EXP,
    OP_LOG,
    OP_SUM,
    OP_MEAN,
    OP_MAX,
    OP_MIN,
    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_CONV2D,
    OP_MAXPOOL2D,
    OP_BATCHNORM,
    OP_LAYERNORM,
    OP_LINEAR,
    OP_EMBEDDING,
    OP_ATTENTION,
    OP_CUSTOM
} OpType;

// 计算图节点
typedef struct GraphNode {
    OpType op_type;
    char* op_name;
    Tensor* output;
    Tensor** inputs;
    size_t num_inputs;
    Tensor** grads;
    void* context;
    struct GraphNode** parents;
    size_t num_parents;
    struct GraphNode** children;
    size_t num_children;
    bool requires_grad;
    void (*backward_func)(struct GraphNode*);
} GraphNode;

// 计算图
typedef struct ComputationalGraph {
    GraphNode** nodes;
    size_t num_nodes;
    size_t capacity;
    bool grad_enabled;
    bool retain_graph;
} ComputationalGraph;

// 自动微分引擎
typedef struct AutogradEngine {
    ComputationalGraph* graph;
    bool is_recording;
    size_t max_graph_size;
} AutogradEngine;

// 自动微分函数声明
AutogradEngine* autograd_create_engine(size_t max_graph_size);
void autograd_free_engine(AutogradEngine* engine);
void autograd_set_grad_enabled(AutogradEngine* engine, bool enabled);
bool autograd_is_grad_enabled(AutogradEngine* engine);
void autograd_zero_grad(AutogradEngine* engine);
void autograd_backward(AutogradEngine* engine, Tensor* loss);

// 计算图操作
GraphNode* graph_node_create(OpType op_type, const char* op_name, Tensor* output, 
                            Tensor** inputs, size_t num_inputs, bool requires_grad);
void graph_node_free(GraphNode* node);
void graph_add_node(ComputationalGraph* graph, GraphNode* node);
void graph_clear(ComputationalGraph* graph);

// 基本操作的自动微分实现
void autograd_add_backward(GraphNode* node);
void autograd_sub_backward(GraphNode* node);
void autograd_mul_backward(GraphNode* node);
void autograd_div_backward(GraphNode* node);
void autograd_matmul_backward(GraphNode* node);
void autograd_relu_backward(GraphNode* node);
void autograd_softmax_backward(GraphNode* node);
void autograd_sigmoid_backward(GraphNode* node);
void autograd_tanh_backward(GraphNode* node);
void autograd_exp_backward(GraphNode* node);
void autograd_log_backward(GraphNode* node);
void autograd_sum_backward(GraphNode* node);
void autograd_mean_backward(GraphNode* node);

// 高级操作的自动微分实现
void autograd_linear_backward(GraphNode* node);
void autograd_conv2d_backward(GraphNode* node);
void autograd_attention_backward(GraphNode* node);
void autograd_layernorm_backward(GraphNode* node);

// 张量操作的自动微分包装器
Tensor* autograd_tensor_add(Tensor* a, Tensor* b, AutogradEngine* engine);
Tensor* autograd_tensor_sub(Tensor* a, Tensor* b, AutogradEngine* engine);
Tensor* autograd_tensor_mul(Tensor* a, Tensor* b, AutogradEngine* engine);
Tensor* autograd_tensor_div(Tensor* a, Tensor* b, AutogradEngine* engine);
Tensor* autograd_tensor_matmul(Tensor* a, Tensor* b, AutogradEngine* engine);
Tensor* autograd_tensor_relu(Tensor* tensor, AutogradEngine* engine);
Tensor* autograd_tensor_softmax(Tensor* tensor, int dim, AutogradEngine* engine);
Tensor* autograd_tensor_sigmoid(Tensor* tensor, AutogradEngine* engine);
Tensor* autograd_tensor_tanh(Tensor* tensor, AutogradEngine* engine);
Tensor* autograd_tensor_sum(Tensor* tensor, int dim, bool keepdim, AutogradEngine* engine);
Tensor* autograd_tensor_mean(Tensor* tensor, int dim, bool keepdim, AutogradEngine* engine);

// 全局自动微分引擎
extern AutogradEngine* g_autograd_engine;
void autograd_init_global_engine(size_t max_graph_size);
void autograd_free_global_engine(void);

#endif // AUTOGRAD_H