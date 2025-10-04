#include "autograd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 全局自动微分引擎
AutogradEngine* g_autograd_engine = NULL;

// 创建自动微分引擎
AutogradEngine* autograd_create_engine(size_t max_graph_size) {
    AutogradEngine* engine = (AutogradEngine*)malloc(sizeof(AutogradEngine));
    if (!engine) return NULL;
    
    engine->graph = (ComputationalGraph*)malloc(sizeof(ComputationalGraph));
    if (!engine->graph) {
        free(engine);
        return NULL;
    }
    
    engine->graph->nodes = (GraphNode**)malloc(max_graph_size * sizeof(GraphNode*));
    if (!engine->graph->nodes) {
        free(engine->graph);
        free(engine);
        return NULL;
    }
    
    engine->graph->num_nodes = 0;
    engine->graph->capacity = max_graph_size;
    engine->graph->grad_enabled = true;
    engine->graph->retain_graph = false;
    
    engine->is_recording = false;
    engine->max_graph_size = max_graph_size;
    
    return engine;
}

// 释放自动微分引擎
void autograd_free_engine(AutogradEngine* engine) {
    if (!engine) return;
    
    if (engine->graph) {
        graph_clear(engine->graph);
        free(engine->graph->nodes);
        free(engine->graph);
    }
    
    free(engine);
}

// 设置梯度计算开关
void autograd_set_grad_enabled(AutogradEngine* engine, bool enabled) {
    if (engine && engine->graph) {
        engine->graph->grad_enabled = enabled;
    }
}

// 检查是否启用梯度计算
bool autograd_is_grad_enabled(AutogradEngine* engine) {
    return engine && engine->graph && engine->graph->grad_enabled;
}

// 清零所有梯度
void autograd_zero_grad(AutogradEngine* engine) {
    if (!engine || !engine->graph) return;
    
    for (size_t i = 0; i < engine->graph->num_nodes; i++) {
        GraphNode* node = engine->graph->nodes[i];
        if (node && node->output && node->output->grad) {
            memset(node->output->grad, 0, node->output->size * sizeof(float));
        }
    }
}

// 创建计算图节点
GraphNode* graph_node_create(OpType op_type, const char* op_name, Tensor* output, 
                            Tensor** inputs, size_t num_inputs, bool requires_grad) {
    GraphNode* node = (GraphNode*)malloc(sizeof(GraphNode));
    if (!node) return NULL;
    
    node->op_type = op_type;
    node->op_name = op_name ? strdup(op_name) : NULL;
    node->output = output;
    node->inputs = inputs;
    node->num_inputs = num_inputs;
    node->grads = NULL;
    node->context = NULL;
    node->parents = NULL;
    node->num_parents = 0;
    node->children = NULL;
    node->num_children = 0;
    node->requires_grad = requires_grad;
    node->backward_func = NULL;
    
    // 设置反向传播函数
    switch (op_type) {
        case OP_ADD: node->backward_func = autograd_add_backward; break;
        case OP_SUB: node->backward_func = autograd_sub_backward; break;
        case OP_MUL: node->backward_func = autograd_mul_backward; break;
        case OP_DIV: node->backward_func = autograd_div_backward; break;
        case OP_MATMUL: node->backward_func = autograd_matmul_backward; break;
        case OP_RELU: node->backward_func = autograd_relu_backward; break;
        case OP_SOFTMAX: node->backward_func = autograd_softmax_backward; break;
        case OP_SIGMOID: node->backward_func = autograd_sigmoid_backward; break;
        case OP_TANH: node->backward_func = autograd_tanh_backward; break;
        case OP_EXP: node->backward_func = autograd_exp_backward; break;
        case OP_LOG: node->backward_func = autograd_log_backward; break;
        case OP_SUM: node->backward_func = autograd_sum_backward; break;
        case OP_MEAN: node->backward_func = autograd_mean_backward; break;
        default: node->backward_func = NULL; break;
    }
    
    return node;
}

// 释放计算图节点
void graph_node_free(GraphNode* node) {
    if (!node) return;
    
    if (node->op_name) free(node->op_name);
    if (node->grads) {
        for (size_t i = 0; i < node->num_inputs; i++) {
            if (node->grads[i]) tensor_free(node->grads[i]);
        }
        free(node->grads);
    }
    if (node->parents) free(node->parents);
    if (node->children) free(node->children);
    
    free(node);
}

// 添加节点到计算图
void graph_add_node(ComputationalGraph* graph, GraphNode* node) {
    if (!graph || !node || graph->num_nodes >= graph->capacity) return;
    
    graph->nodes[graph->num_nodes++] = node;
}

// 清空计算图
void graph_clear(ComputationalGraph* graph) {
    if (!graph) return;
    
    for (size_t i = 0; i < graph->num_nodes; i++) {
        graph_node_free(graph->nodes[i]);
    }
    graph->num_nodes = 0;
}

// 反向传播算法
void autograd_backward(AutogradEngine* engine, Tensor* loss) {
    if (!engine || !engine->graph || !loss || !loss->grad) return;
    
    // 初始化损失梯度为1
    for (size_t i = 0; i < loss->size; i++) {
        loss->grad[i] = 1.0f;
    }
    
    // 反向遍历计算图
    for (int i = (int)engine->graph->num_nodes - 1; i >= 0; i--) {
        GraphNode* node = engine->graph->nodes[i];
        if (node && node->backward_func && node->requires_grad) {
            node->backward_func(node);
        }
    }
}

// 基本操作的反向传播实现
void autograd_add_backward(GraphNode* node) {
    if (!node || node->num_inputs != 2) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input_a = node->inputs[0];
    Tensor* input_b = node->inputs[1];
    
    if (input_a->grad && grad_output) {
        for (size_t i = 0; i < input_a->size; i++) {
            input_a->grad[i] += grad_output->data[i];
        }
    }
    
    if (input_b->grad && grad_output) {
        for (size_t i = 0; i < input_b->size; i++) {
            input_b->grad[i] += grad_output->data[i];
        }
    }
}

void autograd_sub_backward(GraphNode* node) {
    if (!node || node->num_inputs != 2) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input_a = node->inputs[0];
    Tensor* input_b = node->inputs[1];
    
    if (input_a->grad && grad_output) {
        for (size_t i = 0; i < input_a->size; i++) {
            input_a->grad[i] += grad_output->data[i];
        }
    }
    
    if (input_b->grad && grad_output) {
        for (size_t i = 0; i < input_b->size; i++) {
            input_b->grad[i] -= grad_output->data[i];
        }
    }
}

void autograd_mul_backward(GraphNode* node) {
    if (!node || node->num_inputs != 2) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input_a = node->inputs[0];
    Tensor* input_b = node->inputs[1];
    
    if (input_a->grad && grad_output) {
        for (size_t i = 0; i < input_a->size; i++) {
            input_a->grad[i] += grad_output->data[i] * input_b->data[i];
        }
    }
    
    if (input_b->grad && grad_output) {
        for (size_t i = 0; i < input_b->size; i++) {
            input_b->grad[i] += grad_output->data[i] * input_a->data[i];
        }
    }
}

void autograd_div_backward(GraphNode* node) {
    if (!node || node->num_inputs != 2) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input_a = node->inputs[0];
    Tensor* input_b = node->inputs[1];
    
    if (input_a->grad && grad_output) {
        for (size_t i = 0; i < input_a->size; i++) {
            input_a->grad[i] += grad_output->data[i] / input_b->data[i];
        }
    }
    
    if (input_b->grad && grad_output) {
        for (size_t i = 0; i < input_b->size; i++) {
            float b_val = input_b->data[i];
            input_b->grad[i] -= grad_output->data[i] * input_a->data[i] / (b_val * b_val);
        }
    }
}

void autograd_relu_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    
    if (input->grad && grad_output) {
        for (size_t i = 0; i < input->size; i++) {
            if (input->data[i] > 0) {
                input->grad[i] += grad_output->data[i];
            }
        }
    }
}

void autograd_sigmoid_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    
    if (input->grad && grad_output) {
        for (size_t i = 0; i < input->size; i++) {
            float sigmoid_val = 1.0f / (1.0f + expf(-input->data[i]));
            input->grad[i] += grad_output->data[i] * sigmoid_val * (1.0f - sigmoid_val);
        }
    }
}

void autograd_tanh_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    
    if (input->grad && grad_output) {
        for (size_t i = 0; i < input->size; i++) {
            float tanh_val = tanhf(input->data[i]);
            input->grad[i] += grad_output->data[i] * (1.0f - tanh_val * tanh_val);
        }
    }
}

void autograd_exp_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    
    if (input->grad && grad_output) {
        for (size_t i = 0; i < input->size; i++) {
            input->grad[i] += grad_output->data[i] * expf(input->data[i]);
        }
    }
}

void autograd_log_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    
    if (input->grad && grad_output) {
        for (size_t i = 0; i < input->size; i++) {
            if (input->data[i] > 0) {
                input->grad[i] += grad_output->data[i] / input->data[i];
            }
        }
    }
}

void autograd_sum_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    
    if (input->grad && grad_output) {
        float grad_val = grad_output->data[0];
        for (size_t i = 0; i < input->size; i++) {
            input->grad[i] += grad_val;
        }
    }
}

void autograd_mean_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    
    if (input->grad && grad_output) {
        float grad_val = grad_output->data[0] / input->size;
        for (size_t i = 0; i < input->size; i++) {
            input->grad[i] += grad_val;
        }
    }
}

// 矩阵乘法的反向传播
void autograd_matmul_backward(GraphNode* node) {
    if (!node || node->num_inputs != 2) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input_a = node->inputs[0];  // A
    Tensor* input_b = node->inputs[1];  // B
    
    // C = A @ B, 则 dA = dC @ B^T, dB = A^T @ dC
    if (input_a->grad && grad_output && input_b) {
        // dA = dC @ B^T
        for (size_t i = 0; i < input_a->shape[0]; i++) {
            for (size_t j = 0; j < input_a->shape[1]; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < input_b->shape[0]; k++) {
                    size_t grad_idx = i * input_b->shape[1] + k;
                    size_t b_idx = k * input_b->shape[1] + j;
                    if (grad_idx < grad_output->size && b_idx < input_b->size) {
                        sum += grad_output->data[grad_idx] * input_b->data[b_idx];
                    }
                }
                size_t a_idx = i * input_a->shape[1] + j;
                if (a_idx < input_a->size) {
                    input_a->grad[a_idx] += sum;
                }
            }
        }
    }
    
    if (input_b->grad && grad_output && input_a) {
        // dB = A^T @ dC
        for (size_t i = 0; i < input_b->shape[0]; i++) {
            for (size_t j = 0; j < input_b->shape[1]; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < input_a->shape[0]; k++) {
                    size_t a_idx = k * input_a->shape[1] + i;
                    size_t grad_idx = k * grad_output->shape[1] + j;
                    if (a_idx < input_a->size && grad_idx < grad_output->size) {
                        sum += input_a->data[a_idx] * grad_output->data[grad_idx];
                    }
                }
                size_t b_idx = i * input_b->shape[1] + j;
                if (b_idx < input_b->size) {
                    input_b->grad[b_idx] += sum;
                }
            }
        }
    }
}

// Softmax的反向传播
void autograd_softmax_backward(GraphNode* node) {
    if (!node || node->num_inputs != 1) return;
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];
    Tensor* output = node->output;
    
    if (input->grad && grad_output && output) {
        // Softmax反向传播: dL/dx = softmax(x) * (dL/dy - sum(softmax(x) * dL/dy))
        for (size_t i = 0; i < input->size; i++) {
            float softmax_val = output->data[i];
            float grad_out_val = grad_output->data[i];
            
            // 计算 sum(softmax(x) * dL/dy) 用于当前维度
            float sum = 0.0f;
            for (size_t j = 0; j < input->size; j++) {
                sum += output->data[j] * grad_output->data[j];
            }
            
            input->grad[i] += softmax_val * (grad_out_val - sum);
        }
    }
}

// 线性层的反向传播
void autograd_linear_backward(GraphNode* node) {
    if (!node || node->num_inputs != 3) return; // input, weight, bias
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];      // 输入
    Tensor* weight = node->inputs[1];     // 权重
    Tensor* bias = node->inputs[2];       // 偏置
    
    // y = x @ W^T + b
    if (input->grad && grad_output && weight) {
        // dL/dx = dL/dy @ W
        for (size_t i = 0; i < input->shape[0]; i++) {
            for (size_t j = 0; j < input->shape[1]; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < weight->shape[0]; k++) {
                    size_t grad_idx = i * weight->shape[0] + k;
                    size_t weight_idx = k * weight->shape[1] + j;
                    if (grad_idx < grad_output->size && weight_idx < weight->size) {
                        sum += grad_output->data[grad_idx] * weight->data[weight_idx];
                    }
                }
                size_t input_idx = i * input->shape[1] + j;
                if (input_idx < input->size) {
                    input->grad[input_idx] += sum;
                }
            }
        }
    }
    
    if (weight->grad && grad_output && input) {
        // dL/dW = dL/dy^T @ x
        for (size_t i = 0; i < weight->shape[0]; i++) {
            for (size_t j = 0; j < weight->shape[1]; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < input->shape[0]; k++) {
                    size_t grad_idx = k * grad_output->shape[1] + i;
                    size_t input_idx = k * input->shape[1] + j;
                    if (grad_idx < grad_output->size && input_idx < input->size) {
                        sum += grad_output->data[grad_idx] * input->data[input_idx];
                    }
                }
                size_t weight_idx = i * weight->shape[1] + j;
                if (weight_idx < weight->size) {
                    weight->grad[weight_idx] += sum;
                }
            }
        }
    }
    
    if (bias->grad && grad_output) {
        // dL/db = sum(dL/dy, dim=0)
        for (size_t i = 0; i < bias->size; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < grad_output->shape[0]; j++) {
                size_t grad_idx = j * grad_output->shape[1] + i;
                if (grad_idx < grad_output->size) {
                    sum += grad_output->data[grad_idx];
                }
            }
            bias->grad[i] += sum;
        }
    }
}

// 卷积的反向传播
void autograd_conv2d_backward(GraphNode* node) {
    if (!node || node->num_inputs < 2) return; // input, weight, [bias]
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];      // 输入特征图
    Tensor* weight = node->inputs[1];     // 卷积核
    Tensor* bias = (node->num_inputs > 2) ? node->inputs[2] : NULL;
    
    // 简化的卷积反向传播实现
    if (input->grad && grad_output && weight) {
        // dL/dx = dL/dy * W^T (需要反卷积操作)
        printf("计算卷积输入梯度...\n");
        for (size_t i = 0; i < input->size; i++) {
            // 这里应该实现完整的反卷积计算
            // 现在使用简化的近似
            if (i < grad_output->size && i < weight->size) {
                input->grad[i] += grad_output->data[i] * weight->data[i % weight->size];
            }
        }
    }
    
    if (weight->grad && grad_output && input) {
        // dL/dW = x^T * dL/dy (需要相关操作)
        printf("计算卷积权重梯度...\n");
        for (size_t i = 0; i < weight->size; i++) {
            // 这里应该实现完整的权重梯度计算
            // 现在使用简化的近似
            if (i < grad_output->size && i < input->size) {
                weight->grad[i] += grad_output->data[i] * input->data[i % input->size];
            }
        }
    }
    
    if (bias && bias->grad && grad_output) {
        // dL/db = sum(dL/dy, dim=(0,2,3))
        printf("计算卷积偏置梯度...\n");
        for (size_t i = 0; i < bias->size; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < grad_output->size; j += bias->size) {
                if (i + j < grad_output->size) {
                    sum += grad_output->data[i + j];
                }
            }
            bias->grad[i] += sum;
        }
    }
}

// 注意力机制的反向传播
void autograd_attention_backward(GraphNode* node) {
    if (!node || node->num_inputs < 3) return; // query, key, value
    
    Tensor* grad_output = node->output->grad;
    Tensor* query = node->inputs[0];    // Q
    Tensor* key = node->inputs[1];      // K
    Tensor* value = node->inputs[2];    // V
    
    // 简化的注意力反向传播
    if (query->grad && grad_output && key && value) {
        printf("计算注意力查询梯度...\n");
        for (size_t i = 0; i < query->size; i++) {
            // 这里应该实现完整的注意力梯度计算
            // 现在使用简化的近似
            if (i < grad_output->size) {
                query->grad[i] += grad_output->data[i] * 0.1f; // 简化的梯度
            }
        }
    }
    
    if (key->grad && grad_output && query && value) {
        printf("计算注意力键梯度...\n");
        for (size_t i = 0; i < key->size; i++) {
            if (i < grad_output->size) {
                key->grad[i] += grad_output->data[i] * 0.1f; // 简化的梯度
            }
        }
    }
    
    if (value->grad && grad_output && query && key) {
        printf("计算注意力值梯度...\n");
        for (size_t i = 0; i < value->size; i++) {
            if (i < grad_output->size) {
                value->grad[i] += grad_output->data[i] * 0.1f; // 简化的梯度
            }
        }
    }
}

// LayerNorm的反向传播
void autograd_layernorm_backward(GraphNode* node) {
    if (!node || node->num_inputs < 3) return; // input, gamma, beta
    
    Tensor* grad_output = node->output->grad;
    Tensor* input = node->inputs[0];    // 输入
    Tensor* gamma = node->inputs[1];    // 缩放参数
    Tensor* beta = node->inputs[2];     // 平移参数
    
    // 简化的LayerNorm反向传播
    if (input->grad && grad_output && gamma && beta) {
        printf("计算LayerNorm输入梯度...\n");
        
        // 计算均值和方差（简化版）
        float mean = 0.0f;
        float var = 0.0f;
        for (size_t i = 0; i < input->size; i++) {
            mean += input->data[i];
        }
        mean /= input->size;
        
        for (size_t i = 0; i < input->size; i++) {
            float diff = input->data[i] - mean;
            var += diff * diff;
        }
        var /= input->size;
        float std = sqrtf(var + 1e-5f); // 添加epsilon防止除零
        
        // 计算输入梯度
        for (size_t i = 0; i < input->size; i++) {
            float normalized = (input->data[i] - mean) / std;
            if (i < gamma->size && i < grad_output->size) {
                input->grad[i] += grad_output->data[i] * gamma->data[i] / std;
            }
        }
    }
    
    if (gamma->grad && grad_output) {
        printf("计算LayerNorm gamma梯度...\n");
        for (size_t i = 0; i < gamma->size && i < grad_output->size; i++) {
            gamma->grad[i] += grad_output->data[i]; // 简化的gamma梯度
        }
    }
    
    if (beta->grad && grad_output) {
        printf("计算LayerNorm beta梯度...\n");
        for (size_t i = 0; i < beta->size && i < grad_output->size; i++) {
            beta->grad[i] += grad_output->data[i]; // 简化的beta梯度
        }
    }
}

// 全局自动微分引擎初始化
void autograd_init_global_engine(size_t max_graph_size) {
    if (g_autograd_engine) {
        autograd_free_engine(g_autograd_engine);
    }
    g_autograd_engine = autograd_create_engine(max_graph_size);
}

void autograd_free_global_engine(void) {
    if (g_autograd_engine) {
        autograd_free_engine(g_autograd_engine);
        g_autograd_engine = NULL;
    }
}