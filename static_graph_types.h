#ifndef STATIC_GRAPH_TYPES_H
#define STATIC_GRAPH_TYPES_H

#include "unified_tensor.h"
#include "autograd.h"

// 简单的StaticGraph定义（用于编译）
typedef struct StaticGraph {
    GraphNode** execution_order;  // 执行顺序
    size_t num_nodes;             // 节点数量
    void* user_data;              // 用户数据
} StaticGraph;

// OperationType定义（来自autograd.h的OpType别名）
typedef OpType OperationType;

#endif // STATIC_GRAPH_TYPES_H