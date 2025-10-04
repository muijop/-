#ifndef GRAPH_NEURAL_NETWORKS_H
#define GRAPH_NEURAL_NETWORKS_H

#include <stddef.h>
#include <stdbool.h>

// 图神经网络类型枚举
typedef enum {
    GNN_GCN = 0,           // 图卷积网络
    GNN_GAT = 1,           // 图注意力网络
    GNN_GRAPH_SAGE = 2,    // GraphSAGE
    GNN_GIN = 3,           // 图同构网络
    GNN_DGCNN = 4,         // 动态图卷积网络
    GNN_MPNN = 5,          // 消息传递神经网络
    GNN_RGCN = 6,          // 关系图卷积网络
    GNN_STGCN = 7          // 时空图卷积网络
} gnn_type_t;

// 图数据结构
typedef struct {
    size_t num_nodes;           // 节点数量
    size_t num_edges;           // 边数量
    size_t node_feature_dim;    // 节点特征维度
    size_t edge_feature_dim;    // 边特征维度
    
    float* node_features;       // 节点特征矩阵 [num_nodes x node_feature_dim]
    float* edge_features;       // 边特征矩阵 [num_edges x edge_feature_dim]
    size_t* edge_index;         // 边索引 [2 x num_edges]
    
    // 图属性
    bool is_directed;           // 是否为有向图
    bool has_self_loops;        // 是否有自环
    bool is_weighted;           // 是否有边权重
    float* edge_weights;        // 边权重 [num_edges]
    
    // 节点和边标签
    int* node_labels;           // 节点标签
    int* edge_labels;           // 边标签
    
    char* graph_name;           // 图名称
} graph_data_t;

// GCN配置结构体
typedef struct {
    size_t hidden_dims[3];      // 隐藏层维度
    float dropout_rate;         // Dropout率
    bool use_bias;              // 是否使用偏置
    bool normalize;             // 是否归一化
    char activation[16];        // 激活函数
    int num_layers;             // 层数
} gcn_config_t;

// GAT配置结构体
typedef struct {
    size_t hidden_dims[3];      // 隐藏层维度
    int num_heads;              // 注意力头数
    float dropout_rate;         // Dropout率
    float negative_slope;       // LeakyReLU负斜率
    bool concat;                // 是否拼接多头输出
    char activation[16];        // 激活函数
} gat_config_t;

// GraphSAGE配置结构体
typedef struct {
    size_t hidden_dims[3];      // 隐藏层维度
    size_t sample_sizes[3];     // 采样大小
    char aggregator[16];        // 聚合器类型
    float dropout_rate;         // Dropout率
    bool normalize;             // 是否归一化
    char activation[16];        // 激活函数
} graphsage_config_t;

// GNN配置联合体
typedef union {
    gcn_config_t gcn;
    gat_config_t gat;
    graphsage_config_t graphsage;
} gnn_specific_config_t;

// GNN配置结构体
typedef struct {
    gnn_type_t type;                    // GNN类型
    size_t input_dim;                   // 输入维度
    size_t output_dim;                  // 输出维度
    int num_layers;                     // 层数
    float learning_rate;                // 学习率
    float weight_decay;                 // 权重衰减
    bool use_gpu;                       // 是否使用GPU
    int seed;                           // 随机种子
    
    gnn_specific_config_t specific;     // 特定GNN配置
    
    // 训练配置
    int max_epochs;                     // 最大训练轮数
    int patience;                       // 早停耐心值
    float min_delta;                    // 最小改进阈值
    bool early_stopping;                // 是否早停
    
    // 内存和时间限制
    int max_memory_mb;                  // 最大内存使用(MB)
    int max_time_seconds;               // 最大训练时间(秒)
} gnn_config_t;

// GNN模型结构体
typedef struct {
    gnn_type_t type;                    // GNN类型
    gnn_config_t config;                // 配置
    void* model_impl;                   // 模型实现指针
    bool is_trained;                    // 是否已训练
    bool is_initialized;                // 是否已初始化
    
    // 训练状态
    int current_epoch;                  // 当前训练轮数
    float best_loss;                    // 最佳损失值
    float current_loss;                 // 当前损失值
    float training_accuracy;           // 训练准确率
    float validation_accuracy;         // 验证准确率
    
    // 性能统计
    double training_time;               // 训练时间(秒)
    size_t memory_usage;               // 内存使用量(bytes)
} gnn_model_t;

// GNN训练结果结构体
typedef struct {
    float final_loss;                   // 最终损失值
    float final_accuracy;               // 最终准确率
    int total_epochs;                   // 总训练轮数
    double total_time;                  // 总训练时间(秒)
    bool success;                       // 是否成功
    char* status_message;               // 状态消息
    
    // 训练历史
    float* loss_history;                // 损失历史
    float* accuracy_history;            // 准确率历史
    int history_size;                   // 历史记录大小
} gnn_training_result_t;

// GNN预测结果结构体
typedef struct {
    float* node_embeddings;             // 节点嵌入 [num_nodes x output_dim]
    float* graph_embeddings;            // 图嵌入 [batch_size x output_dim]
    int* node_predictions;              // 节点预测结果
    int* graph_predictions;             // 图预测结果
    float* prediction_scores;           // 预测分数
    
    size_t num_nodes;                   // 节点数量
    size_t num_graphs;                  // 图数量
    size_t output_dim;                  // 输出维度
} gnn_prediction_result_t;

// ===========================================
// GNN模型管理API
// ===========================================

// 创建GNN模型
gnn_model_t* create_gnn_model(gnn_type_t type, const gnn_config_t* config);

// 销毁GNN模型
void destroy_gnn_model(gnn_model_t* model);

// 初始化GNN模型
int initialize_gnn_model(gnn_model_t* model, const graph_data_t* graph);

// ===========================================
// 训练和评估API
// ===========================================

// 训练GNN模型
gnn_training_result_t* train_gnn_model(gnn_model_t* model, 
                                      const graph_data_t* train_graph,
                                      const graph_data_t* val_graph,
                                      int task_type); // 0:节点分类, 1:图分类, 2:链接预测

// 评估GNN模型
float evaluate_gnn_model(const gnn_model_t* model, 
                        const graph_data_t* test_graph,
                        int task_type);

// 预测
gnn_prediction_result_t* predict_with_gnn(const gnn_model_t* model,
                                         const graph_data_t* graph,
                                         int task_type);

// ===========================================
// 模型保存和加载API
// ===========================================

// 保存GNN模型
int save_gnn_model(const gnn_model_t* model, const char* filepath);

// 加载GNN模型
gnn_model_t* load_gnn_model(const char* filepath);

// ===========================================
// 图数据处理API
// ===========================================

// 创建图数据
graph_data_t* create_graph_data(size_t num_nodes, size_t num_edges,
                               size_t node_feature_dim, size_t edge_feature_dim);

// 销毁图数据
void destroy_graph_data(graph_data_t* graph);

// 添加边
int add_edge(graph_data_t* graph, size_t source, size_t target, 
            float weight, const float* edge_features);

// 设置节点特征
int set_node_features(graph_data_t* graph, size_t node_index, 
                     const float* features);

// 图数据预处理
int preprocess_graph_data(graph_data_t* graph, int preprocessing_type);

// ===========================================
// 工具函数
// ===========================================

// 创建默认GCN配置
gnn_config_t create_default_gcn_config(size_t input_dim, size_t output_dim);

// 创建默认GAT配置
gnn_config_t create_default_gat_config(size_t input_dim, size_t output_dim);

// 创建默认GraphSAGE配置
gnn_config_t create_default_graphsage_config(size_t input_dim, size_t output_dim);

// 创建示例图数据
graph_data_t* create_example_graph_data(int graph_type); // 0:Karate, 1:Cora, 2:Citeseer

// 获取GNN模型信息
void print_gnn_model_info(const gnn_model_t* model);

// 获取训练统计
void get_gnn_training_stats(const gnn_model_t* model, 
                           float* loss, float* accuracy, int* epoch);

// ===========================================
// 高级功能API
// ===========================================

// 图嵌入学习
gnn_prediction_result_t* learn_graph_embeddings(gnn_model_t* model,
                                               const graph_data_t* graph);

// 图结构学习
int learn_graph_structure(gnn_model_t* model, const graph_data_t* graph);

// 图生成
int generate_graph_with_gnn(gnn_model_t* model, size_t num_nodes,
                           graph_data_t** generated_graph);

// 图异常检测
int detect_graph_anomalies(gnn_model_t* model, const graph_data_t* graph,
                          float* anomaly_scores);

#endif // GRAPH_NEURAL_NETWORKS_H