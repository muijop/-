#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "tensor.h"

// 神经网络层类型枚举
typedef enum {
    LAYER_DENSE,           // 全连接层
    LAYER_CONV1D,          // 1D卷积层
    LAYER_CONV2D,          // 2D卷积层
    LAYER_LSTM,            // LSTM层
    LAYER_GRU,             // GRU层
    LAYER_ATTENTION,       // 注意力层
    LAYER_TRANSFORMER,     // Transformer层
    LAYER_RELU,            // ReLU激活层
    LAYER_SIGMOID,         // Sigmoid激活层
    LAYER_TANH,            // Tanh激活层
    LAYER_LEAKY_RELU,      // LeakyReLU激活层
    LAYER_ELU,             // ELU激活层
    LAYER_SELU,            // SELU激活层
    LAYER_SOFTMAX,         // Softmax激活层
    LAYER_BATCH_NORM,      // 批归一化层
    LAYER_DROPOUT,         // Dropout层
    LAYER_POOLING,         // 池化层
    LAYER_FLATTEN,         // 展平层
    LAYER_RESHAPE,         // 重塑层
    LAYER_CONCAT,          // 连接层
    LAYER_ADD,             // 加法层
    LAYER_MULTIPLY,        // 乘法层
    LAYER_MAX_POOL,        // 最大池化层
    LAYER_AVG_POOL,        // 平均池化层
    LAYER_GLOBAL_MAX_POOL, // 全局最大池化层
    LAYER_GLOBAL_AVG_POOL, // 全局平均池化层
    LAYER_EMBEDDING,       // 嵌入层
    LAYER_LAYER_NORM,      // 层归一化层
    LAYER_INSTANCE_NORM,   // 实例归一化层
    LAYER_GROUP_NORM,      // 组归一化层
    LAYER_SPATIAL_DROPOUT, // 空间Dropout层
    LAYER_ALPHA_DROPOUT,   // Alpha Dropout层
    LAYER_GAUSSIAN_DROPOUT, // 高斯Dropout层
    LAYER_SWISH,           // Swish激活层
    LAYER_MISH,            // Mish激活层
    LAYER_GELU,            // GELU激活层
    LAYER_SILU,            // SiLU激活层
    LAYER_PRELU,           // PReLU激活层
    LAYER_RRELU,           // RReLU激活层
    LAYER_CELU,            // CELU激活层
    LAYER_SOFTPLUS,        // Softplus激活层
    LAYER_SOFTSIGN,        // Softsign激活层
    LAYER_HARDSHRINK,      // Hard Shrink激活层
    LAYER_SOFTSHRINK,      // Soft Shrink激活层
    LAYER_TANHSHRINK,      // Tanh Shrink激活层
    LAYER_THRESHOLD,       // Threshold激活层
    LAYER_LOG_SOFTMAX,     // Log Softmax激活层
    LAYER_HARDSIGMOID,     // Hard Sigmoid激活层
    LAYER_HARDTANH,        // Hard Tanh激活层
    LAYER_IDENTITY,        // 恒等层
    LAYER_ZERO_PADDING,    // 零填充层
    LAYER_REFLECT_PADDING, // 反射填充层
    LAYER_REPLICATE_PADDING, // 复制填充层
    LAYER_CIRCULAR_PADDING, // 循环填充层
    LAYER_CONSTANT_PADDING, // 常数填充层
    LAYER_SYMMETRIC_PADDING, // 对称填充层
    LAYER_EDGE_PADDING,    // 边缘填充层
    LAYER_MIRROR_PADDING,  // 镜像填充层
    LAYER_WRAP_PADDING,    // 环绕填充层
    LAYER_NEAREST_PADDING, // 最近邻填充层
    LAYER_BILINEAR_PADDING, // 双线性填充层
    LAYER_BICUBIC_PADDING, // 双三次填充层
    LAYER_LANCZOS_PADDING, // Lanczos填充层
    LAYER_HAMMING_PADDING, // Hamming填充层
    LAYER_BLACKMAN_PADDING, // Blackman填充层
    LAYER_BARTLETT_PADDING, // Bartlett填充层
    LAYER_PARZEN_PADDING,  // Parzen填充层
    LAYER_KAISER_PADDING,  // Kaiser填充层
    LAYER_DOLPH_PADDING,   // Dolph填充层
    LAYER_GAUSSIAN_PADDING, // 高斯填充层
    LAYER_EXPONENTIAL_PADDING, // 指数填充层
    LAYER_POISSON_PADDING, // 泊松填充层
    LAYER_RAYLEIGH_PADDING, // Rayleigh填充层
    LAYER_WEIBULL_PADDING, // Weibull填充层
    LAYER_GAMMA_PADDING,   // Gamma填充层
    LAYER_BETA_PADDING,    // Beta填充层
    LAYER_CHI_SQUARED_PADDING, // 卡方填充层
    LAYER_STUDENT_T_PADDING, // Student's t填充层
    LAYER_F_PADDING,       // F填充层
    LAYER_LOG_NORMAL_PADDING, // 对数正态填充层
    LAYER_CAUCHY_PADDING,  // Cauchy填充层
    LAYER_LAPLACE_PADDING, // Laplace填充层
    LAYER_UNIFORM_PADDING, // 均匀填充层
    LAYER_TRIANGULAR_PADDING, // 三角填充层
    LAYER_TRAPEZOIDAL_PADDING, // 梯形填充层
    LAYER_PARABOLIC_PADDING, // 抛物线填充层
    LAYER_SINE_PADDING,    // 正弦填充层
    LAYER_COSINE_PADDING,  // 余弦填充层
    LAYER_SQUARE_PADDING,  // 方波填充层
    LAYER_SAWTOOTH_PADDING, // 锯齿波填充层
    LAYER_TRIANGLE_PADDING, // 三角波填充层
    LAYER_PULSE_PADDING,   // 脉冲填充层
    LAYER_NOISE_PADDING,   // 噪声填充层
    LAYER_RANDOM_PADDING,  // 随机填充层
    LAYER_PERIODIC_PADDING, // 周期填充层
    LAYER_APERIODIC_PADDING, // 非周期填充层
    LAYER_DETERMINISTIC_PADDING, // 确定性填充层
    LAYER_STOCHASTIC_PADDING, // 随机填充层
    LAYER_ADAPTIVE_PADDING, // 自适应填充层
    LAYER_LEARNABLE_PADDING, // 可学习填充层
    LAYER_DYNAMIC_PADDING, // 动态填充层
    LAYER_STATIC_PADDING,  // 静态填充层
    LAYER_FIXED_PADDING,   // 固定填充层
    LAYER_VARIABLE_PADDING, // 可变填充层
    LAYER_CONSTANT_VALUE_PADDING, // 常数值填充层
    LAYER_REFLECT_VALUE_PADDING, // 反射值填充层
    LAYER_SYMMETRIC_VALUE_PADDING, // 对称值填充层
    LAYER_EDGE_VALUE_PADDING, // 边缘值填充层
    LAYER_WRAP_VALUE_PADDING, // 环绕值填充层
    LAYER_NEAREST_VALUE_PADDING, // 最近邻值填充层
    LAYER_BILINEAR_VALUE_PADDING, // 双线性值填充层
    LAYER_BICUBIC_VALUE_PADDING, // 双三次值填充层
    LAYER_LANCZOS_VALUE_PADDING, // Lanczos值填充层
    LAYER_HAMMING_VALUE_PADDING, // Hamming值填充层
    LAYER_BLACKMAN_VALUE_PADDING, // Blackman值填充层
    LAYER_BARTLETT_VALUE_PADDING, // Bartlett值填充层
    LAYER_PARZEN_VALUE_PADDING, // Parzen值填充层
    LAYER_KAISER_VALUE_PADDING, // Kaiser值填充层
    LAYER_DOLPH_VALUE_PADDING, // Dolph值填充层
    LAYER_GAUSSIAN_VALUE_PADDING, // 高斯值填充层
    LAYER_EXPONENTIAL_VALUE_PADDING, // 指数值填充层
    LAYER_POISSON_VALUE_PADDING, // 泊松值填充层
    LAYER_RAYLEIGH_VALUE_PADDING, // Rayleigh值填充层
    LAYER_WEIBULL_VALUE_PADDING, // Weibull值填充层
    LAYER_GAMMA_VALUE_PADDING, // Gamma值填充层
    LAYER_BETA_VALUE_PADDING, // Beta值填充层
    LAYER_CHI_SQUARED_VALUE_PADDING, // 卡方值填充层
    LAYER_STUDENT_T_VALUE_PADDING, // Student's t值填充层
    LAYER_F_VALUE_PADDING, // F值填充层
    LAYER_LOG_NORMAL_VALUE_PADDING, // 对数正态值填充层
    LAYER_CAUCHY_VALUE_PADDING, // Cauchy值填充层
    LAYER_LAPLACE_VALUE_PADDING, // Laplace值填充层
    LAYER_UNIFORM_VALUE_PADDING, // 均匀值填充层
    LAYER_TRIANGULAR_VALUE_PADDING, // 三角值填充层
    LAYER_TRAPEZOIDAL_VALUE_PADDING, // 梯形值填充层
    LAYER_PARABOLIC_VALUE_PADDING, // 抛物线值填充层
    LAYER_SINE_VALUE_PADDING, // 正弦值填充层
    LAYER_COSINE_VALUE_PADDING, // 余弦值填充层
    LAYER_SQUARE_VALUE_PADDING, // 方波值填充层
    LAYER_SAWTOOTH_VALUE_PADDING, // 锯齿波值填充层
    LAYER_TRIANGLE_VALUE_PADDING, // 三角波值填充层
    LAYER_PULSE_VALUE_PADDING, // 脉冲值填充层
    LAYER_NOISE_VALUE_PADDING, // 噪声值填充层
    LAYER_RANDOM_VALUE_PADDING, // 随机值填充层
    LAYER_PERIODIC_VALUE_PADDING, // 周期值填充层
    LAYER_APERIODIC_VALUE_PADDING, // 非周期值填充层
    LAYER_DETERMINISTIC_VALUE_PADDING, // 确定性值填充层
    LAYER_STOCHASTIC_VALUE_PADDING, // 随机值填充层
    LAYER_ADAPTIVE_VALUE_PADDING, // 自适应值填充层
    LAYER_LEARNABLE_VALUE_PADDING, // 可学习值填充层
    LAYER_DYNAMIC_VALUE_PADDING, // 动态值填充层
    LAYER_STATIC_VALUE_PADDING, // 静态值填充层
    LAYER_FIXED_VALUE_PADDING, // 固定值填充层
    LAYER_VARIABLE_VALUE_PADDING, // 可变值填充层
    LAYER_LAST              // 最后一个层类型标记
} layer_type_t;

// 层配置结构体
typedef struct {
    layer_type_t type;      // 层类型
    int input_dim;          // 输入维度
    int output_dim;         // 输出维度
    int kernel_size;        // 卷积核大小
    int stride;            // 步长
    int padding;           // 填充
    int dilation;          // 膨胀率
    int groups;            // 分组数
    int bias;              // 是否使用偏置
    float dropout_rate;    // Dropout率
    float alpha;           // 激活函数参数（如LeakyReLU的负斜率）
    float negative_slope;  // 负斜率
    float inplace;         // 是否原地操作
    float momentum;        // 动量（用于批归一化）
    float eps;             // 数值稳定性常数
    float affine;          // 是否使用仿射变换
    float track_running_stats; // 是否跟踪运行统计量
    float num_features;    // 特征数
    float num_groups;      // 组数
    float num_channels;    // 通道数
    float num_heads;       // 头数（用于注意力机制）
    float embed_dim;       // 嵌入维度
    float num_embeddings;  // 嵌入数量
    float padding_idx;     // 填充索引
    float max_norm;        // 最大范数
    float norm_type;       // 范数类型
    float scale_grad_by_freq; // 是否按频率缩放梯度
    float sparse;          // 是否使用稀疏梯度
    float _weight;         // 权重
} layer_config_t;

// 层结构体
typedef struct layer {
    layer_type_t type;     // 层类型
    layer_config_t config; // 层配置
    tensor_t* weights;     // 权重张量
    tensor_t* bias;        // 偏置张量
    tensor_t* input;       // 输入张量
    tensor_t* output;      // 输出张量
    tensor_t* grad_input;  // 输入梯度
    tensor_t* grad_weights; // 权重梯度
    tensor_t* grad_bias;   // 偏置梯度
    struct layer* next;    // 下一层
    struct layer* prev;    // 前一层
} layer_t;

// 层函数声明
layer_t* layer_create(layer_type_t type, layer_config_t config);
void layer_destroy(layer_t* layer);
tensor_t* layer_forward(layer_t* layer, tensor_t* input);
tensor_t* layer_backward(layer_t* layer, tensor_t* grad_output);
void layer_update(layer_t* layer, float learning_rate);
void layer_zero_grad(layer_t* layer);

// 特定层类型的创建函数
layer_t* dense_layer_create(int input_dim, int output_dim, int bias);
layer_t* conv1d_layer_create(int input_channels, int output_channels, 
                             int kernel_size, int stride, int padding, 
                             int dilation, int groups, int bias);
layer_t* conv2d_layer_create(int input_channels, int output_channels, 
                             int kernel_size, int stride, int padding, 
                             int dilation, int groups, int bias);
layer_t* lstm_layer_create(int input_size, int hidden_size, int num_layers, 
                          int bias, int batch_first, int dropout, 
                          int bidirectional);
layer_t* gru_layer_create(int input_size, int hidden_size, int num_layers, 
                         int bias, int batch_first, int dropout, 
                         int bidirectional);
layer_t* attention_layer_create(int embed_dim, int num_heads, int dropout, 
                                int bias, int add_bias_kv, int add_zero_attn, 
                                int kdim, int vdim);
layer_t* transformer_layer_create(int d_model, int nhead, int num_encoder_layers, 
                                   int num_decoder_layers, int dim_feedforward, 
                                   int dropout, int activation, 
                                   int custom_encoder, int custom_decoder);

// 激活层创建函数
layer_t* relu_layer_create(int inplace);
layer_t* sigmoid_layer_create(void);
layer_t* tanh_layer_create(void);
layer_t* leaky_relu_layer_create(float negative_slope, int inplace);
layer_t* elu_layer_create(float alpha, int inplace);
layer_t* selu_layer_create(int inplace);
layer_t* softmax_layer_create(int dim);

// 归一化层创建函数
layer_t* batch_norm_layer_create(int num_features, float eps, float momentum, 
                                int affine, int track_running_stats);
layer_t* layer_norm_layer_create(int normalized_shape, float eps, int elementwise_affine);
layer_t* instance_norm_layer_create(int num_features, float eps, float momentum, 
                                   int affine, int track_running_stats);
layer_t* group_norm_layer_create(int num_groups, int num_channels, float eps, 
                                int affine);

// Dropout层创建函数
layer_t* dropout_layer_create(float p, int inplace);
layer_t* spatial_dropout_layer_create(float p);
layer_t* alpha_dropout_layer_create(float p, int inplace);
layer_t* gaussian_dropout_layer_create(float p);

// 池化层创建函数
layer_t* max_pool_layer_create(int kernel_size, int stride, int padding, 
                              int dilation, int return_indices, int ceil_mode);
layer_t* avg_pool_layer_create(int kernel_size, int stride, int padding, 
                              int ceil_mode, int count_include_pad);
layer_t* global_max_pool_layer_create(void);
layer_t* global_avg_pool_layer_create(void);

// 工具层创建函数
layer_t* flatten_layer_create(int start_dim, int end_dim);
layer_t* reshape_layer_create(int* shape, int shape_size);
layer_t* concat_layer_create(int dim);
layer_t* add_layer_create(void);
layer_t* multiply_layer_create(void);

// 填充层创建函数
layer_t* zero_padding_layer_create(int padding);
layer_t* reflect_padding_layer_create(int padding);
layer_t* replicate_padding_layer_create(int padding);
layer_t* circular_padding_layer_create(int padding);
layer_t* constant_padding_layer_create(int padding, float value);
layer_t* symmetric_padding_layer_create(int padding);
layer_t* edge_padding_layer_create(int padding);
layer_t* mirror_padding_layer_create(int padding);
layer_t* wrap_padding_layer_create(int padding);
layer_t* nearest_padding_layer_create(int padding);
layer_t* bilinear_padding_layer_create(int padding);
layer_t* bicubic_padding_layer_create(int padding);
layer_t* lanczos_padding_layer_create(int padding);

// 嵌入层创建函数
layer_t* embedding_layer_create(int num_embeddings, int embedding_dim, 
                               int padding_idx, float max_norm, 
                               float norm_type, float scale_grad_by_freq, 
                               float sparse);

#endif // NN_LAYERS_H