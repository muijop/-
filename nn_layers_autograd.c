#include "nn_layers_autograd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// 层注册表
typedef struct LayerRegistry {
    const char* name;
    AutogradLayer* (*create_func)(void);
    struct LayerRegistry* next;
} LayerRegistry;

static LayerRegistry* g_layer_registry = NULL;

// 内部辅助函数
static float rand_uniform(float min, float max);
static float rand_normal(float mean, float std);
static void calculate_fan_in_out(AutogradTensor* tensor, int* fan_in, int* fan_out);

// 层管理函数
void autograd_layer_set_training(AutogradLayer* layer, bool training) {
    if (layer) {
        layer->training = training;
    }
}

void autograd_layer_set_kernel_optimization(AutogradLayer* layer, bool use_kernels) {
    if (layer) {
        layer->use_kernel_optimization = use_kernels;
    }
}

void autograd_layer_reset_parameters(AutogradLayer* layer) {
    if (layer && layer->reset_parameters) {
        layer->reset_parameters(layer);
    }
}

void autograd_layer_to_device(AutogradLayer* layer, DeviceType device) {
    if (layer && layer->to_device) {
        layer->to_device(layer, device);
    }
}

void autograd_layer_free(AutogradLayer* layer) {
    if (layer && layer->free) {
        layer->free(layer);
    }
}

// 全连接层实现
AutogradLinear* autograd_linear_create(int in_features, int out_features, bool bias) {
    AutogradLinear* layer = (AutogradLinear*)malloc(sizeof(AutogradLinear));
    if (!layer) return NULL;
    
    // 初始化基础层
    layer->base.training = true;
    layer->base.use_kernel_optimization = true;
    layer->base.name = strdup("Linear");
    layer->base.forward = autograd_linear_forward;
    layer->base.backward = autograd_linear_backward;
    layer->base.reset_parameters = (void (*)(AutogradLayer*))autograd_linear_reset_parameters;
    layer->base.to_device = (void (*)(AutogradLayer*, DeviceType))autograd_linear_to_device;
    layer->base.free = (void (*)(AutogradLayer*))autograd_linear_destroy;
    
    // 初始化参数
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->use_bias = bias;
    
    // 创建权重张量
    int weight_shape[] = {out_features, in_features};
    layer->weight = autograd_tensor_create(weight_shape, 2, DTYPE_FLOAT32, true);
    
    // 创建偏置张量
    if (bias) {
        int bias_shape[] = {out_features};
        layer->bias = autograd_tensor_create(bias_shape, 1, DTYPE_FLOAT32, true);
    } else {
        layer->bias = NULL;
    }
    
    // 重置参数
    autograd_linear_reset_parameters(layer);
    
    return layer;
}

void autograd_linear_destroy(AutogradLinear* layer) {
    if (layer) {
        if (layer->weight) {
            autograd_tensor_destroy(layer->weight);
        }
        if (layer->bias) {
            autograd_tensor_destroy(layer->bias);
        }
        if (layer->base.name) {
            free(layer->base.name);
        }
        free(layer);
    }
}

void autograd_linear_forward(AutogradLayer* base, AutogradTensor* input, AutogradTensor** output) {
    AutogradLinear* layer = (AutogradLinear*)base;
    if (!layer || !input || !output) return;
    
    // 执行线性变换: output = input @ weight.T + bias
    AutogradTensor* weight_T = autograd_tensor_transpose(layer->weight, 0, 1);
    AutogradTensor* matmul_result = autograd_tensor_matmul(input, weight_T);
    
    if (layer->use_bias && layer->bias) {
        *output = autograd_tensor_add(matmul_result, layer->bias);
    } else {
        *output = matmul_result;
    }
    
    // 清理中间结果
    if (weight_T != matmul_result) {
        autograd_tensor_destroy(weight_T);
    }
}

void autograd_linear_backward(AutogradLayer* base, AutogradTensor* grad_output, AutogradTensor* input, AutogradTensor** grad_input) {
    AutogradLinear* layer = (AutogradLinear*)base;
    if (!layer || !grad_output || !grad_input) return;
    
    // 计算输入梯度: grad_input = grad_output @ weight
    *grad_input = autograd_tensor_matmul(grad_output, layer->weight);
    
    // 计算权重梯度: grad_weight = grad_output.T @ input
    AutogradTensor* grad_output_T = autograd_tensor_transpose(grad_output, 0, 1);
    AutogradTensor* grad_weight = autograd_tensor_matmul(grad_output_T, input);
    
    // 更新权重梯度
    if (layer->weight->grad_node && layer->weight->grad_node->grad) {
        autograd_tensor_add(layer->weight->grad_node->grad, grad_weight);
    }
    
    // 计算偏置梯度
    if (layer->use_bias && layer->bias) {
        AutogradTensor* grad_bias = autograd_tensor_sum(grad_output, 0, true);
        if (layer->bias->grad_node && layer->bias->grad_node->grad) {
            autograd_tensor_add(layer->bias->grad_node->grad, grad_bias);
        }
        autograd_tensor_destroy(grad_bias);
    }
    
    // 清理中间结果
    autograd_tensor_destroy(grad_output_T);
    autograd_tensor_destroy(grad_weight);
}

void autograd_linear_reset_parameters(AutogradLinear* layer) {
    if (!layer || !layer->weight) return;
    
    // 使用Xavier初始化
    autograd_init_xavier_uniform(layer->weight);
    
    if (layer->use_bias && layer->bias) {
        autograd_init_constant(layer->bias, 0.0f);
    }
}

void autograd_linear_to_device(AutogradLinear* layer, DeviceType device) {
    if (!layer) return;
    
    if (layer->weight) {
        autograd_tensor_to_device(layer->weight, device);
    }
    if (layer->bias) {
        autograd_tensor_to_device(layer->bias, device);
    }
}

// 卷积层实现
AutogradConv2d* autograd_conv2d_create(int in_channels, int out_channels, int kernel_size, 
                                         int stride, int padding, int dilation, bool bias) {
    AutogradConv2d* layer = (AutogradConv2d*)malloc(sizeof(AutogradConv2d));
    if (!layer) return NULL;
    
    // 初始化基础层
    layer->base.training = true;
    layer->base.use_kernel_optimization = true;
    layer->base.name = strdup("Conv2d");
    layer->base.forward = autograd_conv2d_forward;
    layer->base.backward = autograd_conv2d_backward;
    layer->base.reset_parameters = (void (*)(AutogradLayer*))autograd_conv2d_reset_parameters;
    layer->base.to_device = (void (*)(AutogradLayer*, DeviceType))autograd_conv2d_to_device;
    layer->base.free = (void (*)(AutogradLayer*))autograd_conv2d_destroy;
    
    // 初始化参数
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->dilation = dilation;
    layer->use_bias = bias;
    
    // 创建权重张量
    int weight_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    layer->weight = autograd_tensor_create(weight_shape, 4, DTYPE_FLOAT32, true);
    
    // 创建偏置张量
    if (bias) {
        int bias_shape[] = {out_channels};
        layer->bias = autograd_tensor_create(bias_shape, 1, DTYPE_FLOAT32, true);
    } else {
        layer->bias = NULL;
    }
    
    // 重置参数
    autograd_conv2d_reset_parameters(layer);
    
    return layer;
}

void autograd_conv2d_destroy(AutogradConv2d* layer) {
    if (layer) {
        if (layer->weight) {
            autograd_tensor_destroy(layer->weight);
        }
        if (layer->bias) {
            autograd_tensor_destroy(layer->bias);
        }
        if (layer->base.name) {
            free(layer->base.name);
        }
        free(layer);
    }
}

void autograd_conv2d_forward(AutogradLayer* base, AutogradTensor* input, AutogradTensor** output) {
    AutogradConv2d* layer = (AutogradConv2d*)base;
    if (!layer || !input || !output) return;
    
    // 简化的卷积前向传播实现
    // 实际实现中会使用内核优化的卷积操作
    
    // 获取输入形状
    const int* input_shape = autograd_tensor_shape(input);
    int batch_size = input_shape[0];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    
    // 计算输出尺寸
    int out_h = (in_h + 2 * layer->padding - layer->dilation * (layer->kernel_size - 1) - 1) / layer->stride + 1;
    int out_w = (in_w + 2 * layer->padding - layer->dilation * (layer->kernel_size - 1) - 1) / layer->stride + 1;
    
    // 创建输出张量
    int output_shape[] = {batch_size, layer->out_channels, out_h, out_w};
    *output = autograd_tensor_create(output_shape, 4, DTYPE_FLOAT32, input->requires_grad);
    
    // 简化的卷积实现（实际会使用内核优化）
    float* input_data = autograd_tensor_data_float(input);
    float* weight_data = autograd_tensor_data_float(layer->weight);
    float* output_data = autograd_tensor_data_float(*output);
    
    // 这里应该调用内核优化的卷积函数
    // 现在使用简化的实现
    size_t output_size = batch_size * layer->out_channels * out_h * out_w;
    for (size_t i = 0; i < output_size; i++) {
        output_data[i] = 0.0f; // 简化实现
    }
    
    // 添加偏置
    if (layer->use_bias && layer->bias) {
        AutogradTensor* bias_expanded = autograd_tensor_reshape(layer->bias, (int[]){1, layer->out_channels, 1, 1}, 4);
        AutogradTensor* biased_output = autograd_tensor_add(*output, bias_expanded);
        autograd_tensor_destroy(*output);
        autograd_tensor_destroy(bias_expanded);
        *output = biased_output;
    }
}

void autograd_conv2d_backward(AutogradLayer* base, AutogradTensor* grad_output, AutogradTensor* input, AutogradTensor** grad_input) {
    AutogradConv2d* layer = (AutogradConv2d*)base;
    if (!layer || !grad_output || !grad_input) return;
    
    // 简化的卷积反向传播实现
    // 实际实现中会计算权重梯度、偏置梯度和输入梯度
    
    // 计算输入梯度
    *grad_input = autograd_tensor_create_like(input);
    
    // 计算权重梯度
    // 这里应该使用内核优化的卷积梯度计算
    
    // 计算偏置梯度
    if (layer->use_bias && layer->bias) {
        AutogradTensor* grad_bias = autograd_tensor_sum(grad_output, (int[]){0, 2, 3}, 3, true);
        if (layer->bias->grad_node && layer->bias->grad_node->grad) {
            autograd_tensor_add(layer->bias->grad_node->grad, grad_bias);
        }
        autograd_tensor_destroy(grad_bias);
    }
}

void autograd_conv2d_reset_parameters(AutogradConv2d* layer) {
    if (!layer || !layer->weight) return;
    
    // 使用Kaiming初始化
    autograd_init_kaiming_uniform(layer->weight, sqrtf(5.0f));
    
    if (layer->use_bias && layer->bias) {
        // 计算fan_in用于偏置初始化
        int fan_in, fan_out;
        calculate_fan_in_out(layer->weight, &fan_in, &fan_out);
        float bound = 1.0f / sqrtf(fan_in);
        autograd_init_uniform(layer->bias, -bound, bound);
    }
}

void autograd_conv2d_to_device(AutogradConv2d* layer, DeviceType device) {
    if (!layer) return;
    
    if (layer->weight) {
        autograd_tensor_to_device(layer->weight, device);
    }
    if (layer->bias) {
        autograd_tensor_to_device(layer->bias, device);
    }
}

// 批归一化层实现
AutogradBatchNorm2d* autograd_batch_norm2d_create(int num_features, float eps, float momentum, bool affine, bool track_running_stats) {
    AutogradBatchNorm2d* layer = (AutogradBatchNorm2d*)malloc(sizeof(AutogradBatchNorm2d));
    if (!layer) return NULL;
    
    // 初始化基础层
    layer->base.training = true;
    layer->base.use_kernel_optimization = true;
    layer->base.name = strdup("BatchNorm2d");
    layer->base.forward = autograd_batch_norm2d_forward;
    layer->base.backward = autograd_batch_norm2d_backward;
    layer->base.reset_parameters = (void (*)(AutogradLayer*))autograd_batch_norm2d_reset_parameters;
    layer->base.to_device = (void (*)(AutogradLayer*, DeviceType))autograd_batch_norm2d_to_device;
    layer->base.free = (void (*)(AutogradLayer*))autograd_batch_norm2d_destroy;
    
    // 初始化参数
    layer->num_features = num_features;
    layer->eps = eps;
    layer->momentum = momentum;
    layer->track_running_stats = track_running_stats;
    
    // 创建权重和偏置
    if (affine) {
        int param_shape[] = {num_features};
        layer->weight = autograd_tensor_create(param_shape, 1, DTYPE_FLOAT32, true);
        layer->bias = autograd_tensor_create(param_shape, 1, DTYPE_FLOAT32, true);
    } else {
        layer->weight = NULL;
        layer->bias = NULL;
    }
    
    // 创建运行统计
    if (track_running_stats) {
        int stats_shape[] = {num_features};
        layer->running_mean = autograd_tensor_create(stats_shape, 1, DTYPE_FLOAT32, false);
        layer->running_var = autograd_tensor_create(stats_shape, 1, DTYPE_FLOAT32, false);
        
        // 初始化运行统计
        autograd_init_constant(layer->running_mean, 0.0f);
        autograd_init_constant(layer->running_var, 1.0f);
    } else {
        layer->running_mean = NULL;
        layer->running_var = NULL;
    }
    
    // 重置参数
    autograd_batch_norm2d_reset_parameters(layer);
    
    return layer;
}

void autograd_batch_norm2d_destroy(AutogradBatchNorm2d* layer) {
    if (layer) {
        if (layer->weight) {
            autograd_tensor_destroy(layer->weight);
        }
        if (layer->bias) {
            autograd_tensor_destroy(layer->bias);
        }
        if (layer->running_mean) {
            autograd_tensor_destroy(layer->running_mean);
        }
        if (layer->running_var) {
            autograd_tensor_destroy(layer->running_var);
        }
        if (layer->base.name) {
            free(layer->base.name);
        }
        free(layer);
    }
}

void autograd_batch_norm2d_forward(AutogradLayer* base, AutogradTensor* input, AutogradTensor** output) {
    AutogradBatchNorm2d* layer = (AutogradBatchNorm2d*)base;
    if (!layer || !input || !output) return;
    
    // 简化的批归一化前向传播
    *output = autograd_tensor_create_like(input);
    
    // 实际实现中会计算均值、方差、归一化等
    // 这里使用简化的实现
    float* input_data = autograd_tensor_data_float(input);
    float* output_data = autograd_tensor_data_float(*output);
    size_t size = autograd_tensor_size(input);
    
    for (size_t i = 0; i < size; i++) {
        output_data[i] = input_data[i]; // 简化实现
    }
    
    // 应用仿射变换
    if (layer->weight && layer->bias) {
        AutogradTensor* scaled = autograd_tensor_multiply(*output, layer->weight);
        autograd_tensor_destroy(*output);
        *output = autograd_tensor_add(scaled, layer->bias);
        autograd_tensor_destroy(scaled);
    }
}

void autograd_batch_norm2d_backward(AutogradLayer* base, AutogradTensor* grad_output, AutogradTensor* input, AutogradTensor** grad_input) {
    AutogradBatchNorm2d* layer = (AutogradBatchNorm2d*)base;
    if (!layer || !grad_output || !grad_input) return;
    
    // 简化的批归一化反向传播
    *grad_input = autograd_tensor_create_like(input);
    
    // 计算权重和偏置梯度
    if (layer->weight && layer->bias) {
        // 这里应该计算实际的梯度
        // 简化实现
    }
}

void autograd_batch_norm2d_reset_parameters(AutogradBatchNorm2d* layer) {
    if (!layer) return;
    
    if (layer->weight) {
        autograd_init_constant(layer->weight, 1.0f);
    }
    if (layer->bias) {
        autograd_init_constant(layer->bias, 0.0f);
    }
}

void autograd_batch_norm2d_to_device(AutogradBatchNorm2d* layer, DeviceType device) {
    if (!layer) return;
    
    if (layer->weight) {
        autograd_tensor_to_device(layer->weight, device);
    }
    if (layer->bias) {
        autograd_tensor_to_device(layer->bias, device);
    }
    if (layer->running_mean) {
        autograd_tensor_to_device(layer->running_mean, device);
    }
    if (layer->running_var) {
        autograd_tensor_to_device(layer->running_var, device);
    }
}

// 权重初始化函数
void autograd_init_uniform(AutogradTensor* tensor, float a, float b) {
    if (!tensor) return;
    
    float* data = autograd_tensor_data_float(tensor);
    int size = autograd_tensor_size(tensor);
    
    for (int i = 0; i < size; i++) {
        data[i] = a + (b - a) * rand_uniform(0.0f, 1.0f);
    }
}

void autograd_init_normal(AutogradTensor* tensor, float mean, float std) {
    if (!tensor) return;
    
    float* data = autograd_tensor_data_float(tensor);
    int size = autograd_tensor_size(tensor);
    
    for (int i = 0; i < size; i++) {
        data[i] = mean + std * rand_normal(0.0f, 1.0f);
    }
}

void autograd_init_xavier_uniform(AutogradTensor* tensor) {
    if (!tensor) return;
    
    int fan_in, fan_out;
    calculate_fan_in_out(tensor, &fan_in, &fan_out);
    float std = sqrtf(2.0f / (fan_in + fan_out));
    float a = sqrtf(3.0f) * std;
    
    autograd_init_uniform(tensor, -a, a);
}

void autograd_init_xavier_normal(AutogradTensor* tensor) {
    if (!tensor) return;
    
    int fan_in, fan_out;
    calculate_fan_in_out(tensor, &fan_in, &fan_out);
    float std = sqrtf(2.0f / (fan_in + fan_out));
    
    autograd_init_normal(tensor, 0.0f, std);
}

void autograd_init_kaiming_uniform(AutogradTensor* tensor, float a) {
    if (!tensor) return;
    
    int fan_in, fan_out;
    calculate_fan_in_out(tensor, &fan_in, &fan_out);
    float std = sqrtf(2.0f / ((1.0f + a * a) * fan_in));
    float bound = sqrtf(3.0f) * std;
    
    autograd_init_uniform(tensor, -bound, bound);
}

void autograd_init_kaiming_normal(AutogradTensor* tensor, float a) {
    if (!tensor) return;
    
    int fan_in, fan_out;
    calculate_fan_in_out(tensor, &fan_in, &fan_out);
    float std = sqrtf(2.0f / ((1.0f + a * a) * fan_in));
    
    autograd_init_normal(tensor, 0.0f, std);
}

void autograd_init_constant(AutogradTensor* tensor, float val) {
    if (!tensor) return;
    
    float* data = autograd_tensor_data_float(tensor);
    int size = autograd_tensor_size(tensor);
    
    for (int i = 0; i < size; i++) {
        data[i] = val;
    }
}

void autograd_init_eye(AutogradTensor* tensor) {
    if (!tensor) return;
    
    const int* shape = autograd_tensor_shape(tensor);
    int ndim = autograd_tensor_ndim(tensor);
    
    if (ndim != 2 || shape[0] != shape[1]) return;
    
    float* data = autograd_tensor_data_float(tensor);
    int size = autograd_tensor_size(tensor);
    
    for (int i = 0; i < size; i++) {
        data[i] = 0.0f;
    }
    
    for (int i = 0; i < shape[0]; i++) {
        data[i * shape[1] + i] = 1.0f;
    }
}

// 内部辅助函数
static float rand_uniform(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

static float rand_normal(float mean, float std) {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare * std + mean;
    }
    
    has_spare = 1;
    
    float u = rand_uniform(0.0f, 1.0f);
    float v = rand_uniform(0.0f, 1.0f);
    
    float mag = std * sqrtf(-2.0f * logf(u));
    spare = mag * cosf(2.0f * M_PI * v);
    
    return mag * sinf(2.0f * M_PI * v) + mean;
}

static void calculate_fan_in_out(AutogradTensor* tensor, int* fan_in, int* fan_out) {
    if (!tensor || !fan_in || !fan_out) return;
    
    const int* shape = autograd_tensor_shape(tensor);
    int ndim = autograd_tensor_ndim(tensor);
    
    if (ndim == 2) {
        *fan_in = shape[1];
        *fan_out = shape[0];
    } else if (ndim >= 4) {
        int num_input_fmaps = shape[1];
        int num_output_fmaps = shape[0];
        int receptive_field_size = 1;
        
        for (int i = 2; i < ndim; i++) {
            receptive_field_size *= shape[i];
        }
        
        *fan_in = num_input_fmaps * receptive_field_size;
        *fan_out = num_output_fmaps * receptive_field_size;
    } else {
        *fan_in = *fan_out = autograd_tensor_size(tensor);
    }
}

// 层注册函数
void autograd_register_layer(const char* name, AutogradLayer* (*create_func)(void)) {
    if (!name || !create_func) return;
    
    LayerRegistry* entry = (LayerRegistry*)malloc(sizeof(LayerRegistry));
    if (!entry) return;
    
    entry->name = strdup(name);
    entry->create_func = create_func;
    entry->next = g_layer_registry;
    g_layer_registry = entry;
}

AutogradLayer* autograd_create_layer(const char* name) {
    if (!name) return NULL;
    
    LayerRegistry* current = g_layer_registry;
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current->create_func();
        }
        current = current->next;
    }
    
    return NULL;
}

void autograd_unregister_layer(const char* name) {
    if (!name) return;
    
    LayerRegistry** current = &g_layer_registry;
    while (*current) {
        if (strcmp((*current)->name, name) == 0) {
            LayerRegistry* to_remove = *current;
            *current = (*current)->next;
            free((void*)to_remove->name);
            free(to_remove);
            return;
        }
        current = &(*current)->next;
    }
}

// 初始化函数
void autograd_layers_init(void) {
    // 注册标准层类型
    autograd_register_layer("Linear", (AutogradLayer* (*)(void))autograd_linear_create);
    autograd_register_layer("Conv2d", (AutogradLayer* (*)(void))autograd_conv2d_create);
    autograd_register_layer("BatchNorm2d", (AutogradLayer* (*)(void))autograd_batch_norm2d_create);
    
    // 初始化随机数种子
    srand(time(NULL));
}

void autograd_layers_cleanup(void) {
    // 清理注册表
    while (g_layer_registry) {
        LayerRegistry* to_remove = g_layer_registry;
        g_layer_registry = g_layer_registry->next;
        free((void*)to_remove->name);
        free(to_remove);
    }
}