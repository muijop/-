#include "nn_module.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 基础模块实现

static AutogradTensor* nn_module_default_forward(nn_Module* self, AutogradTensor* input) {
    // 默认前向传播，子类需要重写
    return input;
}

static void nn_module_default_apply(nn_Module* self, void (*fn)(ModuleParameter*)) {
    ModuleParameter* param = self->parameters;
    while (param) {
        fn(param);
        param = param->next;
    }
    
    // 递归应用到子模块
    ModuleChild* child = self->children;
    while (child) {
        if (child->module && child->module->_apply) {
            child->module->_apply(child->module, fn);
        }
        child = child->next;
    }
}

static void nn_module_train_mode(nn_Module* self, bool mode) {
    self->training = mode;
    
    // 递归设置子模块
    ModuleChild* child = self->children;
    while (child) {
        if (child->module) {
            child->module->training = mode;
            if (child->module->train) {
                child->module->train(child->module, mode);
            }
        }
        child = child->next;
    }
}

static void nn_module_eval_mode(nn_Module* self) {
    nn_module_train_mode(self, false);
}

static void nn_module_zero_grad(nn_Module* self) {
    ModuleParameter* param = self->parameters;
    while (param) {
        if (param->tensor && param->tensor->grad_node) {
            autograd_tensor_zero_grad(param->tensor);
        }
        param = param->next;
    }
    
    // 递归清零子模块
    ModuleChild* child = self->children;
    while (child) {
        if (child->module && child->module->zero_grad) {
            child->module->zero_grad(child->module);
        }
        child = child->next;
    }
}

static void nn_module_to_device(nn_Module* self, DeviceType device) {
    // 移动所有参数到指定设备
    ModuleParameter* param = self->parameters;
    while (param) {
        if (param->tensor) {
            // autograd_tensor_to_device(param->tensor, device); // 暂时注释掉，因为函数未实现
        }
        param = param->next;
   }
    
    // 移动所有buffer
    ModuleBuffer* buffer = self->buffers;
    while (buffer) {
        if (buffer->tensor) {
            // autograd_tensor_to_device(buffer->tensor, device); // 暂时注释掉，因为函数未实现
        }
        buffer = buffer->next;
    }
    
    // 递归移动子模块
    ModuleChild* child = self->children;
    while (child) {
        if (child->module && child->module->to) {
            child->module->to(child->module, device);
        }
        child = child->next;
    }
}

static void nn_module_cpu(nn_Module* self) {
    nn_module_to_device(self, DEVICE_CPU);
}

static void nn_module_cuda(nn_Module* self, int device_id) {
    nn_module_to_device(self, DEVICE_CUDA);
}

static size_t nn_module_count_parameters(nn_Module* self) {
    size_t count = 0;
    ModuleParameter* param = self->parameters;
    while (param) {
        count++;
        param = param->next;
    }
    
    // 递归计算子模块参数
    ModuleChild* child = self->children;
    while (child) {
        if (child->module) {
            count += nn_module_count_parameters(child->module);
        }
        child = child->next;
    }
    
    return count;
}

static ModuleParameter* nn_module_get_parameters_list(nn_Module* self) {
    return self->parameters;
}

static ModuleParameter* nn_module_get_named_parameters(nn_Module* self) {
    return self->parameters;
}

static void nn_module_register_parameter(nn_Module* self, const char* name, AutogradTensor* param, bool requires_grad) {
    if (!self || !name || !param) return;
    
    ModuleParameter* new_param = (ModuleParameter*)malloc(sizeof(ModuleParameter));
    if (!new_param) return;
    
    new_param->name = strdup(name);
    new_param->tensor = param;
    new_param->requires_grad = requires_grad;
    new_param->next = NULL;
    
    // 添加到链表末尾
    if (!self->parameters) {
        self->parameters = new_param;
    } else {
        ModuleParameter* last = self->parameters;
        while (last->next) {
            last = last->next;
        }
        last->next = new_param;
    }
    
    // 设置requires_grad
    if (param && param->grad_node) {
        param->grad_node->requires_grad = requires_grad;
    }
}

static void nn_module_register_buffer(nn_Module* self, const char* name, AutogradTensor* buffer) {
    if (!self || !name || !buffer) return;
    
    ModuleBuffer* new_buffer = (ModuleBuffer*)malloc(sizeof(ModuleBuffer));
    if (!new_buffer) return;
    
    new_buffer->name = strdup(name);
    new_buffer->tensor = buffer;
    new_buffer->next = NULL;
    
    // 添加到链表末尾
    if (!self->buffers) {
        self->buffers = new_buffer;
    } else {
        ModuleBuffer* last = self->buffers;
        while (last->next) {
            last = last->next;
        }
        last->next = new_buffer;
    }
}

static void nn_module_register_module(nn_Module* self, const char* name, nn_Module* child) {
    if (!self || !name || !child) return;
    
    ModuleChild* new_child = (ModuleChild*)malloc(sizeof(ModuleChild));
    if (!new_child) return;
    
    new_child->name = strdup(name);
    new_child->module = child;
    new_child->next = NULL;
    
    // 添加到链表末尾
    if (!self->children) {
        self->children = new_child;
    } else {
        ModuleChild* last = self->children;
        while (last->next) {
            last = last->next;
        }
        last->next = new_child;
    }
}

static nn_Module* nn_module_get_submodule(nn_Module* self, const char* name) {
    if (!self || !name) return NULL;
    
    ModuleChild* child = self->children;
    while (child) {
        if (child->name && strcmp(child->name, name) == 0) {
            return child->module;
        }
        child = child->next;
    }
    
    return NULL;
}

static AutogradTensor* nn_module_get_parameter(nn_Module* self, const char* name) {
    if (!self || !name) return NULL;
    
    ModuleParameter* param = self->parameters;
    while (param) {
        if (param->name && strcmp(param->name, name) == 0) {
            return param->tensor;
        }
        param = param->next;
    }
    
    return NULL;
}

static AutogradTensor* nn_module_get_buffer(nn_Module* self, const char* name) {
    if (!self || !name) return NULL;
    
    ModuleBuffer* buffer = self->buffers;
    while (buffer) {
        if (buffer->name && strcmp(buffer->name, name) == 0) {
            return buffer->tensor;
        }
        buffer = buffer->next;
    }
    
    return NULL;
}

static void nn_module_freeze(nn_Module* self) {
    if (!self) return;
    
    self->_is_frozen = true;
    
    // 冻结所有参数
    ModuleParameter* param = self->parameters;
    while (param) {
        param->requires_grad = false;
        if (param->tensor && param->tensor->grad_node) {
            param->tensor->grad_node->requires_grad = false;
        }
        param = param->next;
    }
    
    // 递归冻结子模块
    ModuleChild* child = self->children;
    while (child) {
        if (child->module && child->module->freeze) {
            child->module->freeze(child->module);
        }
        child = child->next;
    }
}

static void nn_module_unfreeze(nn_Module* self) {
    if (!self) return;
    
    self->_is_frozen = false;
    
    // 解冻所有参数
    ModuleParameter* param = self->parameters;
    while (param) {
        param->requires_grad = true;
        if (param->tensor && param->tensor->grad_node) {
            param->tensor->grad_node->requires_grad = true;
        }
        param = param->next;
    }
    
    // 递归解冻子模块
    ModuleChild* child = self->children;
    while (child) {
        if (child->module && child->module->unfreeze) {
            child->module->unfreeze(child->module);
        }
        child = child->next;
    }
}

static bool nn_module_is_frozen(nn_Module* self) {
    return self ? self->_is_frozen : false;
}

static void nn_module_apply_to_tensors(nn_Module* self, void (*fn)(AutogradTensor*)) {
    if (!self || !fn) return;
    
    // 应用到所有参数
    ModuleParameter* param = self->parameters;
    while (param) {
        if (param->tensor) {
            fn(param->tensor);
        }
        param = param->next;
    }
    
    // 应用到所有buffer
    ModuleBuffer* buffer = self->buffers;
    while (buffer) {
        if (buffer->tensor) {
            fn(buffer->tensor);
        }
        buffer = buffer->next;
    }
    
    // 递归应用到子模块
    ModuleChild* child = self->children;
    while (child) {
        if (child->module && child->module->_apply_to_tensors) {
            child->module->_apply_to_tensors(child->module, fn);
        }
        child = child->next;
    }
}

nn_Module* nn_module_create(const char* name) {
    nn_Module* module = (nn_Module*)malloc(sizeof(nn_Module));
    if (!module) return NULL;
    
    module->name = name ? strdup(name) : strdup("Module");
    module->training = true;
    module->parameters = NULL;
    module->buffers = NULL;
    module->children = NULL;
    module->_is_initialized = false;
    module->_is_frozen = false;
    
    // 设置默认函数指针
    module->forward = nn_module_default_forward;
    module->_apply = nn_module_default_apply;
    module->train = nn_module_train_mode;
    module->eval = nn_module_eval_mode;
    module->zero_grad = nn_module_zero_grad;
    module->to = nn_module_to_device;
    module->cpu = nn_module_cpu;
    module->cuda = nn_module_cuda;
    module->num_parameters = nn_module_count_parameters;
    module->parameters_list = nn_module_get_parameters_list;
    module->named_parameters = nn_module_get_named_parameters;
    module->register_parameter = nn_module_register_parameter;
    module->register_buffer = nn_module_register_buffer;
    module->register_module = nn_module_register_module;
    module->get_submodule = nn_module_get_submodule;
    module->get_parameter = nn_module_get_parameter;
    module->get_buffer = nn_module_get_buffer;
    module->freeze = nn_module_freeze;
    module->unfreeze = nn_module_unfreeze;
    module->is_frozen = nn_module_is_frozen;
    module->_apply_to_tensors = nn_module_apply_to_tensors;
    
    module->_is_initialized = true;
    return module;
}

void nn_module_destroy(nn_Module* module) {
    if (!module) return;
    
    // 释放参数
    ModuleParameter* param = module->parameters;
    while (param) {
        ModuleParameter* next = param->next;
        free(param->name);
        // 注意：不释放tensor，因为它可能还在其他地方使用
        free(param);
        param = next;
    }
    
    // 释放buffer
    ModuleBuffer* buffer = module->buffers;
    while (buffer) {
        ModuleBuffer* next = buffer->next;
        free(buffer->name);
        free(buffer);
        buffer = next;
    }
    
    // 释放子模块
    ModuleChild* child = module->children;
    while (child) {
        ModuleChild* next = child->next;
        free(child->name);
        // 递归销毁子模块
        if (child->module) {
            nn_module_destroy(child->module);
        }
        free(child);
        child = next;
    }
    
    free(module->name);
    free(module);
}

// Sequential实现
AutogradTensor* nn_sequential_forward_impl(nn_Sequential* sequential, AutogradTensor* input) {
    if (!sequential || !input) return NULL;
    
    AutogradTensor* output = input;
    
    for (size_t i = 0; i < sequential->num_modules; i++) {
        nn_Module* module = sequential->modules[i];
        if (module && module->forward) {
            output = module->forward(module, output);
            if (!output) return NULL;
        }
    }
    
    return output;
}

nn_Sequential* nn_sequential_create(size_t capacity) {
    nn_Sequential* sequential = (nn_Sequential*)malloc(sizeof(nn_Sequential));
    if (!sequential) return NULL;
    
    // 初始化基类
    nn_Module* base = nn_module_create("Sequential");
    if (!base) {
        free(sequential);
        return NULL;
    }
    
    memcpy(&sequential->base, base, sizeof(nn_Module));
    sequential->base.forward = (AutogradTensor* (*)(nn_Module*, AutogradTensor*))nn_sequential_forward_impl;
    
    sequential->modules = (nn_Module**)malloc(capacity * sizeof(nn_Module*));
    if (!sequential->modules) {
        free(sequential);
        free(base);
        return NULL;
    }
    
    sequential->num_modules = 0;
    sequential->capacity = capacity;
    
    free(base); // 释放临时基类
    return sequential;
}

void nn_sequential_add(nn_Sequential* sequential, nn_Module* module) {
    if (!sequential || !module || sequential->num_modules >= sequential->capacity) return;
    
    sequential->modules[sequential->num_modules++] = module;
    
    // 注册为子模块
    char name[32];
    snprintf(name, sizeof(name), "%zu", sequential->num_modules - 1);
    sequential->base.register_module(&sequential->base, name, module);
}

void nn_sequential_destroy(nn_Sequential* sequential) {
    if (!sequential) return;
    
    // 销毁所有模块
    for (size_t i = 0; i < sequential->num_modules; i++) {
        if (sequential->modules[i]) {
            nn_module_destroy(sequential->modules[i]);
        }
    }
    
    free(sequential->modules);
    free(sequential);
}

// Linear层实现
AutogradTensor* nn_linear_forward_impl(nn_Linear* linear, AutogradTensor* input) {
    if (!linear || !input) return NULL;
    
    // 检查输入维度
    if (input->tensor->ndim != 2) {
        // 这里可以实现更复杂的维度处理
        return NULL;
    }
    
    // 执行线性变换: y = xW^T + b
    // AutogradTensor* result = autograd_tensor_matmul(input, linear->weight); // 需要autograd引擎，暂时简化
    AutogradTensor* result = input; // 简化实现
    if (!result) return NULL;
    
    // 添加偏置
    if (linear->use_bias && linear->bias) {
        // 这里应该实现广播加法
        // result = autograd_tensor_add(result, linear->bias);
    }
    
    return result;
}

void nn_linear_reset_parameters_impl(nn_Linear* linear) {
    if (!linear) return;
    
    // Xavier初始化
    float std = sqrtf(2.0f / (linear->in_features + linear->out_features));
    
    // 初始化权重
    if (linear->weight) {
        int size = linear->weight->tensor->size;
        float* data = linear->weight->tensor->data;
        
        for (int i = 0; i < size; i++) {
            data[i] = std * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }
    }
    
    // 初始化偏置
    if (linear->use_bias && linear->bias) {
        int size = linear->bias->tensor->size;
        float* data = linear->bias->tensor->data;
        
        for (int i = 0; i < size; i++) {
            data[i] = 0.0f;
        }
    }
}

nn_Linear* nn_linear_create(int in_features, int out_features, bool bias) {
    nn_Linear* linear = (nn_Linear*)malloc(sizeof(nn_Linear));
    if (!linear) return NULL;
    
    // 初始化基类
    nn_Module* base = nn_module_create("Linear");
    if (!base) {
        free(linear);
        return NULL;
    }
    
    memcpy(&linear->base, base, sizeof(nn_Module));
    linear->base.forward = (AutogradTensor* (*)(nn_Module*, AutogradTensor*))nn_linear_forward_impl;
    
    linear->in_features = in_features;
    linear->out_features = out_features;
    linear->use_bias = bias;
    
    // 创建权重张量
    int weight_shape[2] = {out_features, in_features};
    linear->weight = autograd_tensor_create(weight_shape, 2, true);
    if (!linear->weight) {
        free(linear);
        free(base);
        return NULL;
    }
    
    // 注册权重参数
    linear->base.register_parameter(&linear->base, "weight", linear->weight, true);
    
    // 创建偏置张量（如果需要）
    if (bias) {
        int bias_shape[1] = {out_features};
        linear->bias = autograd_tensor_create(bias_shape, 1, true);
        if (linear->bias) {
            linear->base.register_parameter(&linear->base, "bias", linear->bias, true);
        }
    } else {
        linear->bias = NULL;
    }
    
    // 重置参数
    nn_linear_reset_parameters_impl(linear);
    
    free(base);
    return linear;
}

void nn_linear_destroy(nn_Linear* linear) {
    if (!linear) return;
    
    // 注意：权重和偏置张量由基类管理，不需要单独释放
    nn_module_destroy(&linear->base);
    free(linear);
}

// ReLU激活函数实现
AutogradTensor* nn_relu_forward_impl(nn_ReLU* relu, AutogradTensor* input) {
    if (!relu || !input) return NULL;
    
    // 执行ReLU: max(0, x)
    return autograd_tensor_relu_simple(input); // 使用简化版本的ReLU函数
}

nn_ReLU* nn_relu_create(bool inplace) {
    nn_ReLU* relu = (nn_ReLU*)malloc(sizeof(nn_ReLU));
    if (!relu) return NULL;
    
    nn_Module* base = nn_module_create("ReLU");
    if (!base) {
        free(relu);
        return NULL;
    }
    
    memcpy(&relu->base, base, sizeof(nn_Module));
    relu->base.forward = (AutogradTensor* (*)(nn_Module*, AutogradTensor*))nn_relu_forward_impl;
    relu->inplace = inplace;
    
    free(base);
    return relu;
}

void nn_relu_destroy(nn_ReLU* relu) {
    if (!relu) return;
    nn_module_destroy(&relu->base);
    free(relu);
}

// Sigmoid激活函数实现
AutogradTensor* nn_sigmoid_forward_impl(nn_Sigmoid* sigmoid, AutogradTensor* input) {
    if (!sigmoid || !input) return NULL;
    
    // 执行Sigmoid: 1 / (1 + exp(-x))
    return autograd_tensor_sigmoid_simple(input); // 使用简化版本的Sigmoid函数
}

nn_Sigmoid* nn_sigmoid_create(void) {
    nn_Sigmoid* sigmoid = (nn_Sigmoid*)malloc(sizeof(nn_Sigmoid));
    if (!sigmoid) return NULL;
    
    nn_Module* base = nn_module_create("Sigmoid");
    if (!base) {
        free(sigmoid);
        return NULL;
    }
    
    memcpy(&sigmoid->base, base, sizeof(nn_Module));
    sigmoid->base.forward = (AutogradTensor* (*)(nn_Module*, AutogradTensor*))nn_sigmoid_forward_impl;
    
    free(base);
    return sigmoid;
}

void nn_sigmoid_destroy(nn_Sigmoid* sigmoid) {
    if (!sigmoid) return;
    nn_module_destroy(&sigmoid->base);
    free(sigmoid);
}

// Tanh激活函数实现
AutogradTensor* nn_tanh_forward_impl(nn_Tanh* tanh, AutogradTensor* input) {
    if (!tanh || !input) return NULL;
    
    // 执行Tanh: tanh(x)
    return autograd_tensor_tanh_simple(input); // 使用简化版本的Tanh函数
}

nn_Tanh* nn_tanh_create(void) {
    nn_Tanh* tanh = (nn_Tanh*)malloc(sizeof(nn_Tanh));
    if (!tanh) return NULL;
    
    nn_Module* base = nn_module_create("Tanh");
    if (!base) {
        free(tanh);
        return NULL;
    }
    
    memcpy(&tanh->base, base, sizeof(nn_Module));
    tanh->base.forward = (AutogradTensor* (*)(nn_Module*, AutogradTensor*))nn_tanh_forward_impl;
    
    free(base);
    return tanh;
}

void nn_tanh_destroy(nn_Tanh* tanh) {
    if (!tanh) return;
    nn_module_destroy(&tanh->base);
    free(tanh);
}

// Dropout实现
AutogradTensor* nn_dropout_forward_impl(nn_Dropout* dropout, AutogradTensor* input) {
    if (!dropout || !input) return NULL;
    
    if (!dropout->training || dropout->p <= 0.0f) {
        return input; // 训练模式下且p>0时才应用dropout
    }
    
    // 创建掩码
    int size = input->tensor->size;
    float* mask = (float*)malloc(size * sizeof(float));
    if (!mask) return NULL;
    
    float scale = 1.0f / (1.0f - dropout->p);
    
    for (int i = 0; i < size; i++) {
        mask[i] = ((float)rand() / RAND_MAX) > dropout->p ? scale : 0.0f;
    }
    
    // 应用掩码
    AutogradTensor* result = autograd_tensor_multiply_scalar(input, 1.0f); // 简化处理
    free(mask);
    
    return result;
}

nn_Dropout* nn_dropout_create(float p, bool inplace) {
    nn_Dropout* dropout = (nn_Dropout*)malloc(sizeof(nn_Dropout));
    if (!dropout) return NULL;
    
    nn_Module* base = nn_module_create("Dropout");
    if (!base) {
        free(dropout);
        return NULL;
    }
    
    memcpy(&dropout->base, base, sizeof(nn_Module));
    dropout->base.forward = (AutogradTensor* (*)(nn_Module*, AutogradTensor*))nn_dropout_forward_impl;
    dropout->p = p;
    dropout->training = true;
    dropout->inplace = inplace;
    
    free(base);
    return dropout;
}

void nn_dropout_destroy(nn_Dropout* dropout) {
    if (!dropout) return;
    nn_module_destroy(&dropout->base);
    free(dropout);
}

// BatchNorm2d实现
AutogradTensor* nn_batchnorm2d_forward_impl(nn_BatchNorm2d* bn, AutogradTensor* input) {
    if (!bn || !input) return NULL;
    
    // 检查输入维度 (N, C, H, W)
    if (input->tensor->ndim != 4) return NULL;
    
    int N = input->tensor->shape[0];
    int C = input->tensor->shape[1];
    int H = input->tensor->shape[2];
    int W = input->tensor->shape[3];
    
    // 计算每个通道的均值和方差
    float* mean = (float*)calloc(C, sizeof(float));
    float* var = (float*)calloc(C, sizeof(float));
    
    if (!mean || !var) {
        free(mean);
        free(var);
        return NULL;
    }
    
    // 计算均值
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    sum += input->tensor->data[idx];
                }
            }
        }
        mean[c] = sum / (N * H * W);
    }
    
    // 计算方差
    for (int c = 0; c < C; c++) {
        float sum_sq = 0.0f;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    float diff = input->tensor->data[idx] - mean[c];
                    sum_sq += diff * diff;
                }
            }
        }
        var[c] = sum_sq / (N * H * W);
    }
    
    // 创建输出张量
    AutogradTensor* output = autograd_tensor_create(input->tensor->shape, input->tensor->ndim, input->requires_grad);
    if (!output) {
        free(mean);
        free(var);
        return NULL;
    }
    
    // 归一化并应用缩放和偏移
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float weight_val = bn->affine ? bn->weight->tensor->data[c] : 1.0f;
            float bias_val = bn->affine ? bn->bias->tensor->data[c] : 0.0f;
            
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    float normalized = (input->tensor->data[idx] - mean[c]) / sqrtf(var[c] + bn->eps);
                    output->tensor->data[idx] = weight_val * normalized + bias_val;
                }
            }
        }
    }
    
    // 更新running stats（训练模式）
    if (bn->base.training && bn->track_running_stats) {
        for (int c = 0; c < C; c++) {
            bn->running_mean->tensor->data[c] = bn->momentum * mean[c] + (1.0f - bn->momentum) * bn->running_mean->tensor->data[c];
            bn->running_var->tensor->data[c] = bn->momentum * var[c] + (1.0f - bn->momentum) * bn->running_var->tensor->data[c];
        }
    }
    
    free(mean);
    free(var);
    return output;
}

nn_BatchNorm2d* nn_batchnorm2d_create(int num_features, float eps, float momentum, bool affine) {
    nn_BatchNorm2d* bn = (nn_BatchNorm2d*)malloc(sizeof(nn_BatchNorm2d));
    if (!bn) return NULL;
    
    nn_Module* base = nn_module_create("BatchNorm2d");
    if (!base) {
        free(bn);
        return NULL;
    }
    
    memcpy(&bn->base, base, sizeof(nn_Module));
    bn->base.forward = (AutogradTensor* (*)(nn_Module*, AutogradTensor*))nn_batchnorm2d_forward_impl;
    
    bn->num_features = num_features;
    bn->eps = eps;
    bn->momentum = momentum;
    bn->affine = affine;
    bn->track_running_stats = true;
    
    // 创建权重和偏置（如果需要）
    if (affine) {
        int shape[1] = {num_features};
        bn->weight = autograd_tensor_create(shape, 1, true);
        bn->bias = autograd_tensor_create(shape, 1, true);
        
        if (bn->weight && bn->bias) {
            bn->base.register_parameter(&bn->base, "weight", bn->weight, true);
            bn->base.register_parameter(&bn->base, "bias", bn->bias, true);
            
            // 初始化权重为1，偏置为0
            for (int i = 0; i < num_features; i++) {
                bn->weight->tensor->data[i] = 1.0f;
                bn->bias->tensor->data[i] = 0.0f;
            }
        }
    } else {
        bn->weight = NULL;
        bn->bias = NULL;
    }
    
    // 创建running stats
    int shape[1] = {num_features};
    bn->running_mean = autograd_tensor_create(shape, 1, false); // 不需要梯度
    bn->running_var = autograd_tensor_create(shape, 1, false);
    
    if (bn->running_mean && bn->running_var) {
        bn->base.register_buffer(&bn->base, "running_mean", bn->running_mean);
        bn->base.register_buffer(&bn->base, "running_var", bn->running_var);
        
        // 初始化running stats
        for (int i = 0; i < num_features; i++) {
            bn->running_mean->tensor->data[i] = 0.0f;
            bn->running_var->tensor->data[i] = 1.0f;
        }
    }
    
    free(base);
    return bn;
}

void nn_batchnorm2d_destroy(nn_BatchNorm2d* bn) {
    if (!bn) return;
    nn_module_destroy(&bn->base);
    free(bn);
}

// MSELoss实现
AutogradTensor* nn_mse_loss_forward_impl(nn_MSELoss* loss, AutogradTensor* input, AutogradTensor* target) {
    if (!loss || !input || !target) return NULL;
    
    // 检查形状匹配
    if (input->tensor->size != target->tensor->size) return NULL;
    
    // 计算平方差 (简化实现)
    AutogradTensor* squared = input; // 简化实现
    
    if (!squared) return NULL;
    
    // 根据reduction模式处理结果
    if (strcmp(loss->reduction, "mean") == 0) {
        float sum = 0.0f;
        for (int i = 0; i < squared->tensor->size; i++) {
            sum += squared->tensor->data[i];
        }
        
        autograd_tensor_free(squared);
        float result = sum / input->tensor->size;
        int shape[1] = {1};
        return autograd_tensor_create(shape, 1, true);
    } else if (strcmp(loss->reduction, "sum") == 0) {
        float sum = 0.0f;
        for (int i = 0; i < squared->tensor->size; i++) {
            sum += squared->tensor->data[i];
        }
        
        autograd_tensor_free(squared);
        int shape[1] = {1};
        return autograd_tensor_create(shape, 1, true);
    } else { // "none"
        return squared;
    }
}

nn_MSELoss* nn_mse_loss_create(const char* reduction) {
    nn_MSELoss* loss = (nn_MSELoss*)malloc(sizeof(nn_MSELoss));
    if (!loss) return NULL;
    
    nn_Module* base = nn_module_create("MSELoss");
    if (!base) {
        free(loss);
        return NULL;
    }
    
    memcpy(&loss->base, base, sizeof(nn_Module));
    loss->reduction = reduction ? strdup(reduction) : strdup("mean");
    
    free(base);
    return loss;
}

void nn_mse_loss_destroy(nn_MSELoss* loss) {
    if (!loss) return;
    free((void*)loss->reduction);
    nn_module_destroy(&loss->base);
    free(loss);
}

// SGD优化器实现
void sgd_step_impl(SGD* sgd) {
    if (!sgd || !sgd->base.parameters) return;
    
    ModuleParameter* param = sgd->base.parameters;
    size_t param_idx = 0;
    
    while (param && param_idx < sgd->num_buffers) {
        if (param->tensor && param->tensor->grad_node) {
            float* param_data = param->tensor->tensor->data;
            int size = param->tensor->tensor->size;
            
            // 权重衰减 (简化实现，跳过梯度计算)
            if (sgd->weight_decay != 0.0f) {
                for (int i = 0; i < size; i++) {
                    param_data[i] -= sgd->base.learning_rate * sgd->weight_decay * param_data[i];
                }
            }
            
            // 动量 (简化实现，跳过梯度计算)
            if (sgd->momentum != 0.0f) {
                AutogradTensor* momentum_buffer = sgd->momentum_buffers[param_idx];
                if (!momentum_buffer) {
                    // 创建动量缓冲区
                    momentum_buffer = autograd_tensor_create(param->tensor->tensor->shape, 
                                                           param->tensor->tensor->ndim, false);
                    sgd->momentum_buffers[param_idx] = momentum_buffer;
                }
                
                float* momentum_data = momentum_buffer->tensor->data;
                
                for (int i = 0; i < size; i++) {
                    // 简化动量更新，跳过梯度
                    momentum_data[i] = sgd->momentum * momentum_data[i];
                    
                    if (sgd->nesterov) {
                        param_data[i] -= sgd->base.learning_rate * momentum_data[i];
                    } else {
                        param_data[i] -= sgd->base.learning_rate * momentum_data[i];
                    }
                }
            } else {
                // 无动量，跳过更新
            }
        }
        
        param = param->next;
        param_idx++;
    }
    
    sgd->base.step_count++;
}

SGD* sgd_create(ModuleParameter* parameters, float lr, float momentum, float weight_decay, bool nesterov) {
    SGD* sgd = (SGD*)malloc(sizeof(SGD));
    if (!sgd) return NULL;
    
    sgd->base.name = strdup("SGD");
    sgd->base.parameters = parameters;
    sgd->base.learning_rate = lr;
    sgd->base.step_count = 0;
    sgd->base.step = (void (*)(Optimizer*))sgd_step_impl;
    sgd->base.zero_grad = (void (*)(Optimizer*))nn_module_zero_grad;
    sgd->base.destroy = (void (*)(Optimizer*))sgd_destroy;
    
    sgd->momentum = momentum;
    sgd->dampening = 0.0f;
    sgd->weight_decay = weight_decay;
    sgd->nesterov = nesterov;
    
    // 计算参数数量
    size_t num_params = 0;
    ModuleParameter* param = parameters;
    while (param) {
        num_params++;
        param = param->next;
    }
    
    sgd->num_buffers = num_params;
    sgd->momentum_buffers = (AutogradTensor**)calloc(num_params, sizeof(AutogradTensor*));
    
    if (!sgd->momentum_buffers) {
        free(sgd);
        return NULL;
    }
    
    return sgd;
}



// 优化器通用函数
void optimizer_zero_grad_impl(Optimizer* optimizer) {
    if (!optimizer || !optimizer->parameters) return;
    
    ModuleParameter* param = optimizer->parameters;
    while (param) {
        if (param->tensor && param->tensor->grad_node) {
            autograd_tensor_zero_grad(param->tensor);
        }
        param = param->next;
    }
}