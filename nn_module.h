#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "nn.h"
#include "tensor.h"
#include "tensor_autograd.h"
#include "nn_layers_autograd.h"

// PyTorch风格的nn.Module系统

typedef struct ModuleParameter {
    char* name;
    tensor_t* tensor;
    int requires_grad;
    struct ModuleParameter* next;
} ModuleParameter;

typedef struct ModuleBuffer {
    char* name;
    tensor_t* tensor;
    struct ModuleBuffer* next;
} ModuleBuffer;

typedef struct ModuleChild {
    char* name;
    struct nn_Module* module;
    struct ModuleChild* next;
} ModuleChild;

typedef struct nn_Module {
    char* name;
    int training;
    ModuleParameter* parameters;
    ModuleBuffer* buffers;
    ModuleChild* children;
    
    // 模块配置
    int _is_initialized;
    int _is_frozen;
    
    // 前向传播函数指针
    tensor_t* (*forward)(struct nn_Module* self, tensor_t* input);
    
    // 模块生命周期函数
    void (*_apply)(struct nn_Module* self, void (*fn)(ModuleParameter*));
    void (*train)(struct nn_Module* self, int mode);
    void (*eval)(struct nn_Module* self);
    void (*zero_grad)(struct nn_Module* self);
    void (*to)(struct nn_Module* self, DeviceType device);
    void (*cpu)(struct nn_Module* self);
    void (*cuda)(struct nn_Module* self, int device_id);
    
    // 参数管理
    size_t (*num_parameters)(struct nn_Module* self);
    ModuleParameter* (*parameters_list)(struct nn_Module* self);
    ModuleParameter* (*named_parameters)(struct nn_Module* self);
    
    // 状态管理
    void (*load_state_dict)(struct nn_Module* self, const char* state_dict_path);
    void (*save_state_dict)(struct nn_Module* self, const char* state_dict_path);
    
    // 模块注册
    void (*register_parameter)(struct nn_Module* self, const char* name, tensor_t* param, int requires_grad);
    void (*register_buffer)(struct nn_Module* self, const char* name, tensor_t* buffer);
    void (*register_module)(struct nn_Module* self, const char* name, struct nn_Module* child);
    
    // 子模块访问
    struct nn_Module* (*get_submodule)(struct nn_Module* self, const char* name);
    tensor_t* (*get_parameter)(struct nn_Module* self, const char* name);
    tensor_t* (*get_buffer)(struct nn_Module* self, const char* name);
    
    // 模块特殊功能
    void (*freeze)(struct nn_Module* self);
    void (*unfreeze)(struct nn_Module* self);
    int (*is_frozen)(struct nn_Module* self);
    
    // 内存管理
    void (*_apply_to_tensors)(struct nn_Module* self, void (*fn)(tensor_t*));
    
} nn_Module;

// 基础模块构造函数
nn_Module* nn_module_create(const char* name);
void nn_module_destroy(nn_Module* module);

// 模块容器类型
typedef struct nn_ModuleList {
    nn_Module base;
    nn_Module** modules;
    size_t num_modules;
    size_t capacity;
} nn_ModuleList;

nn_ModuleList* nn_modulelist_create(size_t capacity);
void nn_modulelist_add(nn_ModuleList* modulelist, nn_Module* module);
void nn_modulelist_destroy(nn_ModuleList* modulelist);

typedef struct nn_Sequential {
    nn_Module base;
    nn_Module** modules;
    size_t num_modules;
    size_t capacity;
} nn_Sequential;

nn_Sequential* nn_sequential_create(size_t capacity);
void nn_sequential_add(nn_Sequential* sequential, nn_Module* module);
AutogradTensor* nn_sequential_forward(nn_Sequential* sequential, AutogradTensor* input);
void nn_sequential_destroy(nn_Sequential* sequential);

// 常用层定义
typedef struct nn_Linear {
    nn_Module base;
    AutogradTensor* weight;
    AutogradTensor* bias;
    int in_features;
    int out_features;
    bool use_bias;
} nn_Linear;

nn_Linear* nn_linear_create(int in_features, int out_features, bool bias);
AutogradTensor* nn_linear_forward(nn_Linear* linear, AutogradTensor* input);
void nn_linear_reset_parameters(nn_Linear* linear);
void nn_linear_destroy(nn_Linear* linear);

typedef struct nn_Conv2d {
    nn_Module base;
    AutogradTensor* weight;
    AutogradTensor* bias;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    bool use_bias;
} nn_Conv2d;

nn_Conv2d* nn_conv2d_create(int in_channels, int out_channels, int kernel_size, 
                             int stride, int padding, bool bias);
AutogradTensor* nn_conv2d_forward(nn_Conv2d* conv, AutogradTensor* input);
void nn_conv2d_reset_parameters(nn_Conv2d* conv);
void nn_conv2d_destroy(nn_Conv2d* conv);

typedef struct nn_ReLU {
    nn_Module base;
    bool inplace;
} nn_ReLU;

nn_ReLU* nn_relu_create(bool inplace);
AutogradTensor* nn_relu_forward(nn_ReLU* relu, AutogradTensor* input);
void nn_relu_destroy(nn_ReLU* relu);

typedef struct nn_Sigmoid {
    nn_Module base;
} nn_Sigmoid;

nn_Sigmoid* nn_sigmoid_create(void);
AutogradTensor* nn_sigmoid_forward(nn_Sigmoid* sigmoid, AutogradTensor* input);
void nn_sigmoid_destroy(nn_Sigmoid* sigmoid);

typedef struct nn_Tanh {
    nn_Module base;
} nn_Tanh;

nn_Tanh* nn_tanh_create(void);
AutogradTensor* nn_tanh_forward(nn_Tanh* tanh, AutogradTensor* input);
void nn_tanh_destroy(nn_Tanh* tanh);

typedef struct nn_Dropout {
    nn_Module base;
    float p;
    bool training;
    bool inplace;
} nn_Dropout;

nn_Dropout* nn_dropout_create(float p, bool inplace);
AutogradTensor* nn_dropout_forward(nn_Dropout* dropout, AutogradTensor* input);
void nn_dropout_destroy(nn_Dropout* dropout);

typedef struct nn_BatchNorm2d {
    nn_Module base;
    AutogradTensor* weight;
    AutogradTensor* bias;
    AutogradTensor* running_mean;
    AutogradTensor* running_var;
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
} nn_BatchNorm2d;

nn_BatchNorm2d* nn_batchnorm2d_create(int num_features, float eps, float momentum, bool affine);
AutogradTensor* nn_batchnorm2d_forward(nn_BatchNorm2d* bn, AutogradTensor* input);
void nn_batchnorm2d_destroy(nn_BatchNorm2d* bn);

// 损失函数
typedef struct nn_MSELoss {
    nn_Module base;
    const char* reduction; // "mean", "sum", "none"
} nn_MSELoss;

nn_MSELoss* nn_mse_loss_create(const char* reduction);
AutogradTensor* nn_mse_loss_forward(nn_MSELoss* loss, AutogradTensor* input, AutogradTensor* target);
void nn_mse_loss_destroy(nn_MSELoss* loss);

typedef struct nn_CrossEntropyLoss {
    nn_Module base;
    AutogradTensor* weight;
    const char* reduction;
    int ignore_index;
} nn_CrossEntropyLoss;

nn_CrossEntropyLoss* nn_crossentropy_loss_create(const char* reduction);
AutogradTensor* nn_crossentropy_loss_forward(nn_CrossEntropyLoss* loss, AutogradTensor* input, AutogradTensor* target);
void nn_crossentropy_loss_destroy(nn_CrossEntropyLoss* loss);

// 优化器基类
typedef struct Optimizer {
    char* name;
    ModuleParameter* parameters;
    float learning_rate;
    size_t step_count;
    
    void (*step)(struct Optimizer* self);
    void (*zero_grad)(struct Optimizer* self);
    void (*state_dict)(struct Optimizer* self, const char* path);
    void (*load_state_dict)(struct Optimizer* self, const char* path);
    
    void (*destroy)(struct Optimizer* self);
} Optimizer;

// SGD优化器
typedef struct SGD {
    Optimizer base;
    float momentum;
    float dampening;
    float weight_decay;
    bool nesterov;
    AutogradTensor** momentum_buffers;
    size_t num_buffers;
} SGD;

SGD* sgd_create(ModuleParameter* parameters, float lr, float momentum, float weight_decay, bool nesterov);
void sgd_step(SGD* sgd);
void sgd_destroy(SGD* sgd);

// Adam优化器
typedef struct Adam {
    Optimizer base;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    AutogradTensor** exp_avg_buffers;
    AutogradTensor** exp_avg_sq_buffers;
    size_t num_buffers;
} Adam;

Adam* adam_create(ModuleParameter* parameters, float lr, float beta1, float beta2, float eps, float weight_decay);
void adam_step(Adam* adam);
void adam_destroy(Adam* adam);

// 学习率调度器
typedef struct LRScheduler {
    Optimizer* optimizer;
    float last_lr;
    size_t last_epoch;
    
    float (*get_lr)(struct LRScheduler* self, size_t epoch);
    void (*step)(struct LRScheduler* self);
    void (*destroy)(struct LRScheduler* self);
} LRScheduler;

// StepLR调度器
typedef struct StepLR {
    LRScheduler base;
    size_t step_size;
    float gamma;
} StepLR;

StepLR* steplr_create(Optimizer* optimizer, size_t step_size, float gamma);
void steplr_step(StepLR* scheduler);
void steplr_destroy(StepLR* scheduler);

#endif // NN_MODULE_H