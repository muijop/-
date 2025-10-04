#include "model_builder.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// 内部辅助函数
static AutogradLayer* create_layer_from_config(const LayerConfig* config);
static void update_model_parameters(AutogradModel* model);
static void collect_model_gradients(AutogradModel* model);
static float calculate_batch_loss(AutogradModel* model, AutogradTensor* input, AutogradTensor* target);
static void shuffle_data(AutogradTensor* data, AutogradTensor* targets, int size);
static void split_data(AutogradTensor* data, AutogradTensor* targets, float split_ratio,
                      AutogradTensor** train_data, AutogradTensor** train_targets,
                      AutogradTensor** val_data, AutogradTensor** val_targets);

// 模型构建器实现
ModelBuilder* model_builder_create(const ModelBuilderConfig* config) {
    ModelBuilder* builder = (ModelBuilder*)malloc(sizeof(ModelBuilder));
    if (!builder) return NULL;
    
    builder->config = *config;
    builder->max_layers = 100;
    builder->layers = (AutogradLayer**)malloc(sizeof(AutogradLayer*) * builder->max_layers);
    builder->num_layers = 0;
    builder->input_shape = NULL;
    builder->is_compiled = false;
    
    if (!builder->layers) {
        free(builder);
        return NULL;
    }
    
    return builder;
}

void model_builder_destroy(ModelBuilder* builder) {
    if (!builder) return;
    
    // 释放层（但不销毁，因为编译后会转移给模型）
    if (!builder->is_compiled) {
        for (int i = 0; i < builder->num_layers; i++) {
            if (builder->layers[i]) {
                autograd_layer_destroy(builder->layers[i]);
            }
        }
    }
    
    if (builder->layers) {
        free(builder->layers);
    }
    
    if (builder->input_shape) {
        autograd_tensor_destroy(builder->input_shape);
    }
    
    free(builder);
}

ModelBuilder* model_builder_add_layer(ModelBuilder* builder, const LayerConfig* config) {
    if (!builder || !config || builder->is_compiled) return NULL;
    
    if (builder->num_layers >= builder->max_layers) {
        // 扩展层数组
        int new_max = builder->max_layers * 2;
        AutogradLayer** new_layers = (AutogradLayer**)realloc(builder->layers, sizeof(AutogradLayer*) * new_max);
        if (!new_layers) return NULL;
        
        builder->layers = new_layers;
        builder->max_layers = new_max;
    }
    
    AutogradLayer* layer = create_layer_from_config(config);
    if (!layer) return NULL;
    
    builder->layers[builder->num_layers++] = layer;
    return builder;
}

ModelBuilder* model_builder_add_dense(ModelBuilder* builder, int units, const char* activation, bool use_bias) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "Linear";
    config.name = "dense";
    config.use_bias = use_bias;
    config.activation = activation;
    config.trainable = true;
    
    // 创建Linear层参数
    AutogradLinearParams* params = (AutogradLinearParams*)malloc(sizeof(AutogradLinearParams));
    params->out_features = units;
    params->bias = use_bias;
    config.params = params;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    free(params);
    return result;
}

ModelBuilder* model_builder_add_conv2d(ModelBuilder* builder, int out_channels, int kernel_size, 
                                      int stride, int padding, const char* activation) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "Conv2d";
    config.name = "conv2d";
    config.activation = activation;
    config.trainable = true;
    
    // 创建Conv2d层参数
    AutogradConv2dParams* params = (AutogradConv2dParams*)malloc(sizeof(AutogradConv2dParams));
    params->out_channels = out_channels;
    params->kernel_size = kernel_size;
    params->stride = stride;
    params->padding = padding;
    params->bias = true;
    config.params = params;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    free(params);
    return result;
}

ModelBuilder* model_builder_add_lstm(ModelBuilder* builder, int hidden_size, int num_layers, 
                                    bool bidirectional, float dropout) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "LSTM";
    config.name = "lstm";
    config.trainable = true;
    config.dropout_rate = dropout;
    
    // 创建LSTM层参数
    AutogradLSTMParams* params = (AutogradLSTMParams*)malloc(sizeof(AutogradLSTMParams));
    params->hidden_size = hidden_size;
    params->num_layers = num_layers;
    params->bidirectional = bidirectional;
    params->dropout = dropout;
    config.params = params;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    free(params);
    return result;
}

ModelBuilder* model_builder_add_attention(ModelBuilder* builder, int embed_dim, int num_heads, float dropout) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "MultiheadAttention";
    config.name = "attention";
    config.trainable = true;
    config.dropout_rate = dropout;
    
    // 创建Attention层参数
    AutogradMultiheadAttentionParams* params = (AutogradMultiheadAttentionParams*)malloc(sizeof(AutogradMultiheadAttentionParams));
    params->embed_dim = embed_dim;
    params->num_heads = num_heads;
    params->dropout = dropout;
    config.params = params;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    free(params);
    return result;
}

ModelBuilder* model_builder_add_dropout(ModelBuilder* builder, float rate) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "Dropout";
    config.name = "dropout";
    config.dropout_rate = rate;
    config.trainable = false;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    return result;
}

ModelBuilder* model_builder_add_batch_norm(ModelBuilder* builder, int num_features, float eps, float momentum) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "BatchNorm2d";
    config.name = "batch_norm";
    config.trainable = true;
    config.use_bias = true;
    
    // 创建BatchNorm层参数
    AutogradBatchNorm2dParams* params = (AutogradBatchNorm2dParams*)malloc(sizeof(AutogradBatchNorm2dParams));
    params->num_features = num_features;
    params->eps = eps;
    params->momentum = momentum;
    params->affine = true;
    config.params = params;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    free(params);
    return result;
}

ModelBuilder* model_builder_add_pooling(ModelBuilder* builder, const char* type, int kernel_size, int stride) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = type; // "MaxPool2d" or "AvgPool2d"
    config.name = "pooling";
    config.trainable = false;
    
    // 创建池化层参数
    AutogradMaxPool2dParams* params = (AutogradMaxPool2dParams*)malloc(sizeof(AutogradMaxPool2dParams));
    params->kernel_size = kernel_size;
    params->stride = stride;
    params->padding = 0;
    config.params = params;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    free(params);
    return result;
}

ModelBuilder* model_builder_add_flatten(ModelBuilder* builder) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "Flatten";
    config.name = "flatten";
    config.trainable = false;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    return result;
}

ModelBuilder* model_builder_add_reshape(ModelBuilder* builder, const int* shape, int ndim) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = "Reshape";
    config.name = "reshape";
    config.trainable = false;
    
    // 创建Reshape层参数
    AutogradReshapeParams* params = (AutogradReshapeParams*)malloc(sizeof(AutogradReshapeParams));
    params->target_shape = (int*)malloc(sizeof(int) * ndim);
    memcpy(params->target_shape, shape, sizeof(int) * ndim);
    params->ndim = ndim;
    config.params = params;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    free(params->target_shape);
    free(params);
    return result;
}

ModelBuilder* model_builder_add_activation(ModelBuilder* builder, const char* activation) {
    if (!builder) return NULL;
    
    LayerConfig config = {0};
    config.type = activation; // "ReLU", "Sigmoid", "Tanh", etc.
    config.name = "activation";
    config.trainable = false;
    
    ModelBuilder* result = model_builder_add_layer(builder, &config);
    return result;
}

ModelBuilder* model_builder_set_input_shape(ModelBuilder* builder, const int* shape, int ndim) {
    if (!builder) return NULL;
    
    if (builder->input_shape) {
        autograd_tensor_destroy(builder->input_shape);
    }
    
    builder->input_shape = autograd_tensor_create(shape, ndim, DTYPE_FLOAT32, false);
    return builder;
}

AutogradModel* model_builder_compile(ModelBuilder* builder, const TrainingConfig* config) {
    if (!builder || !config || builder->is_compiled) return NULL;
    
    AutogradModel* model = autograd_model_create("model", &builder->config);
    if (!model) return NULL;
    
    // 添加所有层到模型
    for (int i = 0; i < builder->num_layers; i++) {
        autograd_model_add_layer(model, builder->layers[i]);
        // 转移所有权给模型
        builder->layers[i] = NULL;
    }
    
    // 设置训练配置
    model->train_config = *config;
    
    // 创建优化器
    AutogradOptimizer* optimizer = NULL;
    if (strcmp(config->optimizer, "SGD") == 0) {
        optimizer = (AutogradOptimizer*)autograd_sgd_optimizer_create(config->learning_rate);
    } else if (strcmp(config->optimizer, "Adam") == 0) {
        optimizer = (AutogradOptimizer*)autograd_adam_optimizer_create(config->learning_rate, 0.9f, 0.999f, 1e-8f);
    } else if (strcmp(config->optimizer, "RMSprop") == 0) {
        optimizer = (AutogradOptimizer*)autograd_rmsprop_optimizer_create(config->learning_rate, 0.99f, 1e-8f);
    }
    
    if (optimizer) {
        autograd_model_set_optimizer(model, optimizer);
    }
    
    // 创建损失函数
    AutogradLossFunction* loss_function = NULL;
    if (strcmp(config->loss_function, "MSELoss") == 0) {
        loss_function = (AutogradLossFunction*)autograd_mse_loss_create(true);
    } else if (strcmp(config->loss_function, "CrossEntropyLoss") == 0) {
        loss_function = (AutogradLossFunction*)autograd_cross_entropy_loss_create(NULL, -1, false);
    } else if (strcmp(config->loss_function, "BCELoss") == 0) {
        loss_function = (AutogradLossFunction*)autograd_bce_loss_create(NULL, NULL);
    }
    
    if (loss_function) {
        autograd_model_set_loss_function(model, loss_function);
    }
    
    builder->is_compiled = true;
    
    // 更新模型参数
    update_model_parameters(model);
    
    return model;
}

// 模型实现
AutogradModel* autograd_model_create(const char* name, const ModelBuilderConfig* config) {
    AutogradModel* model = (AutogradModel*)malloc(sizeof(AutogradModel));
    if (!model) return NULL;
    
    model->name = strdup(name);
    model->layers = (AutogradLayer**)malloc(sizeof(AutogradLayer*) * 100);
    model->num_layers = 0;
    model->optimizer = NULL;
    model->loss_function = NULL;
    model->config = *config;
    model->is_training = false;
    model->parameters = NULL;
    model->num_parameters = 0;
    model->parameter_data = NULL;
    
    // 初始化性能统计
    model->profiling.forward_time = 0.0;
    model->profiling.backward_time = 0.0;
    model->profiling.optimizer_time = 0.0;
    model->profiling.forward_count = 0;
    model->profiling.backward_count = 0;
    
    return model;
}

void autograd_model_destroy(AutogradModel* model) {
    if (!model) return;
    
    // 释放层
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            autograd_layer_destroy(model->layers[i]);
        }
    }
    
    if (model->layers) {
        free(model->layers);
    }
    
    if (model->optimizer) {
        autograd_optimizer_destroy(model->optimizer);
    }
    
    if (model->loss_function) {
        autograd_loss_function_destroy(model->loss_function);
    }
    
    if (model->parameters) {
        autograd_tensor_destroy(model->parameters);
    }
    
    if (model->parameter_data) {
        free(model->parameter_data);
    }
    
    free((void*)model->name);
    free(model);
}

void autograd_model_add_layer(AutogradModel* model, AutogradLayer* layer) {
    if (!model || !layer) return;
    
    if (model->num_layers >= 100) return; // 简化实现
    
    model->layers[model->num_layers++] = layer;
}

void autograd_model_set_optimizer(AutogradModel* model, AutogradOptimizer* optimizer) {
    if (!model) return;
    
    if (model->optimizer) {
        autograd_optimizer_destroy(model->optimizer);
    }
    
    model->optimizer = optimizer;
}

void autograd_model_set_loss_function(AutogradModel* model, AutogradLossFunction* loss_function) {
    if (!model) return;
    
    if (model->loss_function) {
        autograd_loss_function_destroy(model->loss_function);
    }
    
    model->loss_function = loss_function;
}

AutogradTensor* autograd_model_forward(AutogradModel* model, AutogradTensor* input) {
    if (!model || !input) return NULL;
    
    clock_t start = clock();
    
    AutogradTensor* current = input;
    
    // 前向传播通过所有层
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            AutogradTensor* output = autograd_layer_forward(model->layers[i], current);
            if (current != input) {
                autograd_tensor_destroy(current);
            }
            current = output;
        }
    }
    
    clock_t end = clock();
    model->profiling.forward_time += (double)(end - start) / CLOCKS_PER_SEC;
    model->profiling.forward_count++;
    
    return current;
}

AutogradTensor* autograd_model_backward(AutogradModel* model, AutogradTensor* loss) {
    if (!model || !loss) return NULL;
    
    clock_t start = clock();
    
    // 反向传播通过所有层（逆序）
    AutogradTensor* grad_output = loss;
    
    for (int i = model->num_layers - 1; i >= 0; i--) {
        if (model->layers[i]) {
            AutogradTensor* grad_input = autograd_layer_backward(model->layers[i], grad_output);
            if (grad_output != loss) {
                autograd_tensor_destroy(grad_output);
            }
            grad_output = grad_input;
        }
    }
    
    clock_t end = clock();
    model->profiling.backward_time += (double)(end - start) / CLOCKS_PER_SEC;
    model->profiling.backward_count++;
    
    return grad_output;
}

void autograd_model_zero_grad(AutogradModel* model) {
    if (!model) return;
    
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            autograd_layer_zero_grad(model->layers[i]);
        }
    }
}

void autograd_model_step(AutogradModel* model) {
    if (!model || !model->optimizer) return;
    
    clock_t start = clock();
    
    // 收集所有参数和梯度
    AutogradTensor** parameters = NULL;
    AutogradTensor** gradients = NULL;
    int param_count = 0;
    
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            AutogradTensor* layer_params = autograd_layer_get_parameters(model->layers[i]);
            AutogradTensor* layer_grads = autograd_layer_get_gradients(model->layers[i]);
            
            if (layer_params && layer_grads) {
                // 简化实现：这里应该正确处理参数和梯度的收集
                param_count++;
            }
        }
    }
    
    // 执行优化步骤
    if (param_count > 0) {
        autograd_optimizer_step(model->optimizer, parameters, gradients, param_count);
    }
    
    clock_t end = clock();
    model->profiling.optimizer_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    if (parameters) free(parameters);
    if (gradients) free(gradients);
}

void autograd_model_train(AutogradModel* model) {
    if (!model) return;
    
    model->is_training = true;
    
    // 设置所有层为训练模式
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            autograd_layer_train(model->layers[i]);
        }
    }
}

void autograd_model_eval(AutogradModel* model) {
    if (!model) return;
    
    model->is_training = false;
    
    // 设置所有层为评估模式
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            autograd_layer_eval(model->layers[i]);
        }
    }
}

int autograd_model_save(AutogradModel* model, const char* filepath) {
    if (!model || !filepath) return -1;
    
    FILE* file = fopen(filepath, "wb");
    if (!file) return -1;
    
    // 写入模型元数据
    fwrite(&model->num_layers, sizeof(int), 1, file);
    fwrite(&model->num_parameters, sizeof(int), 1, file);
    
    // 写入层配置和参数
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            // 简化实现：这里应该正确序列化层配置和参数
            // 实际实现需要更复杂的序列化逻辑
        }
    }
    
    fclose(file);
    return 0;
}

int autograd_model_load(AutogradModel* model, const char* filepath) {
    if (!model || !filepath) return -1;
    
    FILE* file = fopen(filepath, "rb");
    if (!file) return -1;
    
    // 读取模型元数据
    int num_layers;
    int num_parameters;
    fread(&num_layers, sizeof(int), 1, file);
    fread(&num_parameters, sizeof(int), 1, file);
    
    // 读取层配置和参数
    for (int i = 0; i < num_layers; i++) {
        // 简化实现：这里应该正确反序列化层配置和参数
        // 实际实现需要更复杂的反序列化逻辑
    }
    
    fclose(file);
    return 0;
}

void autograd_model_summary(AutogradModel* model) {
    if (!model) return;
    
    printf("Model: %s\n", model->name);
    printf("=================================================================\n");
    printf("Layer (type)                Output Shape              Param #   \n");
    printf("=================================================================\n");
    
    int total_params = 0;
    
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            const char* layer_type = autograd_layer_get_type(model->layers[i]);
            const int* output_shape = autograd_layer_get_output_shape(model->layers[i]);
            int num_params = autograd_layer_get_num_parameters(model->layers[i]);
            
            printf("%-24s  %-24s  %d\n", layer_type, "(None, ?)", num_params);
            total_params += num_params;
        }
    }
    
    printf("=================================================================\n");
    printf("Total params: %d\n", total_params);
    printf("Trainable params: %d\n", total_params); // 简化实现
    printf("Non-trainable params: 0\n");
    printf("=================================================================\n");
}

void autograd_model_print_parameters(AutogradModel* model) {
    if (!model) return;
    
    printf("Model Parameters:\n");
    printf("=================\n");
    
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            AutogradTensor* params = autograd_layer_get_parameters(model->layers[i]);
            if (params) {
                printf("Layer %d: ", i);
                autograd_tensor_print(params);
            }
        }
    }
}

AutogradTensor* autograd_model_predict(AutogradModel* model, AutogradTensor* input) {
    if (!model || !input) return NULL;
    
    autograd_model_eval(model);
    return autograd_model_forward(model, input);
}

float autograd_model_evaluate(AutogradModel* model, AutogradTensor* input, AutogradTensor* target) {
    if (!model || !input || !target || !model->loss_function) return 0.0f;
    
    autograd_model_eval(model);
    
    AutogradTensor* output = autograd_model_forward(model, input);
    if (!output) return 0.0f;
    
    AutogradTensor* loss = model->loss_function->forward(model->loss_function, output, target);
    float loss_value = 0.0f;
    
    if (loss && autograd_tensor_size(loss) == 1) {
        loss_value = autograd_tensor_data_float(loss)[0];
    }
    
    autograd_tensor_destroy(output);
    if (loss) autograd_tensor_destroy(loss);
    
    return loss_value;
}

// 训练循环实现
TrainingState* training_state_create(void) {
    TrainingState* state = (TrainingState*)malloc(sizeof(TrainingState));
    if (!state) return NULL;
    
    state->current_epoch = 0;
    state->current_step = 0;
    state->current_loss = 0.0f;
    state->best_loss = FLT_MAX;
    state->patience_counter = 0;
    state->early_stopped = false;
    state->last_checkpoint = NULL;
    
    return state;
}

void training_state_destroy(TrainingState* state) {
    if (!state) return;
    
    if (state->last_checkpoint) {
        autograd_tensor_destroy(state->last_checkpoint);
    }
    
    free(state);
}

void training_state_reset(TrainingState* state) {
    if (!state) return;
    
    state->current_epoch = 0;
    state->current_step = 0;
    state->current_loss = 0.0f;
    state->best_loss = FLT_MAX;
    state->patience_counter = 0;
    state->early_stopped = false;
}

float autograd_train_step(AutogradModel* model, AutogradTensor* input, AutogradTensor* target) {
    if (!model || !input || !target || !model->loss_function) return 0.0f;
    
    autograd_model_train(model);
    autograd_model_zero_grad(model);
    
    // 前向传播
    AutogradTensor* output = autograd_model_forward(model, input);
    if (!output) return 0.0f;
    
    // 计算损失
    AutogradTensor* loss = model->loss_function->forward(model->loss_function, output, target);
    float loss_value = 0.0f;
    
    if (loss && autograd_tensor_size(loss) == 1) {
        loss_value = autograd_tensor_data_float(loss)[0];
    }
    
    // 反向传播
    autograd_model_backward(model, loss);
    
    // 优化步骤
    autograd_model_step(model);
    
    // 清理
    autograd_tensor_destroy(output);
    if (loss) autograd_tensor_destroy(loss);
    
    return loss_value;
}

float autograd_validate_step(AutogradModel* model, AutogradTensor* input, AutogradTensor* target) {
    return autograd_model_evaluate(model, input, target);
}

void autograd_train_epoch(AutogradModel* model, AutogradTensor* train_data, AutogradTensor* train_targets,
                         int batch_size, TrainingState* state, TrainingCallback* callback) {
    if (!model || !train_data || !train_targets || !state) return;
    
    int num_samples = autograd_tensor_shape(train_data)[0];
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    float epoch_loss = 0.0f;
    
    // 打乱数据
    shuffle_data(train_data, train_targets, num_samples);
    
    // 调用epoch开始回调
    if (callback && callback->on_epoch_start) {
        callback->on_epoch_start(model, state, callback->user_data);
    }
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int start_idx = batch_idx * batch_size;
        int end_idx = (start_idx + batch_size < num_samples) ? start_idx + batch_size : num_samples;
        int current_batch_size = end_idx - start_idx;
        
        // 调用batch开始回调
        if (callback && callback->on_batch_start) {
            callback->on_batch_start(model, state, batch_idx, callback->user_data);
        }
        
        // 获取批次数据
        AutogradTensor* batch_data = autograd_tensor_slice(train_data, 0, start_idx, end_idx);
        AutogradTensor* batch_targets = autograd_tensor_slice(train_targets, 0, start_idx, end_idx);
        
        // 训练步骤
        float batch_loss = autograd_train_step(model, batch_data, batch_targets);
        epoch_loss += batch_loss;
        
        // 更新训练状态
        state->current_step++;
        state->current_loss = batch_loss;
        
        // 调用batch结束回调
        if (callback && callback->on_batch_end) {
            callback->on_batch_end(model, state, batch_idx, batch_loss, callback->user_data);
        }
        
        // 清理批次数据
        autograd_tensor_destroy(batch_data);
        autograd_tensor_destroy(batch_targets);
        
        // 日志输出
        if (state->current_step % model->train_config.log_interval == 0) {
            printf("Epoch %d, Step %d, Loss: %.6f\n", state->current_epoch, state->current_step, batch_loss);
        }
    }
    
    // 计算平均损失
    epoch_loss /= num_batches;
    state->current_loss = epoch_loss;
    
    // 调用epoch结束回调
    if (callback && callback->on_epoch_end) {
        callback->on_epoch_end(model, state, callback->user_data);
    }
}

void autograd_validate_epoch(AutogradModel* model, AutogradTensor* val_data, AutogradTensor* val_targets,
                           int batch_size, TrainingState* state) {
    if (!model || !val_data || !val_targets || !state) return;
    
    int num_samples = autograd_tensor_shape(val_data)[0];
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    float val_loss = 0.0f;
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int start_idx = batch_idx * batch_size;
        int end_idx = (start_idx + batch_size < num_samples) ? start_idx + batch_size : num_samples;
        
        // 获取验证批次数据
        AutogradTensor* batch_data = autograd_tensor_slice(val_data, 0, start_idx, end_idx);
        AutogradTensor* batch_targets = autograd_tensor_slice(val_targets, 0, start_idx, end_idx);
        
        // 验证步骤
        float batch_loss = autograd_validate_step(model, batch_data, batch_targets);
        val_loss += batch_loss;
        
        // 清理批次数据
        autograd_tensor_destroy(batch_data);
        autograd_tensor_destroy(batch_targets);
    }
    
    val_loss /= num_batches;
    
    printf("Validation Loss: %.6f\n", val_loss);
    
    // 早停检查
    if (val_loss < state->best_loss) {
        state->best_loss = val_loss;
        state->patience_counter = 0;
    } else {
        state->patience_counter++;
        
        // 检查早停条件
        if (state->patience_counter >= 10) { // 简化的早停条件
            state->early_stopped = true;
            printf("Early stopping triggered\n");
        }
    }
}

void autograd_train_model(AutogradModel* model, AutogradTensor* train_data, AutogradTensor* train_targets,
                        AutogradTensor* val_data, AutogradTensor* val_targets, TrainingState* state, TrainingCallback* callback) {
    if (!model || !train_data || !train_targets || !state) return;
    
    // 调用训练开始回调
    if (callback && callback->on_train_start) {
        callback->on_train_start(model, state, callback->user_data);
    }
    
    int epochs = model->train_config.epochs;
    int batch_size = model->train_config.batch_size;
    
    for (int epoch = 0; epoch < epochs && !state->early_stopped; epoch++) {
        state->current_epoch = epoch;
        
        printf("Epoch %d/%d\n", epoch + 1, epochs);
        printf("--------------------\n");
        
        // 训练轮次
        autograd_train_epoch(model, train_data, train_targets, batch_size, state, callback);
        
        // 验证轮次
        if (val_data && val_targets) {
            autograd_validate_epoch(model, val_data, val_targets, batch_size, state);
        }
        
        // 检查点保存
        if (model->train_config.checkpoint_interval > 0 && 
            (epoch + 1) % model->train_config.checkpoint_interval == 0) {
            char checkpoint_path[256];
            snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/checkpoint_epoch_%d.bin", 
                    model->config.checkpoint_dir, epoch + 1);
            autograd_model_save_checkpoint(model, state, checkpoint_path);
        }
    }
    
    // 调用训练结束回调
    if (callback && callback->on_train_end) {
        callback->on_train_end(model, state, callback->user_data);
    }
    
    printf("Training completed!\n");
}

// 配置函数实现
ModelBuilderConfig model_builder_config_default(void) {
    ModelBuilderConfig config = {0};
    config.use_autograd = true;
    config.use_kernels = true;
    config.mixed_precision = false;
    config.device_type = DEVICE_CPU;
    config.device_id = 0;
    config.enable_profiling = false;
    config.enable_checkpointing = false;
    config.checkpoint_dir = NULL;
    config.max_checkpoints = 5;
    return config;
}

ModelBuilderConfig model_builder_config_cpu(void) {
    ModelBuilderConfig config = model_builder_config_default();
    config.device_type = DEVICE_CPU;
    return config;
}

ModelBuilderConfig model_builder_config_gpu(int device_id) {
    ModelBuilderConfig config = model_builder_config_default();
    config.device_type = DEVICE_GPU;
    config.device_id = device_id;
    return config;
}

ModelBuilderConfig model_builder_config_mixed_precision(void) {
    ModelBuilderConfig config = model_builder_config_default();
    config.mixed_precision = true;
    return config;
}

ModelBuilderConfig model_builder_config_distributed(void) {
    ModelBuilderConfig config = model_builder_config_default();
    config.enable_checkpointing = true;
    return config;
}

void model_builder_config_set_checkpoint_dir(ModelBuilderConfig* config, const char* dir) {
    if (!config || !dir) return;
    
    if (config->checkpoint_dir) {
        free(config->checkpoint_dir);
    }
    
    config->checkpoint_dir = strdup(dir);
    config->enable_checkpointing = true;
}

void model_builder_config_set_profiling(ModelBuilderConfig* config, bool enable) {
    if (!config) return;
    config->enable_profiling = enable;
}

TrainingConfig training_config_default(void) {
    TrainingConfig config = {0};
    config.epochs = 10;
    config.batch_size = 32;
    config.learning_rate = 0.001f;
    config.optimizer = "Adam";
    config.loss_function = "CrossEntropyLoss";
    config.weight_decay = 0.0f;
    config.use_lr_scheduler = false;
    config.lr_scheduler = NULL;
    config.warmup_epochs = 0;
    config.gradient_clip_norm = 0.0f;
    config.use_amp = false;
    config.use_distributed = false;
    config.num_workers = 4;
    config.shuffle = true;
    config.pin_memory = false;
    config.validation_split = 0.2f;
    config.checkpoint_interval = 0;
    config.log_interval = 100;
    return config;
}

TrainingConfig training_config_sgd(float learning_rate, int epochs, int batch_size) {
    TrainingConfig config = training_config_default();
    config.optimizer = "SGD";
    config.learning_rate = learning_rate;
    config.epochs = epochs;
    config.batch_size = batch_size;
    return config;
}

TrainingConfig training_config_adam(float learning_rate, int epochs, int batch_size) {
    TrainingConfig config = training_config_default();
    config.optimizer = "Adam";
    config.learning_rate = learning_rate;
    config.epochs = epochs;
    config.batch_size = batch_size;
    return config;
}

TrainingConfig training_config_classification(int num_classes, int epochs, int batch_size) {
    TrainingConfig config = training_config_default();
    config.loss_function = "CrossEntropyLoss";
    config.epochs = epochs;
    config.batch_size = batch_size;
    return config;
}

TrainingConfig training_config_regression(int epochs, int batch_size) {
    TrainingConfig config = training_config_default();
    config.loss_function = "MSELoss";
    config.epochs = epochs;
    config.batch_size = batch_size;
    return config;
}

TrainingConfig training_config_sequence(int sequence_length, int epochs, int batch_size) {
    TrainingConfig config = training_config_default();
    config.optimizer = "Adam";
    config.loss_function = "CrossEntropyLoss";
    config.epochs = epochs;
    config.batch_size = batch_size;
    return config;
}

void training_config_set_optimizer(TrainingConfig* config, const char* optimizer, float learning_rate) {
    if (!config) return;
    config->optimizer = optimizer;
    config->learning_rate = learning_rate;
}

void training_config_set_loss_function(TrainingConfig* config, const char* loss_function) {
    if (!config) return;
    config->loss_function = loss_function;
}

void training_config_set_lr_scheduler(TrainingConfig* config, const char* scheduler, int warmup_epochs) {
    if (!config) return;
    config->lr_scheduler = scheduler;
    config->warmup_epochs = warmup_epochs;
    config->use_lr_scheduler = true;
}

void training_config_set_early_stopping(TrainingConfig* config, int patience, float min_delta) {
    if (!config) return;
    // 简化实现，实际应该在TrainingState中处理
}

void training_config_set_gradient_clipping(TrainingConfig* config, float max_norm) {
    if (!config) return;
    config->gradient_clip_norm = max_norm;
}

void training_config_set_distributed(TrainingConfig* config, bool enable, int num_workers) {
    if (!config) return;
    config->use_distributed = enable;
    config->num_workers = num_workers;
}

// 内部辅助函数实现
static AutogradLayer* create_layer_from_config(const LayerConfig* config) {
    if (!config || !config->type) return NULL;
    
    if (strcmp(config->type, "Linear") == 0) {
        AutogradLinearParams* params = (AutogradLinearParams*)config->params;
        AutogradLinear* linear = autograd_linear_create(0, params->out_features, params->bias); // in_features会在运行时确定
        return (AutogradLayer*)linear;
    }
    else if (strcmp(config->type, "Conv2d") == 0) {
        AutogradConv2dParams* params = (AutogradConv2dParams*)config->params;
        AutogradConv2d* conv = autograd_conv2d_create(0, params->out_channels, params->kernel_size, 
                                                       params->stride, params->padding, params->bias); // in_channels会在运行时确定
        return (AutogradLayer*)conv;
    }
    else if (strcmp(config->type, "ReLU") == 0) {
        return (AutogradLayer*)autograd_relu_create();
    }
    else if (strcmp(config->type, "Sigmoid") == 0) {
        return (AutogradLayer*)autograd_sigmoid_create();
    }
    else if (strcmp(config->type, "Tanh") == 0) {
        return (AutogradLayer*)autograd_tanh_create();
    }
    else if (strcmp(config->type, "Dropout") == 0) {
        return (AutogradLayer*)autograd_dropout_create(config->dropout_rate);
    }
    else if (strcmp(config->type, "Flatten") == 0) {
        return (AutogradLayer*)autograd_flatten_create();
    }
    
    return NULL;
}

static void update_model_parameters(AutogradModel* model) {
    if (!model) return;
    
    // 简化实现：收集所有参数
    int total_params = 0;
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) {
            total_params += autograd_layer_get_num_parameters(model->layers[i]);
        }
    }
    
    model->num_parameters = total_params;
}

static void collect_model_gradients(AutogradModel* model) {
    if (!model) return;
    
    // 简化实现：收集所有梯度
    // 实际实现需要更复杂的逻辑
}

static float calculate_batch_loss(AutogradModel* model, AutogradTensor* input, AutogradTensor* target) {
    if (!model || !input || !target || !model->loss_function) return 0.0f;
    
    AutogradTensor* output = autograd_model_forward(model, input);
    if (!output) return 0.0f;
    
    AutogradTensor* loss = model->loss_function->forward(model->loss_function, output, target);
    float loss_value = 0.0f;
    
    if (loss && autograd_tensor_size(loss) == 1) {
        loss_value = autograd_tensor_data_float(loss)[0];
    }
    
    autograd_tensor_destroy(output);
    if (loss) autograd_tensor_destroy(loss);
    
    return loss_value;
}

static void shuffle_data(AutogradTensor* data, AutogradTensor* targets, int size) {
    if (!data || !targets || size <= 1) return;
    
    // 简化实现：这里应该实现Fisher-Yates洗牌算法
    // 实际实现需要更复杂的逻辑来处理张量数据
}

static void split_data(AutogradTensor* data, AutogradTensor* targets, float split_ratio,
                      AutogradTensor** train_data, AutogradTensor** train_targets,
                      AutogradTensor** val_data, AutogradTensor** val_targets) {
    if (!data || !targets) return;
    
    int total_samples = autograd_tensor_shape(data)[0];
    int train_size = (int)(total_samples * (1.0f - split_ratio));
    int val_size = total_samples - train_size;
    
    // 简化实现：实际应该正确分割数据
    *train_data = autograd_tensor_slice(data, 0, 0, train_size);
    *train_targets = autograd_tensor_slice(targets, 0, 0, train_size);
    *val_data = autograd_tensor_slice(data, 0, train_size, total_samples);
    *val_targets = autograd_tensor_slice(targets, 0, train_size, total_samples);
}