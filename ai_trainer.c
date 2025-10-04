#include "ai_trainer.h"
#include <time.h>

// ==================== 张量操作实现 ====================

tensor_t* tensor_create(const float* data, const size_t* shape, size_t ndim, bool requires_grad) {
    tensor_t* tensor = (tensor_t*)malloc(sizeof(tensor_t));
    if (!tensor) return NULL;
    
    // 计算总大小
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    
    tensor->data = (float*)calloc(size, sizeof(float));
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    
    // 如果提供了数据，则复制数据
    if (data) {
        memcpy(tensor->data, data, size * sizeof(float));
    }
    
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        free(tensor->data);
        free(tensor);
        return NULL;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->ndim = ndim;
    tensor->size = size;
    tensor->requires_grad = requires_grad;
    tensor->grad = NULL;
    tensor->grad_fn = NULL;
    
    return tensor;
}

void tensor_destroy(tensor_t* tensor) {
    if (!tensor) return;
    
    if (tensor->data) free(tensor->data);
    if (tensor->shape) free(tensor->shape);
    if (tensor->grad) free(tensor->grad);
    
    free(tensor);
}

tensor_t* tensor_zeros(const size_t* shape, size_t ndim, bool requires_grad) {
    tensor_t* tensor = tensor_create(NULL, shape, ndim, requires_grad);
    if (!tensor) return NULL;
    
    // 数据已经是0，因为calloc初始化为0
    return tensor;
}

tensor_t* tensor_ones(const size_t* shape, size_t ndim, bool requires_grad) {
    tensor_t* tensor = tensor_create(NULL, shape, ndim, requires_grad);
    if (!tensor) return NULL;
    
    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = 1.0f;
    }
    
    return tensor;
}

tensor_t* tensor_randn(const size_t* shape, size_t ndim, bool requires_grad) {
    tensor_t* tensor = tensor_create(NULL, shape, ndim, requires_grad);
    if (!tensor) return NULL;
    
    srand((unsigned int)time(NULL));
    
    for (size_t i = 0; i < tensor->size; i++) {
        // 简单的随机数生成（标准正态分布近似）
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        tensor->data[i] = z * 0.1f; // 缩小方差
    }
    
    return tensor;
}

void tensor_copy(tensor_t* dest, tensor_t* src) {
    if (!dest || !src || dest->size != src->size) return;
    
    memcpy(dest->data, src->data, src->size * sizeof(float));
}

tensor_t* tensor_create_like(tensor_t* tensor) {
    if (!tensor) return NULL;
    
    return tensor_create(NULL, tensor->shape, tensor->ndim, false);
}

// ==================== 神经网络层实现 ====================

linear_layer_t* linear_layer_create(int input_size, int output_size) {
    linear_layer_t* layer = (linear_layer_t*)malloc(sizeof(linear_layer_t));
    if (!layer) return NULL;
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->last_input = NULL;
    
    // 创建权重和偏置
    size_t weight_shape[] = {input_size, output_size};
    layer->weight = tensor_randn(weight_shape, 2, true);
    
    size_t bias_shape[] = {output_size};
    layer->bias = tensor_zeros(bias_shape, 1, true);
    
    return layer;
}

void linear_layer_destroy(linear_layer_t* layer) {
    if (!layer) return;
    
    if (layer->weight) tensor_destroy(layer->weight);
    if (layer->bias) tensor_destroy(layer->bias);
    if (layer->last_input) tensor_destroy(layer->last_input);
    
    free(layer);
}

tensor_t* linear_layer_forward(linear_layer_t* layer, tensor_t* input) {
    if (!layer || !input || input->ndim != 2) return NULL;
    
    size_t batch_size = input->shape[0];
    size_t output_shape[] = {batch_size, layer->output_size};
    tensor_t* output = tensor_create(NULL, output_shape, 2, true);
    
    // 保存输入用于反向传播
    if (layer->last_input) tensor_destroy(layer->last_input);
    layer->last_input = tensor_create(NULL, input->shape, input->ndim, false);
    tensor_copy(layer->last_input, input);
    
    // 矩阵乘法: output = input * weight + bias
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < layer->output_size; j++) {
            float sum = 0.0f;
            for (int i = 0; i < layer->input_size; i++) {
                sum += input->data[b * layer->input_size + i] * 
                       layer->weight->data[i * layer->output_size + j];
            }
            output->data[b * layer->output_size + j] = sum + layer->bias->data[j];
        }
    }
    
    return output;
}

tensor_t* linear_layer_backward(linear_layer_t* layer, tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->last_input) return NULL;
    
    size_t batch_size = layer->last_input->shape[0];
    
    // 初始化梯度
    if (!layer->weight->grad) {
        layer->weight->grad = (float*)calloc(layer->weight->size, sizeof(float));
    }
    if (!layer->bias->grad) {
        layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
    }
    
    // 计算权重梯度: dL/dW = input^T * grad_output
    for (size_t i = 0; i < layer->input_size; i++) {
        for (size_t j = 0; j < layer->output_size; j++) {
            float grad = 0.0f;
            for (size_t b = 0; b < batch_size; b++) {
                grad += layer->last_input->data[b * layer->input_size + i] * 
                        grad_output->data[b * layer->output_size + j];
            }
            layer->weight->grad[i * layer->output_size + j] = grad / batch_size;
        }
    }
    
    // 计算偏置梯度: dL/db = sum(grad_output, axis=0)
    for (size_t j = 0; j < layer->output_size; j++) {
        float grad = 0.0f;
        for (size_t b = 0; b < batch_size; b++) {
            grad += grad_output->data[b * layer->output_size + j];
        }
        layer->bias->grad[j] = grad / batch_size;
    }
    
    // 计算输入梯度: dL/dinput = grad_output * weight^T
    size_t input_shape[] = {batch_size, layer->input_size};
    tensor_t* grad_input = tensor_create(NULL, input_shape, 2, false);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < layer->input_size; i++) {
            float grad = 0.0f;
            for (size_t j = 0; j < layer->output_size; j++) {
                grad += grad_output->data[b * layer->output_size + j] * 
                        layer->weight->data[i * layer->output_size + j];
            }
            grad_input->data[b * layer->input_size + i] = grad;
        }
    }
    
    return grad_input;
}

relu_layer_t* relu_layer_create() {
    relu_layer_t* layer = (relu_layer_t*)malloc(sizeof(relu_layer_t));
    if (layer) {
        layer->last_input = NULL;
    }
    return layer;
}

void relu_layer_destroy(relu_layer_t* layer) {
    if (layer && layer->last_input) {
        tensor_destroy(layer->last_input);
    }
    free(layer);
}

tensor_t* relu_layer_forward(relu_layer_t* layer, tensor_t* input) {
    if (!input || !layer) return NULL;
    
    // 保存输入用于反向传播
    if (layer->last_input) {
        tensor_destroy(layer->last_input);
    }
    layer->last_input = tensor_create(NULL, input->shape, input->ndim, false);
    tensor_copy(layer->last_input, input);
    
    tensor_t* output = tensor_create(NULL, input->shape, input->ndim, true);
    
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = fmaxf(0.0f, input->data[i]);
    }
    
    return output;
}

tensor_t* relu_layer_backward(relu_layer_t* layer, tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->last_input) {
        fprintf(stderr, "错误：ReLU反向传播参数无效\n");
        return NULL;
    }
    
    // 验证维度一致性
    if (grad_output->size != layer->last_input->size) {
        fprintf(stderr, "错误：ReLU反向传播维度不匹配\n");
        return NULL;
    }
    
    tensor_t* grad_input = tensor_create_like(grad_output);
    if (!grad_input) {
        fprintf(stderr, "错误：无法创建ReLU反向传播梯度张量\n");
        return NULL;
    }
    
    // 数值稳定性：避免浮点数比较的精度问题
    const float RELU_EPSILON = 1e-8f;
    
    // 优化：使用更高效的循环和缓存友好访问
    for (size_t i = 0; i < grad_output->size; i++) {
        // 使用阈值判断，避免浮点数精度问题
        if (layer->last_input->data[i] > RELU_EPSILON) {
            grad_input->data[i] = grad_output->data[i];
        } else {
            grad_input->data[i] = 0.0f;
        }
    }
    
    return grad_input;
}

softmax_layer_t* softmax_layer_create() {
    softmax_layer_t* layer = (softmax_layer_t*)malloc(sizeof(softmax_layer_t));
    if (layer) {
        layer->last_output = NULL;
    }
    return layer;
}

void softmax_layer_destroy(softmax_layer_t* layer) {
    if (layer && layer->last_output) {
        tensor_destroy(layer->last_output);
    }
    free(layer);
}

tensor_t* softmax_layer_forward(softmax_layer_t* layer, tensor_t* input) {
    if (!input || input->ndim != 2 || !layer) return NULL;
    
    size_t batch_size = input->shape[0];
    size_t num_classes = input->shape[1];
    tensor_t* output = tensor_create(NULL, input->shape, input->ndim, true);
    
    for (size_t b = 0; b < batch_size; b++) {
        // 计算最大值（数值稳定性）
        float max_val = input->data[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (input->data[b * num_classes + c] > max_val) {
                max_val = input->data[b * num_classes + c];
            }
        }
        
        // 计算指数和
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; c++) {
            output->data[b * num_classes + c] = expf(input->data[b * num_classes + c] - max_val);
            sum_exp += output->data[b * num_classes + c];
        }
        
        // 归一化
        for (size_t c = 0; c < num_classes; c++) {
            output->data[b * num_classes + c] /= sum_exp;
        }
    }
    
    // 保存输出用于反向传播
    if (layer->last_output) {
        tensor_destroy(layer->last_output);
    }
    layer->last_output = tensor_create(NULL, output->shape, output->ndim, false);
    tensor_copy(layer->last_output, output);
    
    return output;
}

tensor_t* softmax_layer_backward(softmax_layer_t* layer, tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->last_output) {
        fprintf(stderr, "错误：Softmax反向传播参数无效\n");
        return NULL;
    }
    
    if (grad_output->ndim != 2 || layer->last_output->ndim != 2) {
        fprintf(stderr, "错误：Softmax反向传播输入维度无效\n");
        return NULL;
    }
    
    size_t batch_size = grad_output->shape[0];
    size_t num_classes = grad_output->shape[1];
    
    // 验证维度一致性
    if (batch_size != layer->last_output->shape[0] || 
        num_classes != layer->last_output->shape[1]) {
        fprintf(stderr, "错误：Softmax反向传播维度不匹配\n");
        return NULL;
    }
    
    tensor_t* grad_input = tensor_create_like(grad_output);
    if (!grad_input) {
        fprintf(stderr, "错误：无法创建Softmax反向传播梯度张量\n");
        return NULL;
    }
    
    // 优化：使用矩阵运算替代三重循环
    // grad_input = grad_output * (diag(s) - s * s^T)
    // 可以优化为：grad_input_i = s_i * (grad_output_i - sum_j(s_j * grad_output_j))
    
    for (size_t b = 0; b < batch_size; b++) {
        size_t base_idx = b * num_classes;
        
        // 计算加权和：sum_j(s_j * grad_output_j)
        float weighted_sum = 0.0f;
        for (size_t j = 0; j < num_classes; j++) {
            weighted_sum += layer->last_output->data[base_idx + j] * 
                           grad_output->data[base_idx + j];
        }
        
        // 计算每个类别的梯度
        for (size_t i = 0; i < num_classes; i++) {
            grad_input->data[base_idx + i] = layer->last_output->data[base_idx + i] * 
                                            (grad_output->data[base_idx + i] - weighted_sum);
        }
    }
    
    return grad_input;
}

// ==================== 优化器实现 ====================

sgd_optimizer_t* sgd_optimizer_create(float learning_rate, float momentum) {
    sgd_optimizer_t* optimizer = (sgd_optimizer_t*)malloc(sizeof(sgd_optimizer_t));
    if (!optimizer) return NULL;
    
    optimizer->learning_rate = learning_rate;
    optimizer->momentum = momentum;
    optimizer->velocity = NULL;
    optimizer->num_params = 0;
    
    return optimizer;
}

void sgd_optimizer_destroy(sgd_optimizer_t* optimizer) {
    if (!optimizer) return;
    
    if (optimizer->velocity) free(optimizer->velocity);
    free(optimizer);
}

void sgd_optimizer_step(sgd_optimizer_t* optimizer, tensor_t** params, int num_params) {
    if (!optimizer || !params || num_params <= 0) return;
    
    // 初始化速度
    if (!optimizer->velocity) {
        optimizer->num_params = 0;
        for (int i = 0; i < num_params; i++) {
            if (params[i] && params[i]->requires_grad) {
                optimizer->num_params += params[i]->size;
            }
        }
        
        optimizer->velocity = (float*)calloc(optimizer->num_params, sizeof(float));
        if (!optimizer->velocity) return;
    }
    
    // SGD更新
    int velocity_idx = 0;
    for (int i = 0; i < num_params; i++) {
        if (params[i] && params[i]->requires_grad && params[i]->grad) {
            for (int j = 0; j < params[i]->size; j++) {
                // 动量更新
                optimizer->velocity[velocity_idx] = 
                    optimizer->momentum * optimizer->velocity[velocity_idx] - 
                    optimizer->learning_rate * params[i]->grad[j];
                
                // 参数更新
                params[i]->data[j] += optimizer->velocity[velocity_idx];
                velocity_idx++;
            }
        }
    }
}

adam_optimizer_t* adam_optimizer_create(float learning_rate, float beta1, float beta2, float epsilon) {
    adam_optimizer_t* optimizer = (adam_optimizer_t*)malloc(sizeof(adam_optimizer_t));
    if (!optimizer) return NULL;
    
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->step = 0;
    optimizer->m = NULL;
    optimizer->v = NULL;
    optimizer->num_params = 0;
    
    return optimizer;
}

void adam_optimizer_destroy(adam_optimizer_t* optimizer) {
    if (!optimizer) return;
    
    if (optimizer->m) free(optimizer->m);
    if (optimizer->v) free(optimizer->v);
    free(optimizer);
}

void adam_optimizer_step(adam_optimizer_t* optimizer, tensor_t** params, int num_params) {
    if (!optimizer || !params || num_params <= 0) return;
    
    optimizer->step++;
    
    // 初始化动量
    if (!optimizer->m) {
        optimizer->num_params = 0;
        for (int i = 0; i < num_params; i++) {
            if (params[i] && params[i]->requires_grad) {
                optimizer->num_params += params[i]->size;
            }
        }
        
        optimizer->m = (float*)calloc(optimizer->num_params, sizeof(float));
        optimizer->v = (float*)calloc(optimizer->num_params, sizeof(float));
        if (!optimizer->m || !optimizer->v) return;
    }
    
    // Adam更新
    int param_idx = 0;
    for (int i = 0; i < num_params; i++) {
        if (params[i] && params[i]->requires_grad && params[i]->grad) {
            for (int j = 0; j < params[i]->size; j++) {
                // 计算一阶和二阶矩估计
                optimizer->m[param_idx] = optimizer->beta1 * optimizer->m[param_idx] + 
                                         (1.0f - optimizer->beta1) * params[i]->grad[j];
                optimizer->v[param_idx] = optimizer->beta2 * optimizer->v[param_idx] + 
                                         (1.0f - optimizer->beta2) * params[i]->grad[j] * params[i]->grad[j];
                
                // 偏差修正
                float m_hat = optimizer->m[param_idx] / (1.0f - powf(optimizer->beta1, optimizer->step));
                float v_hat = optimizer->v[param_idx] / (1.0f - powf(optimizer->beta2, optimizer->step));
                
                // 参数更新
                params[i]->data[j] -= optimizer->learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
                param_idx++;
            }
        }
    }
}

// ==================== 损失函数实现 ====================

mse_loss_t* mse_loss_create() {
    mse_loss_t* loss = (mse_loss_t*)malloc(sizeof(mse_loss_t));
    return loss;
}

void mse_loss_destroy(mse_loss_t* loss) {
    if (loss) {
        free(loss);
    }
}

float mse_loss_forward(mse_loss_t* loss, tensor_t* predictions, tensor_t* targets) {
    if (!predictions || !targets || predictions->size != targets->size) return 0.0f;
    
    float total_loss = 0.0f;
    for (int i = 0; i < predictions->size; i++) {
        float diff = predictions->data[i] - targets->data[i];
        total_loss += diff * diff;
    }
    
    return total_loss / predictions->size;
}

tensor_t* mse_loss_backward(mse_loss_t* loss, tensor_t* predictions, tensor_t* targets) {
    if (!predictions || !targets) return NULL;
    
    // 创建梯度张量
    tensor_t* grad = tensor_create_like(predictions);
    if (!grad) return NULL;
    
    // MSE梯度: dL/dpred = 2 * (pred - target) / n
    for (int i = 0; i < predictions->size; i++) {
        grad->data[i] = 2.0f * (predictions->data[i] - targets->data[i]) / predictions->size;
    }
    
    return grad;
}

cross_entropy_loss_t* cross_entropy_loss_create() {
    cross_entropy_loss_t* loss = (cross_entropy_loss_t*)malloc(sizeof(cross_entropy_loss_t));
    return loss;
}

void cross_entropy_loss_destroy(cross_entropy_loss_t* loss) {
    if (loss) {
        free(loss);
    }
}

float cross_entropy_loss_forward(cross_entropy_loss_t* loss, tensor_t* predictions, tensor_t* targets) {
    if (!predictions || !targets || predictions->size != targets->size) return 0.0f;
    
    float total_loss = 0.0f;
    for (int i = 0; i < predictions->size; i++) {
        // 交叉熵损失: -target * log(pred)
        if (predictions->data[i] > 1e-8f) { // 避免log(0)
            total_loss += -targets->data[i] * logf(predictions->data[i]);
        }
    }
    
    return total_loss / predictions->size;
}

tensor_t* cross_entropy_loss_backward(cross_entropy_loss_t* loss, tensor_t* predictions, tensor_t* targets) {
    if (!predictions || !targets) return NULL;
    
    // 创建梯度张量
    tensor_t* grad = tensor_create_like(predictions);
    if (!grad) return NULL;
    
    // 交叉熵梯度: dL/dpred = -target / pred
    for (int i = 0; i < predictions->size; i++) {
        if (predictions->data[i] > 1e-8f) {
            grad->data[i] = -targets->data[i] / predictions->data[i] / predictions->size;
        } else {
            grad->data[i] = 0.0f;
        }
    }
    
    return grad;
}

// ==================== 模型实现 ====================

sequential_model_t* sequential_model_create() {
    sequential_model_t* model = (sequential_model_t*)malloc(sizeof(sequential_model_t));
    if (!model) return NULL;
    
    model->layers = NULL;
    model->layer_types = NULL;
    model->num_layers = 0;
    
    return model;
}

void sequential_model_destroy(sequential_model_t* model) {
    if (!model) return;
    
    for (int i = 0; i < model->num_layers; i++) {
        switch (model->layer_types[i]) {
            case 0: // linear
                linear_layer_destroy((linear_layer_t*)model->layers[i]);
                break;
            case 1: // relu
                relu_layer_destroy((relu_layer_t*)model->layers[i]);
                break;
            case 2: // softmax
                softmax_layer_destroy((softmax_layer_t*)model->layers[i]);
                break;
        }
    }
    
    if (model->layers) free(model->layers);
    if (model->layer_types) free(model->layer_types);
    free(model);
}

void sequential_model_add_layer(sequential_model_t* model, void* layer, int layer_type) {
    if (!model || !layer) return;
    
    model->num_layers++;
    model->layers = realloc(model->layers, model->num_layers * sizeof(void*));
    model->layer_types = realloc(model->layer_types, model->num_layers * sizeof(int));
    
    model->layers[model->num_layers - 1] = layer;
    model->layer_types[model->num_layers - 1] = layer_type;
}

tensor_t* sequential_model_forward(sequential_model_t* model, tensor_t* input) {
    if (!model || !input) return NULL;
    
    tensor_t* current = input;
    
    for (int i = 0; i < model->num_layers; i++) {
        tensor_t* next = NULL;
        
        switch (model->layer_types[i]) {
            case 0: // linear
                next = linear_layer_forward((linear_layer_t*)model->layers[i], current);
                break;
            case 1: // relu
                next = relu_layer_forward((relu_layer_t*)model->layers[i], current);
                break;
            case 2: // softmax
                next = softmax_layer_forward((softmax_layer_t*)model->layers[i], current);
                break;
        }
        
        if (current != input) {
            tensor_destroy(current);
        }
        current = next;
    }
    
    return current;
}

void sequential_model_backward(sequential_model_t* model, tensor_t* grad_output) {
    if (!model || !grad_output) return;
    
    // 从输出层反向传播到输入层
    tensor_t* current_grad = grad_output;
    
    for (int i = model->num_layers - 1; i >= 0; i--) {
        switch (model->layer_types[i]) {
            case 0: // 线性层
                current_grad = linear_layer_backward((linear_layer_t*)model->layers[i], current_grad);
                break;
            case 1: // ReLU层
                current_grad = relu_layer_backward((relu_layer_t*)model->layers[i], current_grad);
                break;
            case 2: // Softmax层
                current_grad = softmax_layer_backward((softmax_layer_t*)model->layers[i], current_grad);
                break;
        }
    }
    
    // 销毁中间梯度张量
    if (current_grad != grad_output) {
        tensor_destroy(current_grad);
    }
}

// ==================== 训练器实现 ====================

trainer_t* trainer_create(sequential_model_t* model, int optimizer_type, int loss_type) {
    trainer_t* trainer = (trainer_t*)malloc(sizeof(trainer_t));
    if (!trainer) return NULL;
    
    trainer->model = model;
    trainer->optimizer_type = optimizer_type;
    trainer->loss_type = loss_type;
    
    // 创建优化器
    switch (optimizer_type) {
        case 0: // SGD
            trainer->optimizer = sgd_optimizer_create(0.01f, 0.9f);
            break;
        case 1: // Adam
            trainer->optimizer = adam_optimizer_create(0.001f, 0.9f, 0.999f, 1e-8f);
            break;
        default:
            trainer->optimizer = NULL;
    }
    
    // 创建损失函数
    switch (loss_type) {
        case 0: // MSE
            trainer->loss_function = mse_loss_create();
            break;
        case 1: // Cross Entropy
            trainer->loss_function = cross_entropy_loss_create();
            break;
        default:
            trainer->loss_function = NULL;
    }
    
    trainer->batch_size = 32;
    trainer->epochs = 10;
    trainer->training_loss = NULL;
    trainer->validation_loss = NULL;
    
    return trainer;
}

void trainer_destroy(trainer_t* trainer) {
    if (!trainer) return;
    
    // 销毁优化器
    switch (trainer->optimizer_type) {
        case 0: // SGD
            sgd_optimizer_destroy((sgd_optimizer_t*)trainer->optimizer);
            break;
        case 1: // Adam
            adam_optimizer_destroy((adam_optimizer_t*)trainer->optimizer);
            break;
    }
    
    // 销毁损失函数
    switch (trainer->loss_type) {
        case 0: // MSE
            mse_loss_destroy((mse_loss_t*)trainer->loss_function);
            break;
        case 1: // Cross Entropy
            cross_entropy_loss_destroy((cross_entropy_loss_t*)trainer->loss_function);
            break;
    }
    
    if (trainer->training_loss) free(trainer->training_loss);
    if (trainer->validation_loss) free(trainer->validation_loss);
    
    free(trainer);
}

void trainer_set_hyperparameters(trainer_t* trainer, int batch_size, int epochs) {
    if (!trainer) return;
    
    trainer->batch_size = batch_size;
    trainer->epochs = epochs;
}

// 收集模型中的所有参数
tensor_t** collect_model_parameters(sequential_model_t* model, int* num_params) {
    if (!model || !num_params) return NULL;
    
    // 计算参数总数
    *num_params = 0;
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layer_types[i] == 0) { // 线性层
            linear_layer_t* layer = (linear_layer_t*)model->layers[i];
            *num_params += 2; // 权重和偏置
        }
    }
    
    tensor_t** params = (tensor_t**)malloc(*num_params * sizeof(tensor_t*));
    if (!params) return NULL;
    
    int param_idx = 0;
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layer_types[i] == 0) { // 线性层
            linear_layer_t* layer = (linear_layer_t*)model->layers[i];
            params[param_idx++] = layer->weight;
            params[param_idx++] = layer->bias;
        }
    }
    
    return params;
}

void trainer_train(trainer_t* trainer, tensor_t** inputs, tensor_t** targets, int num_samples) {
    if (!trainer || !inputs || !targets || num_samples <= 0) return;
    
    printf("开始训练...\n");
    printf("样本数量: %d, 批次大小: %d, 训练轮数: %d\n", num_samples, trainer->batch_size, trainer->epochs);
    
    // 收集模型参数
    int num_params = 0;
    tensor_t** params = collect_model_parameters(trainer->model, &num_params);
    
    // 分配损失记录数组
    trainer->training_loss = (float*)malloc(trainer->epochs * sizeof(float));
    
    for (int epoch = 0; epoch < trainer->epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = (num_samples + trainer->batch_size - 1) / trainer->batch_size;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * trainer->batch_size;
            int end_idx = (batch + 1) * trainer->batch_size;
            if (end_idx > num_samples) end_idx = num_samples;
            int batch_size = end_idx - start_idx;
            
            float batch_loss = 0.0f;
            
            // 清空梯度
            for (int i = 0; i < num_params; i++) {
                if (params[i] && params[i]->grad) {
                    memset(params[i]->grad, 0, params[i]->size * sizeof(float));
                }
            }
            
            for (int i = start_idx; i < end_idx; i++) {
                // 前向传播
                tensor_t* prediction = sequential_model_forward(trainer->model, inputs[i]);
                
                // 计算损失
                float loss = 0.0f;
                tensor_t* loss_grad = NULL;
                
                switch (trainer->loss_type) {
                    case 0: // MSE
                        loss = mse_loss_forward((mse_loss_t*)trainer->loss_function, prediction, targets[i]);
                        // 计算损失梯度
                        loss_grad = mse_loss_backward((mse_loss_t*)trainer->loss_function, prediction, targets[i]);
                        break;
                    case 1: // Cross Entropy
                        loss = cross_entropy_loss_forward((cross_entropy_loss_t*)trainer->loss_function, prediction, targets[i]);
                        // 计算损失梯度
                        loss_grad = cross_entropy_loss_backward((cross_entropy_loss_t*)trainer->loss_function, prediction, targets[i]);
                        break;
                }
                
                batch_loss += loss;
                
                // 反向传播
                if (loss_grad) {
                    sequential_model_backward(trainer->model, loss_grad);
                    tensor_destroy(loss_grad);
                }
                
                tensor_destroy(prediction);
            }
            
            batch_loss /= batch_size;
            epoch_loss += batch_loss;
            
            // 参数更新
            switch (trainer->optimizer_type) {
                case 0: // SGD
                    sgd_optimizer_step((sgd_optimizer_t*)trainer->optimizer, params, num_params);
                    break;
                case 1: // Adam
                    adam_optimizer_step((adam_optimizer_t*)trainer->optimizer, params, num_params);
                    break;
            }
        }
        
        epoch_loss /= num_batches;
        trainer->training_loss[epoch] = epoch_loss;
        
        printf("轮次 %d/%d, 训练损失: %.6f\n", epoch + 1, trainer->epochs, epoch_loss);
    }
    
    free(params);
    printf("训练完成!\n");
}

// ==================== 模型保存/加载功能 ====================

int sequential_model_save(sequential_model_t* model, const char* filename) {
    if (!model || !filename) {
        fprintf(stderr, "错误：模型保存参数无效\n");
        return TRAINER_ERROR_INVALID_PARAM;
    }
    
    if (model->num_layers <= 0 || model->num_layers > MAX_MODEL_LAYERS) {
        fprintf(stderr, "错误：模型层数无效\n");
        return TRAINER_ERROR_INVALID_PARAM;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "错误：无法创建模型文件 %s\n", filename);
        return TRAINER_ERROR_FILE_OPERATION;
    }
    
    // 写入文件头（用于验证文件格式）
    const char* FILE_HEADER = "AI_MODEL_V1";
    fwrite(FILE_HEADER, sizeof(char), strlen(FILE_HEADER), file);
    
    // 写入模型基本信息
    if (fwrite(&model->num_layers, sizeof(int), 1, file) != 1) {
        fclose(file);
        fprintf(stderr, "错误：写入模型层数失败\n");
        return TRAINER_ERROR_FILE_OPERATION;
    }
    
    // 写入每层信息
    for (int i = 0; i < model->num_layers; i++) {
        if (i >= MAX_MODEL_LAYERS) {
            fclose(file);
            fprintf(stderr, "错误：模型层数超出限制\n");
            return TRAINER_ERROR_INVALID_PARAM;
        }
        
        // 写入层类型
        if (fwrite(&model->layer_types[i], sizeof(int), 1, file) != 1) {
            fclose(file);
            fprintf(stderr, "错误：写入层类型失败\n");
            return TRAINER_ERROR_FILE_OPERATION;
        }
        
        if (model->layer_types[i] == 0) { // 线性层
            linear_layer_t* layer = (linear_layer_t*)model->layers[i];
            if (!layer || !layer->weight || !layer->bias) {
                fclose(file);
                fprintf(stderr, "错误：线性层数据无效\n");
                return TRAINER_ERROR_INVALID_PARAM;
            }
            
            // 写入权重信息
            if (fwrite(&layer->input_size, sizeof(int), 1, file) != 1 ||
                fwrite(&layer->output_size, sizeof(int), 1, file) != 1) {
                fclose(file);
                fprintf(stderr, "错误：写入权重信息失败\n");
                return TRAINER_ERROR_FILE_OPERATION;
            }
            
            // 写入权重数据
            if (fwrite(layer->weight->data, sizeof(float), layer->weight->size, file) != layer->weight->size) {
                fclose(file);
                fprintf(stderr, "错误：写入权重数据失败\n");
                return TRAINER_ERROR_FILE_OPERATION;
            }
            
            // 写入偏置数据
            if (fwrite(layer->bias->data, sizeof(float), layer->bias->size, file) != layer->bias->size) {
                fclose(file);
                fprintf(stderr, "错误：写入偏置数据失败\n");
                return TRAINER_ERROR_FILE_OPERATION;
            }
        }
        // ReLU和Softmax层没有参数，不需要保存
    }
    
    // 写入文件尾（用于验证文件完整性）
    const char* FILE_FOOTER = "END_MODEL";
    fwrite(FILE_FOOTER, sizeof(char), strlen(FILE_FOOTER), file);
    
    if (fclose(file) != 0) {
        fprintf(stderr, "警告：关闭模型文件失败\n");
    }
    
    printf("模型已成功保存到文件：%s\n", filename);
    return TRAINER_SUCCESS;
}

sequential_model_t* sequential_model_load(const char* filename) {
    if (!filename) {
        fprintf(stderr, "错误：模型加载文件名无效\n");
        return NULL;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "错误：无法打开模型文件 %s\n", filename);
        return NULL;
    }
    
    // 验证文件头
    char header[12];
    if (fread(header, sizeof(char), 11, file) != 11) {
        fclose(file);
        fprintf(stderr, "错误：读取文件头失败\n");
        return NULL;
    }
    header[11] = '\0';
    
    if (strcmp(header, "AI_MODEL_V1") != 0) {
        fclose(file);
        fprintf(stderr, "错误：无效的模型文件格式\n");
        return NULL;
    }
    
    sequential_model_t* model = sequential_model_create();
    if (!model) {
        fclose(file);
        fprintf(stderr, "错误：创建模型失败\n");
        return NULL;
    }
    
    // 读取层数
    int num_layers;
    if (fread(&num_layers, sizeof(int), 1, file) != 1) {
        sequential_model_destroy(model);
        fclose(file);
        fprintf(stderr, "错误：读取模型层数失败\n");
        return NULL;
    }
    
    if (num_layers <= 0 || num_layers > MAX_MODEL_LAYERS) {
        sequential_model_destroy(model);
        fclose(file);
        fprintf(stderr, "错误：无效的模型层数 %d\n", num_layers);
        return NULL;
    }
    
    // 读取每层信息
    for (int i = 0; i < num_layers; i++) {
        int layer_type;
        if (fread(&layer_type, sizeof(int), 1, file) != 1) {
            sequential_model_destroy(model);
            fclose(file);
            fprintf(stderr, "错误：读取层类型失败\n");
            return NULL;
        }
        
        if (layer_type < 0 || layer_type > 2) {
            sequential_model_destroy(model);
            fclose(file);
            fprintf(stderr, "错误：无效的层类型 %d\n", layer_type);
            return NULL;
        }
        
        if (layer_type == 0) { // 线性层
            int input_size, output_size;
            if (fread(&input_size, sizeof(int), 1, file) != 1 ||
                fread(&output_size, sizeof(int), 1, file) != 1) {
                sequential_model_destroy(model);
                fclose(file);
                fprintf(stderr, "错误：读取线性层参数失败\n");
                return NULL;
            }
            
            if (input_size <= 0 || output_size <= 0) {
                sequential_model_destroy(model);
                fclose(file);
                fprintf(stderr, "错误：无效的线性层尺寸 %d->%d\n", input_size, output_size);
                return NULL;
            }
            
            linear_layer_t* layer = linear_layer_create(input_size, output_size);
            if (!layer) {
                sequential_model_destroy(model);
                fclose(file);
                fprintf(stderr, "错误：创建线性层失败\n");
                return NULL;
            }
            
            // 读取权重数据
            if (fread(layer->weight->data, sizeof(float), layer->weight->size, file) != layer->weight->size) {
                linear_layer_destroy(layer);
                sequential_model_destroy(model);
                fclose(file);
                fprintf(stderr, "错误：读取权重数据失败\n");
                return NULL;
            }
            
            // 读取偏置数据
            if (fread(layer->bias->data, sizeof(float), layer->bias->size, file) != layer->bias->size) {
                linear_layer_destroy(layer);
                sequential_model_destroy(model);
                fclose(file);
                fprintf(stderr, "错误：读取偏置数据失败\n");
                return NULL;
            }
            
            sequential_model_add_layer(model, layer, layer_type);
        } else if (layer_type == 1) { // ReLU层
            relu_layer_t* layer = relu_layer_create();
            if (!layer) {
                sequential_model_destroy(model);
                fclose(file);
                fprintf(stderr, "错误：创建ReLU层失败\n");
                return NULL;
            }
            sequential_model_add_layer(model, layer, layer_type);
        } else if (layer_type == 2) { // Softmax层
            softmax_layer_t* layer = softmax_layer_create();
            if (!layer) {
                sequential_model_destroy(model);
                fclose(file);
                fprintf(stderr, "错误：创建Softmax层失败\n");
                return NULL;
            }
            sequential_model_add_layer(model, layer, layer_type);
        }
    }
    
    // 验证文件尾
    char footer[10];
    if (fread(footer, sizeof(char), 9, file) != 9) {
        sequential_model_destroy(model);
        fclose(file);
        fprintf(stderr, "错误：读取文件尾失败\n");
        return NULL;
    }
    footer[9] = '\0';
    
    if (strcmp(footer, "END_MODEL") != 0) {
        sequential_model_destroy(model);
        fclose(file);
        fprintf(stderr, "错误：模型文件不完整\n");
        return NULL;
    }
    
    fclose(file);
    printf("模型已成功从文件 %s 加载\n", filename);
    return model;
}

// ==================== 梯度检查功能 ====================

float gradient_check(sequential_model_t* model, tensor_t* input, tensor_t* target, int loss_type, float epsilon) {
    if (!model || !input || !target) {
        fprintf(stderr, "错误：梯度检查参数无效\n");
        return -1.0f;
    }
    
    if (loss_type != 0 && loss_type != 1) {
        fprintf(stderr, "错误：无效的损失类型 %d\n", loss_type);
        return -1.0f;
    }
    
    if (epsilon <= 0.0f) {
        fprintf(stderr, "错误：无效的epsilon值 %f\n", epsilon);
        return -1.0f;
    }
    
    // 收集模型参数
    int num_params = 0;
    tensor_t** params = collect_model_parameters(model, &num_params);
    if (!params || num_params == 0) {
        fprintf(stderr, "错误：无法收集模型参数\n");
        return -1.0f;
    }
    
    float max_error = 0.0f;
    int checked_params = 0;
    
    // 创建损失函数对象（避免重复创建）
    void* loss_func = NULL;
    if (loss_type == 0) {
        loss_func = mse_loss_create();
    } else {
        loss_func = cross_entropy_loss_create();
    }
    
    if (!loss_func) {
        fprintf(stderr, "错误：创建损失函数失败\n");
        free(params);
        return -1.0f;
    }
    
    // 对每个参数进行梯度检查
    for (int param_idx = 0; param_idx < num_params; param_idx++) {
        tensor_t* param = params[param_idx];
        if (!param || !param->requires_grad || !param->grad) continue;
        
        for (int i = 0; i < param->size; i++) {
            // 保存原始参数值
            float original_value = param->data[i];
            
            // 计算数值梯度：f(x+ε) - f(x-ε) / (2ε)
            
            // f(x+ε)
            param->data[i] = original_value + epsilon;
            tensor_t* output_plus = sequential_model_forward(model, input);
            if (!output_plus) {
                fprintf(stderr, "错误：前向传播失败（x+ε）\n");
                free(params);
                return -1.0f;
            }
            
            float loss_plus = 0.0f;
            if (loss_type == 0) { // MSE
                loss_plus = mse_loss_forward((mse_loss_t*)loss_func, output_plus, target);
            } else { // Cross Entropy
                loss_plus = cross_entropy_loss_forward((cross_entropy_loss_t*)loss_func, output_plus, target);
            }
            tensor_destroy(output_plus);
            
            // f(x-ε)
            param->data[i] = original_value - epsilon;
            tensor_t* output_minus = sequential_model_forward(model, input);
            if (!output_minus) {
                fprintf(stderr, "错误：前向传播失败（x-ε）\n");
                free(params);
                return -1.0f;
            }
            
            float loss_minus = 0.0f;
            if (loss_type == 0) { // MSE
                loss_minus = mse_loss_forward((mse_loss_t*)loss_func, output_minus, target);
            } else { // Cross Entropy
                loss_minus = cross_entropy_loss_forward((cross_entropy_loss_t*)loss_func, output_minus, target);
            }
            tensor_destroy(output_minus);
            
            // 数值梯度
            float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
            
            // 恢复原始参数值
            param->data[i] = original_value;
            
            // 计算反向传播梯度
            tensor_t* output = sequential_model_forward(model, input);
            if (!output) {
                fprintf(stderr, "错误：前向传播失败（解析梯度）\n");
                free(params);
                return -1.0f;
            }
            
            tensor_t* loss_grad = NULL;
            if (loss_type == 0) { // MSE
                mse_loss_forward((mse_loss_t*)loss_func, output, target);
                loss_grad = mse_loss_backward((mse_loss_t*)loss_func, output, target);
            } else { // Cross Entropy
                cross_entropy_loss_forward((cross_entropy_loss_t*)loss_func, output, target);
                loss_grad = cross_entropy_loss_backward((cross_entropy_loss_t*)loss_func, output, target);
            }
            
            if (!loss_grad) {
                fprintf(stderr, "错误：损失反向传播失败\n");
                tensor_destroy(output);
                free(params);
                return -1.0f;
            }
            
            sequential_model_backward(model, loss_grad);
            
            // 解析梯度
            float analytical_grad = param->grad[i];
            
            // 计算相对误差（数值稳定性处理）
            float denominator = fmaxf(fabsf(numerical_grad), fabsf(analytical_grad));
            float error = 0.0f;
            if (denominator > 1e-10f) { // 避免除以零
                error = fabsf(numerical_grad - analytical_grad) / denominator;
            } else {
                error = fabsf(numerical_grad - analytical_grad);
            }
            
            if (error > max_error) {
                max_error = error;
            }
            
            checked_params++;
            
            tensor_destroy(output);
            tensor_destroy(loss_grad);
            
            // 清空梯度
            memset(param->grad, 0, param->size * sizeof(float));
        }
    }
    
    // 销毁损失函数
    if (loss_type == 0) {
        mse_loss_destroy((mse_loss_t*)loss_func);
    } else {
        cross_entropy_loss_destroy((cross_entropy_loss_t*)loss_func);
    }
    
    free(params);
    
    if (checked_params == 0) {
        fprintf(stderr, "警告：没有可检查的参数\n");
        return -1.0f;
    }
    
    printf("梯度检查完成：检查了 %d 个参数，最大相对误差：%.6e\n", checked_params, max_error);
    return max_error;
}