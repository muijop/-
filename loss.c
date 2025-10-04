#include "loss.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

MSELoss* mse_loss_create(float reduction) {
    MSELoss* loss = (MSELoss*)malloc(sizeof(MSELoss));
    if (!loss) return NULL;
    
    loss->reduction = reduction;
    return loss;
}

void mse_loss_free(MSELoss* loss) {
    free(loss);
}

Tensor* mse_loss_forward(MSELoss* loss, Tensor* input, Tensor* target) {
    if (input->size != target->size) return NULL;
    
    size_t* result_shape = (size_t*)malloc(1 * sizeof(size_t));
    result_shape[0] = 1;
    Tensor* result = tensor_zeros(result_shape, 1, input->requires_grad || target->requires_grad);
    free(result_shape);
    
    if (!result) return NULL;
    
    float sum = 0.0f;
    for (size_t i = 0; i < input->size; i++) {
        float diff = input->data[i] - target->data[i];
        sum += diff * diff;
    }
    
    if (loss->reduction == 0.0f) {
        result->data[0] = sum;
    } else {
        result->data[0] = sum / input->size;
    }
    
    return result;
}

CrossEntropyLoss* cross_entropy_loss_create(float reduction) {
    CrossEntropyLoss* loss = (CrossEntropyLoss*)malloc(sizeof(CrossEntropyLoss));
    if (!loss) return NULL;
    
    loss->reduction = reduction;
    return loss;
}

void cross_entropy_loss_free(CrossEntropyLoss* loss) {
    free(loss);
}

Tensor* cross_entropy_loss_forward(CrossEntropyLoss* loss, Tensor* input, Tensor* target) {
    if (input->ndim != 2) return NULL;
    if (target->ndim != 1) return NULL;
    if (input->shape[0] != target->shape[0]) return NULL;
    
    size_t batch_size = input->shape[0];
    size_t num_classes = input->shape[1];
    
    size_t* result_shape = (size_t*)malloc(1 * sizeof(size_t));
    result_shape[0] = 1;
    Tensor* result = tensor_zeros(result_shape, 1, input->requires_grad || target->requires_grad);
    free(result_shape);
    
    if (!result) return NULL;
    
    float sum = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
        size_t target_class = (size_t)target->data[i];
        if (target_class >= num_classes) {
            tensor_free(result);
            return NULL;
        }
        
        float max_val = input->data[i * num_classes];
        for (size_t j = 1; j < num_classes; j++) {
            if (input->data[i * num_classes + j] > max_val) {
                max_val = input->data[i * num_classes + j];
            }
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < num_classes; j++) {
            sum_exp += exp(input->data[i * num_classes + j] - max_val);
        }
        
        float log_prob = log(sum_exp) + max_val - input->data[i * num_classes + target_class];
        sum += log_prob;
    }
    
    if (loss->reduction == 0.0f) {
        result->data[0] = sum;
    } else {
        result->data[0] = sum / batch_size;
    }
    
    return result;
}

BCELoss* bce_loss_create(float reduction) {
    BCELoss* loss = (BCELoss*)malloc(sizeof(BCELoss));
    if (!loss) return NULL;
    
    loss->reduction = reduction;
    return loss;
}

void bce_loss_free(BCELoss* loss) {
    free(loss);
}

Tensor* bce_loss_forward(BCELoss* loss, Tensor* input, Tensor* target) {
    if (input->size != target->size) return NULL;
    
    size_t* result_shape = (size_t*)malloc(1 * sizeof(size_t));
    result_shape[0] = 1;
    Tensor* result = tensor_zeros(result_shape, 1, input->requires_grad || target->requires_grad);
    free(result_shape);
    
    if (!result) return NULL;
    
    float sum = 0.0f;
    float eps = 1e-7f;
    
    for (size_t i = 0; i < input->size; i++) {
        float pred = input->data[i];
        float target_val = target->data[i];
        
        pred = fmaxf(eps, fminf(1.0f - eps, pred));
        
        float bce_loss = -(target_val * log(pred) + (1.0f - target_val) * log(1.0f - pred));
        sum += bce_loss;
    }
    
    if (loss->reduction == 0.0f) {
        result->data[0] = sum;
    } else {
        result->data[0] = sum / input->size;
    }
    
    return result;
}

MarginRankingLoss* margin_ranking_loss_create(float margin, float reduction) {
    MarginRankingLoss* loss = (MarginRankingLoss*)malloc(sizeof(MarginRankingLoss));
    if (!loss) return NULL;
    
    loss->margin = margin;
    loss->reduction = reduction;
    return loss;
}

void margin_ranking_loss_free(MarginRankingLoss* loss) {
    free(loss);
}

Tensor* margin_ranking_loss_forward(MarginRankingLoss* loss, Tensor* input1, Tensor* input2, Tensor* target) {
    if (input1->size != input2->size || input1->size != target->size) return NULL;
    
    size_t* result_shape = (size_t*)malloc(1 * sizeof(size_t));
    result_shape[0] = 1;
    Tensor* result = tensor_zeros(result_shape, 1, input1->requires_grad || input2->requires_grad || target->requires_grad);
    free(result_shape);
    
    if (!result) return NULL;
    
    float sum = 0.0f;
    for (size_t i = 0; i < input1->size; i++) {
        float diff = input1->data[i] - input2->data[i];
        float target_val = target->data[i];
        float margin_loss = fmaxf(0.0f, -target_val * diff + loss->margin);
        sum += margin_loss;
    }
    
    if (loss->reduction == 0.0f) {
        result->data[0] = sum;
    } else {
        result->data[0] = sum / input1->size;
    }
    
    return result;
}

L1Loss* l1_loss_create(float reduction) {
    L1Loss* loss = (L1Loss*)malloc(sizeof(L1Loss));
    if (!loss) return NULL;
    
    loss->reduction = reduction;
    return loss;
}

void l1_loss_free(L1Loss* loss) {
    free(loss);
}

Tensor* l1_loss_forward(L1Loss* loss, Tensor* input, Tensor* target) {
    if (input->size != target->size) return NULL;
    
    size_t* result_shape = (size_t*)malloc(1 * sizeof(size_t));
    result_shape[0] = 1;
    Tensor* result = tensor_zeros(result_shape, 1, input->requires_grad || target->requires_grad);
    free(result_shape);
    
    if (!result) return NULL;
    
    float sum = 0.0f;
    for (size_t i = 0; i < input->size; i++) {
        sum += fabs(input->data[i] - target->data[i]);
    }
    
    if (loss->reduction == 0.0f) {
        result->data[0] = sum;
    } else {
        result->data[0] = sum / input->size;
    }
    
    return result;
}

SmoothL1Loss* smooth_l1_loss_create(float beta, float reduction) {
    SmoothL1Loss* loss = (SmoothL1Loss*)malloc(sizeof(SmoothL1Loss));
    if (!loss) return NULL;
    
    loss->beta = beta;
    loss->reduction = reduction;
    return loss;
}

void smooth_l1_loss_free(SmoothL1Loss* loss) {
    free(loss);
}

Tensor* smooth_l1_loss_forward(SmoothL1Loss* loss, Tensor* input, Tensor* target) {
    if (input->size != target->size) return NULL;
    
    size_t* result_shape = (size_t*)malloc(1 * sizeof(size_t));
    result_shape[0] = 1;
    Tensor* result = tensor_zeros(result_shape, 1, input->requires_grad || target->requires_grad);
    free(result_shape);
    
    if (!result) return NULL;
    
    float sum = 0.0f;
    float beta = loss->beta;
    
    for (size_t i = 0; i < input->size; i++) {
        float diff = input->data[i] - target->data[i];
        float abs_diff = fabs(diff);
        
        float smooth_loss;
        if (abs_diff < beta) {
            smooth_loss = 0.5f * diff * diff / beta;
        } else {
            smooth_loss = abs_diff - 0.5f * beta;
        }
        
        sum += smooth_loss;
    }
    
    if (loss->reduction == 0.0f) {
        result->data[0] = sum;
    } else {
        result->data[0] = sum / input->size;
    }
    
    return result;
}