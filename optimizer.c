#include "optimizer.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

SGD* sgd_create(float learning_rate) {
    SGD* optimizer = (SGD*)malloc(sizeof(SGD));
    if (!optimizer) return NULL;
    
    optimizer->learning_rate = learning_rate;
    return optimizer;
}

void sgd_free(SGD* optimizer) {
    free(optimizer);
}

void sgd_step(SGD* optimizer, Tensor** params, Tensor** grads, size_t param_count) {
    for (size_t i = 0; i < param_count; i++) {
        if (!params[i] || !grads[i]) continue;
        if (!params[i]->requires_grad) continue;
        
        for (size_t j = 0; j < params[i]->size; j++) {
            if (params[i]->grad) {
                params[i]->data[j] -= optimizer->learning_rate * grads[i]->data[j];
            }
        }
    }
}

Adam* adam_create(float learning_rate, float beta1, float beta2, float eps) {
    Adam* optimizer = (Adam*)malloc(sizeof(Adam));
    if (!optimizer) return NULL;
    
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->eps = eps;
    optimizer->t = 0;
    optimizer->param_count = 0;
    optimizer->m = NULL;
    optimizer->v = NULL;
    
    return optimizer;
}

void adam_free(Adam* optimizer) {
    if (optimizer) {
        if (optimizer->m) {
            for (size_t i = 0; i < optimizer->param_count; i++) {
                tensor_free(optimizer->m[i]);
            }
            free(optimizer->m);
        }
        if (optimizer->v) {
            for (size_t i = 0; i < optimizer->param_count; i++) {
                tensor_free(optimizer->v[i]);
            }
            free(optimizer->v);
        }
        free(optimizer);
    }
}

void adam_step(Adam* optimizer, Tensor** params, Tensor** grads, size_t param_count) {
    if (optimizer->param_count != param_count) {
        adam_free(optimizer);
        optimizer->param_count = param_count;
        optimizer->m = (Tensor**)malloc(param_count * sizeof(Tensor*));
        optimizer->v = (Tensor**)malloc(param_count * sizeof(Tensor*));
        
        if (!optimizer->m || !optimizer->v) {
            free(optimizer->m);
            free(optimizer->v);
            return;
        }
        
        for (size_t i = 0; i < param_count; i++) {
            if (params[i]) {
                optimizer->m[i] = tensor_zeros(params[i]->shape, params[i]->ndim, false);
                optimizer->v[i] = tensor_zeros(params[i]->shape, params[i]->ndim, false);
            } else {
                optimizer->m[i] = NULL;
                optimizer->v[i] = NULL;
            }
        }
    }
    
    optimizer->t++;
    float lr_t = optimizer->learning_rate * sqrt(1.0f - pow(optimizer->beta2, optimizer->t)) / (1.0f - pow(optimizer->beta1, optimizer->t));
    
    for (size_t i = 0; i < param_count; i++) {
        if (!params[i] || !grads[i]) continue;
        if (!params[i]->requires_grad) continue;
        if (!optimizer->m[i] || !optimizer->v[i]) continue;
        
        for (size_t j = 0; j < params[i]->size; j++) {
            float g = grads[i]->data[j];
            
            optimizer->m[i]->data[j] = optimizer->beta1 * optimizer->m[i]->data[j] + (1.0f - optimizer->beta1) * g;
            optimizer->v[i]->data[j] = optimizer->beta2 * optimizer->v[i]->data[j] + (1.0f - optimizer->beta2) * g * g;
            
            float m_corrected = optimizer->m[i]->data[j] / (1.0f - pow(optimizer->beta1, optimizer->t));
            float v_corrected = optimizer->v[i]->data[j] / (1.0f - pow(optimizer->beta2, optimizer->t));
            
            params[i]->data[j] -= lr_t * m_corrected / (sqrt(v_corrected) + optimizer->eps);
        }
    }
}

MomentumSGD* momentum_sgd_create(float learning_rate, float momentum, float dampening, float weight_decay) {
    MomentumSGD* optimizer = (MomentumSGD*)malloc(sizeof(MomentumSGD));
    if (!optimizer) return NULL;
    
    optimizer->learning_rate = learning_rate;
    optimizer->momentum = momentum;
    optimizer->dampening = dampening;
    optimizer->weight_decay = weight_decay;
    optimizer->param_count = 0;
    optimizer->momentum_buffers = NULL;
    
    return optimizer;
}

void momentum_sgd_free(MomentumSGD* optimizer) {
    if (optimizer) {
        if (optimizer->momentum_buffers) {
            for (size_t i = 0; i < optimizer->param_count; i++) {
                tensor_free(optimizer->momentum_buffers[i]);
            }
            free(optimizer->momentum_buffers);
        }
        free(optimizer);
    }
}

void momentum_sgd_step(MomentumSGD* optimizer, Tensor** params, Tensor** grads, size_t param_count) {
    if (optimizer->param_count != param_count) {
        momentum_sgd_free(optimizer);
        optimizer->param_count = param_count;
        optimizer->momentum_buffers = (Tensor**)malloc(param_count * sizeof(Tensor*));
        
        if (!optimizer->momentum_buffers) return;
        
        for (size_t i = 0; i < param_count; i++) {
            if (params[i]) {
                optimizer->momentum_buffers[i] = tensor_zeros(params[i]->shape, params[i]->ndim, false);
            } else {
                optimizer->momentum_buffers[i] = NULL;
            }
        }
    }
    
    for (size_t i = 0; i < param_count; i++) {
        if (!params[i] || !grads[i]) continue;
        if (!params[i]->requires_grad) continue;
        if (!optimizer->momentum_buffers[i]) continue;
        
        for (size_t j = 0; j < params[i]->size; j++) {
            float grad = grads[i]->data[j];
            
            if (optimizer->weight_decay != 0) {
                grad += optimizer->weight_decay * params[i]->data[j];
            }
            
            if (optimizer->momentum != 0) {
                optimizer->momentum_buffers[i]->data[j] = 
                    optimizer->momentum * optimizer->momentum_buffers[i]->data[j] + 
                    (1.0f - optimizer->dampening) * grad;
                
                grad = optimizer->momentum_buffers[i]->data[j];
            }
            
            params[i]->data[j] -= optimizer->learning_rate * grad;
        }
    }
}

RMSprop* rmsprop_create(float learning_rate, float alpha, float eps, float weight_decay, float momentum, float centered) {
    RMSprop* optimizer = (RMSprop*)malloc(sizeof(RMSprop));
    if (!optimizer) return NULL;
    
    optimizer->learning_rate = learning_rate;
    optimizer->alpha = alpha;
    optimizer->eps = eps;
    optimizer->weight_decay = weight_decay;
    optimizer->momentum = momentum;
    optimizer->centered = centered;
    optimizer->param_count = 0;
    optimizer->square_avg = NULL;
    optimizer->momentum_buffers = NULL;
    
    return optimizer;
}

void rmsprop_free(RMSprop* optimizer) {
    if (optimizer) {
        if (optimizer->square_avg) {
            for (size_t i = 0; i < optimizer->param_count; i++) {
                tensor_free(optimizer->square_avg[i]);
            }
            free(optimizer->square_avg);
        }
        if (optimizer->momentum_buffers) {
            for (size_t i = 0; i < optimizer->param_count; i++) {
                tensor_free(optimizer->momentum_buffers[i]);
            }
            free(optimizer->momentum_buffers);
        }
        free(optimizer);
    }
}

void rmsprop_step(RMSprop* optimizer, Tensor** params, Tensor** grads, size_t param_count) {
    if (optimizer->param_count != param_count) {
        rmsprop_free(optimizer);
        optimizer->param_count = param_count;
        optimizer->square_avg = (Tensor**)malloc(param_count * sizeof(Tensor*));
        optimizer->momentum_buffers = (Tensor**)malloc(param_count * sizeof(Tensor*));
        
        if (!optimizer->square_avg || !optimizer->momentum_buffers) {
            free(optimizer->square_avg);
            free(optimizer->momentum_buffers);
            return;
        }
        
        for (size_t i = 0; i < param_count; i++) {
            if (params[i]) {
                optimizer->square_avg[i] = tensor_zeros(params[i]->shape, params[i]->ndim, false);
                optimizer->momentum_buffers[i] = tensor_zeros(params[i]->shape, params[i]->ndim, false);
            } else {
                optimizer->square_avg[i] = NULL;
                optimizer->momentum_buffers[i] = NULL;
            }
        }
    }
    
    for (size_t i = 0; i < param_count; i++) {
        if (!params[i] || !grads[i]) continue;
        if (!params[i]->requires_grad) continue;
        if (!optimizer->square_avg[i]) continue;
        
        for (size_t j = 0; j < params[i]->size; j++) {
            float grad = grads[i]->data[j];
            
            if (optimizer->weight_decay != 0) {
                grad += optimizer->weight_decay * params[i]->data[j];
            }
            
            optimizer->square_avg[i]->data[j] = 
                optimizer->alpha * optimizer->square_avg[i]->data[j] + 
                (1.0f - optimizer->alpha) * grad * grad;
            
            float avg = optimizer->square_avg[i]->data[j];
            
            if (optimizer->momentum > 0) {
                optimizer->momentum_buffers[i]->data[j] = 
                    optimizer->momentum * optimizer->momentum_buffers[i]->data[j] + 
                    grad / (sqrt(avg) + optimizer->eps);
                
                params[i]->data[j] -= optimizer->learning_rate * optimizer->momentum_buffers[i]->data[j];
            } else {
                params[i]->data[j] -= optimizer->learning_rate * grad / (sqrt(avg) + optimizer->eps);
            }
        }
    }
}

Optimizer* optimizer_create_sgd(float learning_rate) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    if (!optimizer) return NULL;
    
    optimizer->optimizer = sgd_create(learning_rate);
    optimizer->type = OPTIM_SGD;
    optimizer->step = (void(*)(void*, Tensor**, Tensor**, size_t))sgd_step;
    optimizer->free = (void(*)(void*))sgd_free;
    
    if (!optimizer->optimizer) {
        free(optimizer);
        return NULL;
    }
    
    return optimizer;
}

Optimizer* optimizer_create_adam(float learning_rate, float beta1, float beta2, float eps) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    if (!optimizer) return NULL;
    
    optimizer->optimizer = adam_create(learning_rate, beta1, beta2, eps);
    optimizer->type = OPTIM_ADAM;
    optimizer->step = (void(*)(void*, Tensor**, Tensor**, size_t))adam_step;
    optimizer->free = (void(*)(void*))adam_free;
    
    if (!optimizer->optimizer) {
        free(optimizer);
        return NULL;
    }
    
    return optimizer;
}

Optimizer* optimizer_create_momentum_sgd(float learning_rate, float momentum, float dampening, float weight_decay) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    if (!optimizer) return NULL;
    
    optimizer->optimizer = momentum_sgd_create(learning_rate, momentum, dampening, weight_decay);
    optimizer->type = OPTIM_MOMENTUM_SGD;
    optimizer->step = (void(*)(void*, Tensor**, Tensor**, size_t))momentum_sgd_step;
    optimizer->free = (void(*)(void*))momentum_sgd_free;
    
    if (!optimizer->optimizer) {
        free(optimizer);
        return NULL;
    }
    
    return optimizer;
}

Optimizer* optimizer_create_rmsprop(float learning_rate, float alpha, float eps, float weight_decay, float momentum, float centered) {
    Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
    if (!optimizer) return NULL;
    
    optimizer->optimizer = rmsprop_create(learning_rate, alpha, eps, weight_decay, momentum, centered);
    optimizer->type = OPTIM_RMSPROP;
    optimizer->step = (void(*)(void*, Tensor**, Tensor**, size_t))rmsprop_step;
    optimizer->free = (void(*)(void*))rmsprop_free;
    
    if (!optimizer->optimizer) {
        free(optimizer);
        return NULL;
    }
    
    return optimizer;
}

void optimizer_free(Optimizer* optimizer) {
    if (optimizer) {
        optimizer->free(optimizer->optimizer);
        free(optimizer);
    }
}

void optimizer_step(Optimizer* optimizer, Tensor** params, Tensor** grads, size_t param_count) {
    optimizer->step(optimizer->optimizer, params, grads, param_count);
}