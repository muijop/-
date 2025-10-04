// Optimizers for AI training framework
// Provides common optimization algorithms similar to PyTorch

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "nn.h"

// Stochastic Gradient Descent optimizer
typedef struct {
    float learning_rate;         // Learning rate
} SGD;

// Adam optimizer (Adaptive Moment Estimation)
typedef struct {
    float learning_rate;         // Learning rate
    float beta1;                 // First moment decay rate
    float beta2;                 // Second moment decay rate
    float eps;                   // Numerical stability term
    size_t t;                    // Timestep counter
    Tensor** m;                   // First moment estimates
    Tensor** v;                  // Second moment estimates
    size_t param_count;          // Number of parameters
} Adam;

// Momentum SGD optimizer
typedef struct {
    float learning_rate;         // Learning rate
    float momentum;              // Momentum factor
    float dampening;             // Dampening for momentum
    float weight_decay;          // Weight decay (L2 regularization)
    Tensor** momentum_buffers;   // Momentum buffers
    size_t param_count;          // Number of parameters
} MomentumSGD;

// RMSprop optimizer
typedef struct {
    float learning_rate;         // Learning rate
    float alpha;                 // Smoothing constant
    float eps;                   // Numerical stability term
    float weight_decay;          // Weight decay
    float momentum;              // Momentum factor
    float centered;              // Whether to use centered RMSprop
    Tensor** square_avg;         // Running average of squared gradients
    Tensor** momentum_buffers;   // Momentum buffers
    size_t param_count;          // Number of parameters
} RMSprop;

// SGD optimizer functions
SGD* sgd_create(float learning_rate);
void sgd_free(SGD* optimizer);
void sgd_step(SGD* optimizer, Tensor** params, Tensor** grads, size_t param_count);

// Adam optimizer functions
Adam* adam_create(float learning_rate, float beta1, float beta2, float eps);
void adam_free(Adam* optimizer);
void adam_step(Adam* optimizer, Tensor** params, Tensor** grads, size_t param_count);

// Momentum SGD optimizer functions
MomentumSGD* momentum_sgd_create(float learning_rate, float momentum, float dampening, float weight_decay);
void momentum_sgd_free(MomentumSGD* optimizer);
void momentum_sgd_step(MomentumSGD* optimizer, Tensor** params, Tensor** grads, size_t param_count);

// RMSprop optimizer functions
RMSprop* rmsprop_create(float learning_rate, float alpha, float eps, float weight_decay, float momentum, float centered);
void rmsprop_free(RMSprop* optimizer);
void rmsprop_step(RMSprop* optimizer, Tensor** params, Tensor** grads, size_t param_count);

// Generic optimizer interface
typedef struct {
    void* optimizer;             // Specific optimizer implementation
    void (*step)(void*, Tensor**, Tensor**, size_t); // Step function
    void (*free)(void*);         // Free function
    enum { 
        OPTIM_SGD,               // SGD optimizer
        OPTIM_ADAM,              // Adam optimizer
        OPTIM_MOMENTUM_SGD,      // Momentum SGD optimizer
        OPTIM_RMSPROP            // RMSprop optimizer
    } type;                      // Optimizer type
} Optimizer;

// Generic optimizer creation functions
Optimizer* optimizer_create_sgd(float learning_rate);
Optimizer* optimizer_create_adam(float learning_rate, float beta1, float beta2, float eps);
Optimizer* optimizer_create_momentum_sgd(float learning_rate, float momentum, float dampening, float weight_decay);
Optimizer* optimizer_create_rmsprop(float learning_rate, float alpha, float eps, float weight_decay, float momentum, float centered);

// Generic optimizer management functions
void optimizer_free(Optimizer* optimizer);
void optimizer_step(Optimizer* optimizer, Tensor** params, Tensor** grads, size_t param_count);

#endif