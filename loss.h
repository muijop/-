// Loss functions for AI training framework
// Provides common loss functions for neural network training

#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

// Mean Squared Error loss for regression tasks
typedef struct {
    float reduction;              // Reduction method (mean, sum, none)
} MSELoss;

// Cross Entropy loss for classification tasks
typedef struct {
    float reduction;              // Reduction method (mean, sum, none)
} CrossEntropyLoss;

// Binary Cross Entropy loss for binary classification
typedef struct {
    float reduction;              // Reduction method (mean, sum, none)
} BCELoss;

// Margin Ranking loss for ranking tasks
typedef struct {
    float margin;                 // Margin parameter
    float reduction;              // Reduction method (mean, sum, none)
} MarginRankingLoss;

// L1 loss (Mean Absolute Error) for robust regression
typedef struct {
    float reduction;              // Reduction method (mean, sum, none)
} L1Loss;

// Smooth L1 loss (Huber loss) for robust regression
typedef struct {
    float beta;                   // Beta parameter for smooth transition
    float reduction;              // Reduction method (mean, sum, none)
} SmoothL1Loss;

// MSE loss functions
MSELoss* mse_loss_create(float reduction);
void mse_loss_free(MSELoss* loss);
Tensor* mse_loss_forward(MSELoss* loss, Tensor* input, Tensor* target);

// Cross entropy loss functions
CrossEntropyLoss* cross_entropy_loss_create(float reduction);
void cross_entropy_loss_free(CrossEntropyLoss* loss);
Tensor* cross_entropy_loss_forward(CrossEntropyLoss* loss, Tensor* input, Tensor* target);

// Binary cross entropy loss functions
BCELoss* bce_loss_create(float reduction);
void bce_loss_free(BCELoss* loss);
Tensor* bce_loss_forward(BCELoss* loss, Tensor* input, Tensor* target);

// Margin ranking loss functions
MarginRankingLoss* margin_ranking_loss_create(float margin, float reduction);
void margin_ranking_loss_free(MarginRankingLoss* loss);
Tensor* margin_ranking_loss_forward(MarginRankingLoss* loss, Tensor* input1, Tensor* input2, Tensor* target);

// L1 loss functions
L1Loss* l1_loss_create(float reduction);
void l1_loss_free(L1Loss* loss);
Tensor* l1_loss_forward(L1Loss* loss, Tensor* input, Tensor* target);

// Smooth L1 loss functions
SmoothL1Loss* smooth_l1_loss_create(float beta, float reduction);
void smooth_l1_loss_free(SmoothL1Loss* loss);
Tensor* smooth_l1_loss_forward(SmoothL1Loss* loss, Tensor* input, Tensor* target);

#endif