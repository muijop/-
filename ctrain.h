#ifndef CTRAIN_H
#define CTRAIN_H

#include "tensor.h"
#include "nn.h"
#include "optimizer.h"
#include "loss.h"
#include "dataloader.h"

typedef struct {
    void* model;
    Tensor* (*forward)(void*, Tensor*);
    void (*free)(void*);
} ModelWrapper;

typedef struct {
    ModelWrapper* model;
    Optimizer* optimizer;
    void* loss_fn;
    Tensor* (*loss_forward)(void*, Tensor*, Tensor*);
    void (*loss_free)(void*);
} TrainingConfig;

typedef struct {
    float train_loss;
    float val_loss;
    float train_accuracy;
    float val_accuracy;
    size_t epoch;
    size_t step;
} TrainingMetrics;

typedef struct {
    TrainingMetrics* history;
    size_t history_size;
    size_t history_capacity;
} TrainingHistory;

TrainingConfig* training_config_create(ModelWrapper* model, Optimizer* optimizer, void* loss_fn,
                                     Tensor* (*loss_forward)(void*, Tensor*, Tensor*),
                                     void (*loss_free)(void*));
void training_config_free(TrainingConfig* config);

TrainingHistory* training_history_create(size_t initial_capacity);
void training_history_free(TrainingHistory* history);
void training_history_add(TrainingHistory* history, TrainingMetrics metrics);

void train_epoch(TrainingConfig* config, DataLoader* train_loader, TrainingHistory* history);
void validate_epoch(TrainingConfig* config, DataLoader* val_loader, TrainingHistory* history);
void fit(TrainingConfig* config, DataModule* datamodule, size_t num_epochs);

float calculate_accuracy(Tensor* predictions, Tensor* targets);
Tensor* predict_batch(TrainingConfig* config, Tensor* batch_data);

ModelWrapper* model_wrapper_create(void* model, Tensor* (*forward)(void*, Tensor*), void (*free)(void*));
void model_wrapper_free(ModelWrapper* wrapper);

#endif