#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include <stdbool.h>

typedef struct {
    Tensor** data;
    Tensor** targets;
    size_t size;
    size_t capacity;
} Dataset;

typedef struct {
    Dataset* dataset;
    size_t batch_size;
    bool shuffle;
    bool drop_last;
    size_t* indices;
    size_t current_index;
    size_t num_workers;
} DataLoader;

typedef struct {
    Dataset* train_dataset;
    Dataset* val_dataset;
    Dataset* test_dataset;
    DataLoader* train_loader;
    DataLoader* val_loader;
    DataLoader* test_loader;
} DataModule;

Dataset* dataset_create(size_t initial_capacity);
void dataset_free(Dataset* dataset);
void dataset_add_sample(Dataset* dataset, Tensor* data, Tensor* target);
Dataset* dataset_create_random_classification(size_t num_samples, size_t num_features, size_t num_classes);
Dataset* dataset_create_random_regression(size_t num_samples, size_t num_features);

DataLoader* dataloader_create(Dataset* dataset, size_t batch_size, bool shuffle, bool drop_last, size_t num_workers);
void dataloader_free(DataLoader* loader);
bool dataloader_next_batch(DataLoader* loader, Tensor** batch_data, Tensor** batch_targets);
void dataloader_reset(DataLoader* loader);

DataModule* datamodule_create(size_t num_samples, size_t num_features, size_t num_classes, 
                             float train_split, float val_split, size_t batch_size);
void datamodule_free(DataModule* dm);

#endif