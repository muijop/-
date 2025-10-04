#include "dataloader.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

Dataset* dataset_create(size_t initial_capacity) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) return NULL;
    
    dataset->data = (Tensor**)malloc(initial_capacity * sizeof(Tensor*));
    dataset->targets = (Tensor**)malloc(initial_capacity * sizeof(Tensor*));
    
    if (!dataset->data || !dataset->targets) {
        free(dataset->data);
        free(dataset->targets);
        free(dataset);
        return NULL;
    }
    
    dataset->size = 0;
    dataset->capacity = initial_capacity;
    
    return dataset;
}

void dataset_free(Dataset* dataset) {
    if (dataset) {
        for (size_t i = 0; i < dataset->size; i++) {
            tensor_free(dataset->data[i]);
            tensor_free(dataset->targets[i]);
        }
        free(dataset->data);
        free(dataset->targets);
        free(dataset);
    }
}

void dataset_add_sample(Dataset* dataset, Tensor* data, Tensor* target) {
    if (dataset->size >= dataset->capacity) {
        size_t new_capacity = dataset->capacity * 2;
        Tensor** new_data = (Tensor**)realloc(dataset->data, new_capacity * sizeof(Tensor*));
        Tensor** new_targets = (Tensor**)realloc(dataset->targets, new_capacity * sizeof(Tensor*));
        
        if (!new_data || !new_targets) {
            return;
        }
        
        dataset->data = new_data;
        dataset->targets = new_targets;
        dataset->capacity = new_capacity;
    }
    
    dataset->data[dataset->size] = data;
    dataset->targets[dataset->size] = target;
    dataset->size++;
}

Dataset* dataset_create_random_classification(size_t num_samples, size_t num_features, size_t num_classes) {
    Dataset* dataset = dataset_create(num_samples);
    if (!dataset) return NULL;
    
    srand(time(NULL));
    
    for (size_t i = 0; i < num_samples; i++) {
        size_t data_shape[2] = {1, num_features};
        Tensor* data = tensor_create(NULL, data_shape, 2, false);
        
        size_t target_shape[1] = {1};
        Tensor* target = tensor_create(NULL, target_shape, 1, false);
        
        if (!data || !target) {
            tensor_free(data);
            tensor_free(target);
            dataset_free(dataset);
            return NULL;
        }
        
        for (size_t j = 0; j < num_features; j++) {
            data->data[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        target->data[0] = (float)(rand() % num_classes);
        
        dataset_add_sample(dataset, data, target);
    }
    
    return dataset;
}

Dataset* dataset_create_random_regression(size_t num_samples, size_t num_features) {
    Dataset* dataset = dataset_create(num_samples);
    if (!dataset) return NULL;
    
    srand(time(NULL));
    
    for (size_t i = 0; i < num_samples; i++) {
        size_t data_shape[2] = {1, num_features};
        Tensor* data = tensor_create(NULL, data_shape, 2, false);
        
        size_t target_shape[1] = {1};
        Tensor* target = tensor_create(NULL, target_shape, 1, false);
        
        if (!data || !target) {
            tensor_free(data);
            tensor_free(target);
            dataset_free(dataset);
            return NULL;
        }
        
        for (size_t j = 0; j < num_features; j++) {
            data->data[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        target->data[0] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        
        dataset_add_sample(dataset, data, target);
    }
    
    return dataset;
}

static void shuffle_array(size_t* array, size_t size) {
    for (size_t i = size - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

DataLoader* dataloader_create(Dataset* dataset, size_t batch_size, bool shuffle, bool drop_last, size_t num_workers) {
    DataLoader* loader = (DataLoader*)malloc(sizeof(DataLoader));
    if (!loader) return NULL;
    
    loader->dataset = dataset;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->drop_last = drop_last;
    loader->num_workers = num_workers;
    loader->current_index = 0;
    
    loader->indices = (size_t*)malloc(dataset->size * sizeof(size_t));
    if (!loader->indices) {
        free(loader);
        return NULL;
    }
    
    for (size_t i = 0; i < dataset->size; i++) {
        loader->indices[i] = i;
    }
    
    if (shuffle) {
        shuffle_array(loader->indices, dataset->size);
    }
    
    return loader;
}

void dataloader_free(DataLoader* loader) {
    if (loader) {
        free(loader->indices);
        free(loader);
    }
}

bool dataloader_next_batch(DataLoader* loader, Tensor** batch_data, Tensor** batch_targets) {
    if (!loader || !batch_data || !batch_targets) return false;
    
    size_t remaining_samples = loader->dataset->size - loader->current_index;
    
    if (remaining_samples < loader->batch_size) {
        if (loader->drop_last) {
            return false;
        } else if (remaining_samples == 0) {
            return false;
        }
    }
    
    size_t current_batch_size = (remaining_samples >= loader->batch_size) ? loader->batch_size : remaining_samples;
    
    if (loader->dataset->size == 0 || loader->dataset->data[0] == NULL) {
        return false;
    }
    
    size_t data_size = loader->dataset->data[0]->size;
    size_t target_size = loader->dataset->targets[0]->size;
    size_t data_ndim = loader->dataset->data[0]->ndim;
    size_t target_ndim = loader->dataset->targets[0]->ndim;
    
    size_t* batch_data_shape = (size_t*)malloc(data_ndim * sizeof(size_t));
    size_t* batch_target_shape = (size_t*)malloc(target_ndim * sizeof(size_t));
    
    if (!batch_data_shape || !batch_target_shape) {
        free(batch_data_shape);
        free(batch_target_shape);
        return false;
    }
    
    batch_data_shape[0] = current_batch_size;
    batch_target_shape[0] = current_batch_size;
    
    for (size_t i = 1; i < data_ndim; i++) {
        batch_data_shape[i] = loader->dataset->data[0]->shape[i];
    }
    for (size_t i = 1; i < target_ndim; i++) {
        batch_target_shape[i] = loader->dataset->targets[0]->shape[i];
    }
    
    *batch_data = tensor_create(NULL, batch_data_shape, data_ndim, false);
    *batch_targets = tensor_create(NULL, batch_target_shape, target_ndim, false);
    
    if (!*batch_data || !*batch_targets) {
        tensor_free(*batch_data);
        tensor_free(*batch_targets);
        free(batch_data_shape);
        free(batch_target_shape);
        return false;
    }
    
    for (size_t i = 0; i < current_batch_size; i++) {
        size_t sample_idx = loader->indices[loader->current_index + i];
        
        memcpy((*batch_data)->data + i * data_size, 
               loader->dataset->data[sample_idx]->data, 
               data_size * sizeof(float));
        
        memcpy((*batch_targets)->data + i * target_size, 
               loader->dataset->targets[sample_idx]->data, 
               target_size * sizeof(float));
    }
    
    loader->current_index += current_batch_size;
    
    free(batch_data_shape);
    free(batch_target_shape);
    
    return true;
}

void dataloader_reset(DataLoader* loader) {
    if (!loader) return;
    
    loader->current_index = 0;
    
    if (loader->shuffle) {
        shuffle_array(loader->indices, loader->dataset->size);
    }
}

DataModule* datamodule_create(size_t num_samples, size_t num_features, size_t num_classes, 
                             float train_split, float val_split, size_t batch_size) {
    DataModule* dm = (DataModule*)malloc(sizeof(DataModule));
    if (!dm) return NULL;
    
    size_t train_size = (size_t)(num_samples * train_split);
    size_t val_size = (size_t)(num_samples * val_split);
    size_t test_size = num_samples - train_size - val_size;
    
    dm->train_dataset = dataset_create_random_classification(train_size, num_features, num_classes);
    dm->val_dataset = dataset_create_random_classification(val_size, num_features, num_classes);
    dm->test_dataset = dataset_create_random_classification(test_size, num_features, num_classes);
    
    if (!dm->train_dataset || !dm->val_dataset || !dm->test_dataset) {
        dataset_free(dm->train_dataset);
        dataset_free(dm->val_dataset);
        dataset_free(dm->test_dataset);
        free(dm);
        return NULL;
    }
    
    dm->train_loader = dataloader_create(dm->train_dataset, batch_size, true, false, 0);
    dm->val_loader = dataloader_create(dm->val_dataset, batch_size, false, false, 0);
    dm->test_loader = dataloader_create(dm->test_dataset, batch_size, false, false, 0);
    
    if (!dm->train_loader || !dm->val_loader || !dm->test_loader) {
        dataloader_free(dm->train_loader);
        dataloader_free(dm->val_loader);
        dataloader_free(dm->test_loader);
        dataset_free(dm->train_dataset);
        dataset_free(dm->val_dataset);
        dataset_free(dm->test_dataset);
        free(dm);
        return NULL;
    }
    
    return dm;
}

void datamodule_free(DataModule* dm) {
    if (dm) {
        dataloader_free(dm->train_loader);
        dataloader_free(dm->val_loader);
        dataloader_free(dm->test_loader);
        dataset_free(dm->train_dataset);
        dataset_free(dm->val_dataset);
        dataset_free(dm->test_dataset);
        free(dm);
    }
}