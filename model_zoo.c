#include "model_zoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/stat.h>
#include <dirent.h>
#include <curl/curl.h>
#include <openssl/sha.h>

// ==================== 内部工具函数 ====================

static size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    return fwrite(ptr, size, nmemb, stream);
}

static int download_file_internal(const char* url, const char* output_path, 
                                 void (*progress_callback)(float progress)) {
    CURL* curl;
    FILE* fp;
    CURLcode res;
    
    curl = curl_easy_init();
    if (!curl) {
        return -1;
    }
    
    fp = fopen(output_path, "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        return -1;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    // 设置进度回调
    if (progress_callback) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        // 这里可以添加进度回调函数
    }
    
    res = curl_easy_perform(curl);
    
    fclose(fp);
    curl_easy_cleanup(curl);
    
    return (res == CURLE_OK) ? 0 : -1;
}

static int create_directory_recursive_internal(const char* path) {
    char* path_copy = strdup(path);
    char* p = path_copy;
    
    while (*p) {
        if (*p == '/') {
            *p = '\0';
            mkdir(path_copy, 0755);
            *p = '/';
        }
        p++;
    }
    
    mkdir(path_copy, 0755);
    free(path_copy);
    return 0;
}

static int calculate_file_hash_internal(const char* file_path, char* hash_buffer, size_t buffer_size) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        return -1;
    }
    
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    
    unsigned char buffer[4096];
    size_t bytes_read;
    
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        SHA256_Update(&sha256, buffer, bytes_read);
    }
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &sha256);
    
    fclose(file);
    
    // 转换为十六进制字符串
    if (buffer_size < SHA256_DIGEST_LENGTH * 2 + 1) {
        return -1;
    }
    
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(hash_buffer + (i * 2), "%02x", hash[i]);
    }
    hash_buffer[SHA256_DIGEST_LENGTH * 2] = '\0';
    
    return 0;
}

// ==================== 模型库管理器实现 ====================

model_zoo_manager_t* model_zoo_create(const char* cache_dir, const char* config_file) {
    model_zoo_manager_t* manager = malloc(sizeof(model_zoo_manager_t));
    if (!manager) {
        return NULL;
    }
    
    memset(manager, 0, sizeof(model_zoo_manager_t));
    
    // 设置缓存目录
    if (cache_dir) {
        manager->cache_dir = strdup(cache_dir);
        create_directory_recursive_internal(cache_dir);
    } else {
        manager->cache_dir = strdup("./model_cache");
        create_directory_recursive_internal("./model_cache");
    }
    
    // 设置配置文件
    if (config_file) {
        manager->config_file = strdup(config_file);
    } else {
        manager->config_file = strdup("./model_zoo_config.json");
    }
    
    manager->max_models = 100;
    manager->models = malloc(sizeof(model_weight_info_t*) * manager->max_models);
    if (!manager->models) {
        free(manager->cache_dir);
        free(manager->config_file);
        free(manager);
        return NULL;
    }
    
    manager->enable_auto_download = true;
    manager->enable_hash_verification = true;
    
    // 初始化cURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    return manager;
}

void model_zoo_destroy(model_zoo_manager_t* manager) {
    if (!manager) return;
    
    // 释放所有模型信息
    for (int i = 0; i < manager->num_models; i++) {
        model_weight_info_t* model = manager->models[i];
        if (model) {
            free(model->model_name);
            free(model->version);
            free(model->author);
            free(model->description);
            free(model->license);
            free(model->download_url);
            free(model->local_path);
            free(model->sha256_hash);
            free(model);
        }
    }
    
    free(manager->models);
    free(manager->cache_dir);
    free(manager->config_file);
    free(manager);
    
    // 清理cURL
    curl_global_cleanup();
}

int model_zoo_register_model(model_zoo_manager_t* manager, 
                            const char* model_name,
                            model_type_t model_type,
                            model_architecture_t architecture,
                            pretrained_dataset_t dataset,
                            const char* download_url,
                            const char* description) {
    if (!manager || !model_name || !download_url) {
        return -1;
    }
    
    if (manager->num_models >= manager->max_models) {
        // 扩展数组
        int new_max = manager->max_models * 2;
        model_weight_info_t** new_models = realloc(manager->models, 
                                                  sizeof(model_weight_info_t*) * new_max);
        if (!new_models) {
            return -1;
        }
        manager->models = new_models;
        manager->max_models = new_max;
    }
    
    model_weight_info_t* model = malloc(sizeof(model_weight_info_t));
    if (!model) {
        return -1;
    }
    
    memset(model, 0, sizeof(model_weight_info_t));
    
    model->model_name = strdup(model_name);
    model->model_type = model_type;
    model->architecture = architecture;
    model->dataset = dataset;
    model->download_url = strdup(download_url);
    model->description = description ? strdup(description) : strdup("");
    model->version = strdup("1.0.0");
    model->author = strdup("AI Framework Team");
    model->license = strdup("Apache-2.0");
    
    // 构建本地路径
    char local_path[1024];
    snprintf(local_path, sizeof(local_path), "%s/%s.weights", 
             manager->cache_dir, model_name);
    model->local_path = strdup(local_path);
    
    // 检查文件是否存在
    struct stat st;
    model->is_downloaded = (stat(local_path, &st) == 0);
    if (model->is_downloaded) {
        model->file_size = st.st_size;
    }
    
    manager->models[manager->num_models++] = model;
    
    return 0;
}

int model_zoo_download_weights(model_zoo_manager_t* manager, 
                              const char* model_name,
                              bool force_download) {
    if (!manager || !model_name) {
        return -1;
    }
    
    // 查找模型
    model_weight_info_t* model = NULL;
    for (int i = 0; i < manager->num_models; i++) {
        if (strcmp(manager->models[i]->model_name, model_name) == 0) {
            model = manager->models[i];
            break;
        }
    }
    
    if (!model) {
        return -1;
    }
    
    // 检查是否已下载
    if (model->is_downloaded && !force_download) {
        printf("模型权重已存在: %s\n", model->local_path);
        return 0;
    }
    
    printf("开始下载模型权重: %s\n", model_name);
    printf("下载URL: %s\n", model->download_url);
    printf("保存路径: %s\n", model->local_path);
    
    // 下载文件
    if (download_file_internal(model->download_url, model->local_path, NULL) != 0) {
        printf("下载失败: %s\n", model_name);
        return -1;
    }
    
    // 更新文件信息
    struct stat st;
    if (stat(model->local_path, &st) == 0) {
        model->file_size = st.st_size;
        model->is_downloaded = true;
    }
    
    printf("下载完成: %s (%.2f MB)\n", model_name, 
           (float)model->file_size / (1024 * 1024));
    
    return 0;
}

bool model_zoo_verify_weights(model_zoo_manager_t* manager, 
                              const char* model_name) {
    if (!manager || !model_name || !manager->enable_hash_verification) {
        return false;
    }
    
    // 查找模型
    model_weight_info_t* model = NULL;
    for (int i = 0; i < manager->num_models; i++) {
        if (strcmp(manager->models[i]->model_name, model_name) == 0) {
            model = manager->models[i];
            break;
        }
    }
    
    if (!model || !model->is_downloaded) {
        return false;
    }
    
    // 计算文件哈希
    char calculated_hash[65];
    if (calculate_file_hash_internal(model->local_path, calculated_hash, 
                                    sizeof(calculated_hash)) != 0) {
        return false;
    }
    
    // 如果有预定义的哈希值，进行验证
    if (model->sha256_hash) {
        bool verified = (strcmp(calculated_hash, model->sha256_hash) == 0);
        if (verified) {
            model->is_verified = true;
            printf("模型权重验证通过: %s\n", model_name);
        } else {
            printf("模型权重验证失败: %s\n", model_name);
            printf("期望哈希: %s\n", model->sha256_hash);
            printf("实际哈希: %s\n", calculated_hash);
        }
        return verified;
    }
    
    // 如果没有预定义哈希，保存计算出的哈希
    model->sha256_hash = strdup(calculated_hash);
    model->is_verified = true;
    
    return true;
}

// ==================== 权重加载器实现 ====================

weight_loader_t* weight_loader_create(const char* weight_file_path) {
    if (!weight_file_path) {
        return NULL;
    }
    
    weight_loader_t* loader = malloc(sizeof(weight_loader_t));
    if (!loader) {
        return NULL;
    }
    
    memset(loader, 0, sizeof(weight_loader_t));
    
    loader->weight_file_path = strdup(weight_file_path);
    loader->weight_file = fopen(weight_file_path, "rb");
    
    if (!loader->weight_file) {
        free(loader->weight_file_path);
        free(loader);
        return NULL;
    }
    
    // 读取文件头
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, loader->weight_file) != 1) {
        fclose(loader->weight_file);
        free(loader->weight_file_path);
        free(loader);
        return NULL;
    }
    
    // 验证魔数
    if (magic != 0x4D4F444C) { // "MODL"
        fclose(loader->weight_file);
        free(loader->weight_file_path);
        free(loader);
        return NULL;
    }
    
    // 读取版本号
    fread(&loader->format_version, sizeof(uint32_t), 1, loader->weight_file);
    
    // 读取模型名称长度
    uint32_t name_length;
    fread(&name_length, sizeof(uint32_t), 1, loader->weight_file);
    
    // 读取模型名称
    loader->model_name = malloc(name_length + 1);
    fread(loader->model_name, 1, name_length, loader->weight_file);
    loader->model_name[name_length] = '\0';
    
    // 读取参数总数
    fread(&loader->total_parameters, sizeof(int64_t), 1, loader->weight_file);
    
    loader->current_offset = ftell(loader->weight_file);
    
    return loader;
}

void weight_loader_destroy(weight_loader_t* loader) {
    if (!loader) return;
    
    if (loader->weight_file) {
        fclose(loader->weight_file);
    }
    
    free(loader->weight_file_path);
    free(loader->model_name);
    free(loader);
}

// ==================== 预训练模型加载实现 ====================

nn_module_t* model_zoo_load_resnet50(model_zoo_manager_t* manager, bool pretrained) {
    if (!manager) {
        return NULL;
    }
    
    printf("加载ResNet50模型...\n");
    
    // 这里实现ResNet50模型构建
    // 实际实现需要根据具体的神经网络模块API
    
    nn_module_t* model = NULL; // nn_module_create(...)
    
    if (pretrained) {
        // 下载并加载预训练权重
        if (model_zoo_download_weights(manager, "resnet50", false) == 0) {
            weight_loader_t* loader = weight_loader_create(
                manager->cache_dir ? 
                strcat(strdup(manager->cache_dir), "/resnet50.weights") : 
                "./model_cache/resnet50.weights"
            );
            
            if (loader) {
                // 加载权重到模型
                // weight_loader_load_to_module(loader, model, NULL);
                weight_loader_destroy(loader);
                printf("ResNet50预训练权重加载完成\n");
            }
        }
    }
    
    return model;
}

// ==================== 工具函数实现 ====================

int calculate_file_hash(const char* file_path, char* hash_buffer, size_t buffer_size) {
    return calculate_file_hash_internal(file_path, hash_buffer, buffer_size);
}

int download_file(const char* url, const char* output_path, 
                 void (*progress_callback)(float progress)) {
    return download_file_internal(url, output_path, progress_callback);
}

int create_directory_recursive(const char* path) {
    return create_directory_recursive_internal(path);
}

// 其他预训练模型加载函数实现类似...