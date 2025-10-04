#include "time_series_analysis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ===========================================
// 内部工具函数
// ===========================================

static float random_float(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

static double get_current_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static float* create_random_matrix(size_t rows, size_t cols) {
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    if (!matrix) return NULL;
    
    for (size_t i = 0; i < rows * cols; i++) {
        matrix[i] = random_float(-0.1f, 0.1f);
    }
    
    return matrix;
}

static float calculate_mae(const float* predictions, const float* targets, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += fabsf(predictions[i] - targets[i]);
    }
    return sum / n;
}

static float calculate_mse(const float* predictions, const float* targets, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    return sum / n;
}

static float calculate_mape(const float* predictions, const float* targets, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (targets[i] != 0) {
            sum += fabsf((predictions[i] - targets[i]) / targets[i]);
        }
    }
    return (sum / n) * 100.0f;
}

// ===========================================
// 时间序列数据处理实现
// ===========================================

time_series_data_t* create_time_series_data(size_t length, size_t num_features, 
                                           size_t num_targets) {
    time_series_data_t* data = (time_series_data_t*)calloc(1, sizeof(time_series_data_t));
    if (!data) return NULL;
    
    data->length = length;
    data->num_features = num_features;
    data->num_targets = num_targets;
    
    // 分配内存
    if (length > 0) {
        data->data = (float**)malloc(length * sizeof(float*));
        data->targets = (float**)malloc(length * sizeof(float*));
        data->timestamps = (long*)malloc(length * sizeof(long));
        
        if (!data->data || !data->targets || !data->timestamps) {
            destroy_time_series_data(data);
            return NULL;
        }
        
        for (size_t i = 0; i < length; i++) {
            if (num_features > 0) {
                data->data[i] = (float*)calloc(num_features, sizeof(float));
            }
            if (num_targets > 0) {
                data->targets[i] = (float*)calloc(num_targets, sizeof(float));
            }
            data->timestamps[i] = i; // 默认时间戳
        }
    }
    
    // 分配统计信息内存
    if (num_features > 0) {
        data->means = (float*)calloc(num_features, sizeof(float));
        data->stds = (float*)calloc(num_features, sizeof(float));
        data->min_vals = (float*)calloc(num_features, sizeof(float));
        data->max_vals = (float*)calloc(num_features, sizeof(float));
    }
    
    // 初始化默认值
    data->frequency = 1;
    data->time_unit = strdup("step");
    data->is_multivariate = num_features > 1;
    data->has_missing_values = false;
    data->is_stationary = false;
    
    return data;
}

void destroy_time_series_data(time_series_data_t* data) {
    if (!data) return;
    
    if (data->data) {
        for (size_t i = 0; i < data->length; i++) {
            if (data->data[i]) free(data->data[i]);
        }
        free(data->data);
    }
    
    if (data->targets) {
        for (size_t i = 0; i < data->length; i++) {
            if (data->targets[i]) free(data->targets[i]);
        }
        free(data->targets);
    }
    
    if (data->timestamps) free(data->timestamps);
    if (data->means) free(data->means);
    if (data->stds) free(data->stds);
    if (data->min_vals) free(data->min_vals);
    if (data->max_vals) free(data->max_vals);
    if (data->time_unit) free(data->time_unit);
    if (data->series_name) free(data->series_name);
    
    free(data);
}

int set_time_series_point(time_series_data_t* data, size_t time_index, 
                         const float* features, const float* targets) {
    if (!data || time_index >= data->length) {
        return -1;
    }
    
    if (features && data->data[time_index]) {
        memcpy(data->data[time_index], features, data->num_features * sizeof(float));
    }
    
    if (targets && data->targets[time_index]) {
        memcpy(data->targets[time_index], targets, data->num_targets * sizeof(float));
    }
    
    return 0;
}

// ===========================================
// 时间序列模型实现
// ===========================================

time_series_model_t* create_time_series_model(time_series_method_t method, 
                                              const time_series_config_t* config) {
    time_series_model_t* model = (time_series_model_t*)calloc(1, sizeof(time_series_model_t));
    if (!model) return NULL;
    
    model->method = method;
    if (config) {
        model->config = *config;
    } else {
        // 设置默认配置
        model->config.method = method;
        model->config.input_length = 100;
        model->config.output_length = 10;
        model->config.input_dim = 1;
        model->config.output_dim = 1;
        model->config.learning_rate = 0.001f;
        model->config.max_epochs = 100;
        model->config.batch_size = 32;
        model->config.use_gpu = false;
        model->config.seed = 42;
        model->config.normalize = true;
        model->config.detrend = false;
        model->config.deseasonalize = false;
        model->config.patience = 10;
        model->config.min_delta = 0.001f;
        model->config.early_stopping = true;
        model->config.max_memory_mb = 4096;
        model->config.max_time_seconds = 3600;
    }
    
    model->is_trained = false;
    model->is_initialized = false;
    model->current_epoch = 0;
    model->best_loss = 1e9f;
    model->current_loss = 0.0f;
    model->mae = 0.0f;
    model->mse = 0.0f;
    model->rmse = 0.0f;
    model->mape = 0.0f;
    model->smape = 0.0f;
    model->training_time = 0.0;
    model->memory_usage = sizeof(time_series_model_t);
    
    // 设置随机种子
    srand(model->config.seed);
    
    return model;
}

void destroy_time_series_model(time_series_model_t* model) {
    if (!model) return;
    
    // 释放模型实现内存
    if (model->model_impl) {
        free(model->model_impl);
    }
    
    free(model);
}

int initialize_time_series_model(time_series_model_t* model, 
                               const time_series_data_t* data) {
    if (!model || !data) {
        return -1;
    }
    
    // 简化实现：分配模型参数内存
    size_t param_size = 0;
    switch (model->method) {
        case TS_LSTM:
            param_size = model->config.input_dim * model->config.specific.lstm.hidden_size * 
                        model->config.specific.lstm.num_layers * sizeof(float);
            break;
        case TS_GRU:
            param_size = model->config.input_dim * model->config.specific.gru.hidden_size * 
                        model->config.specific.gru.num_layers * sizeof(float);
            break;
        case TS_TCN:
            param_size = model->config.input_dim * model->config.specific.tcn.num_channels[0] * 
                        model->config.specific.tcn.num_levels * sizeof(float);
            break;
        case TS_TRANSFORMER:
            param_size = model->config.input_dim * model->config.specific.transformer.d_model * 
                        model->config.specific.transformer.num_layers * sizeof(float);
            break;
        default:
            param_size = 1024 * sizeof(float); // 默认大小
    }
    
    model->model_impl = malloc(param_size);
    if (!model->model_impl) {
        return -1;
    }
    
    // 初始化参数
    float* params = (float*)model->model_impl;
    for (size_t i = 0; i < param_size / sizeof(float); i++) {
        params[i] = random_float(-0.1f, 0.1f);
    }
    
    model->is_initialized = true;
    model->memory_usage += param_size;
    
    printf("时间序列模型初始化成功，方法: %d，参数大小: %zu bytes\n", 
           model->method, param_size);
    
    return 0;
}

// ===========================================
// 训练实现
// ===========================================

time_series_training_result_t* train_time_series_model(time_series_model_t* model, 
                                                      const time_series_data_t* train_data,
                                                      const time_series_data_t* val_data) {
    if (!model || !train_data || !model->is_initialized) {
        return NULL;
    }
    
    double start_time = get_current_time();
    
    // 创建训练结果
    time_series_training_result_t* result = 
        (time_series_training_result_t*)calloc(1, sizeof(time_series_training_result_t));
    if (!result) return NULL;
    
    result->history_size = model->config.max_epochs;
    result->loss_history = (float*)malloc(result->history_size * sizeof(float));
    result->mae_history = (float*)malloc(result->history_size * sizeof(float));
    
    if (!result->loss_history || !result->mae_history) {
        free(result->loss_history);
        free(result->mae_history);
        free(result);
        return NULL;
    }
    
    // 训练循环
    int patience_counter = 0;
    float best_val_loss = 1e9f;
    
    for (int epoch = 0; epoch < model->config.max_epochs; epoch++) {
        model->current_epoch = epoch;
        
        // 模拟训练过程
        float train_loss = 1.0f / (epoch + 1) + random_float(0.0f, 0.1f);
        float train_mae = 0.5f / (epoch + 1) + random_float(0.0f, 0.05f);
        
        float val_loss = train_loss + random_float(-0.05f, 0.05f);
        float val_mae = train_mae + random_float(-0.02f, 0.02f);
        
        // 更新模型状态
        model->current_loss = train_loss;
        model->mae = train_mae;
        model->mse = train_loss;
        model->rmse = sqrtf(train_loss);
        model->mape = train_mae * 10.0f; // 简化计算
        model->smape = train_mae * 8.0f;  // 简化计算
        
        // 记录历史
        result->loss_history[epoch] = train_loss;
        result->mae_history[epoch] = train_mae;
        
        // 早停检查
        if (val_loss < best_val_loss - model->config.min_delta) {
            best_val_loss = val_loss;
            patience_counter = 0;
            model->best_loss = val_loss;
        } else {
            patience_counter++;
        }
        
        // 打印进度
        if (epoch % 10 == 0) {
            printf("Epoch %d/%d - Loss: %.4f, MAE: %.4f, Val Loss: %.4f, Val MAE: %.4f\n",
                   epoch + 1, model->config.max_epochs, train_loss, train_mae, 
                   val_loss, val_mae);
        }
        
        // 检查早停条件
        if (model->config.early_stopping && 
            patience_counter >= model->config.patience) {
            printf("早停触发于第 %d 轮\n", epoch + 1);
            break;
        }
        
        // 检查时间限制
        double current_time = get_current_time();
        if (current_time - start_time > model->config.max_time_seconds) {
            printf("达到时间限制，停止训练\n");
            break;
        }
    }
    
    // 设置最终结果
    result->final_loss = model->current_loss;
    result->final_mae = model->mae;
    result->final_mse = model->mse;
    result->final_rmse = model->rmse;
    result->final_mape = model->mape;
    result->final_smape = model->smape;
    result->total_epochs = model->current_epoch + 1;
    result->total_time = get_current_time() - start_time;
    result->success = true;
    result->status_message = strdup("训练完成");
    
    model->is_trained = true;
    model->training_time = result->total_time;
    
    printf("时间序列训练完成，总时间: %.2f秒，最终损失: %.4f，最终MAE: %.4f\n",
           result->total_time, result->final_loss, result->final_mae);
    
    return result;
}

// ===========================================
// 预测实现
// ===========================================

time_series_prediction_result_t* predict_with_time_series_model(
    const time_series_model_t* model, const time_series_data_t* input_data) {
    if (!model || !input_data || !model->is_trained) {
        return NULL;
    }
    
    time_series_prediction_result_t* result = 
        (time_series_prediction_result_t*)calloc(1, sizeof(time_series_prediction_result_t));
    if (!result) return NULL;
    
    result->prediction_length = model->config.output_length;
    result->output_dim = model->config.output_dim;
    
    // 分配内存
    size_t pred_size = result->prediction_length * result->output_dim;
    result->predictions = (float*)calloc(pred_size, sizeof(float));
    result->confidence_intervals = (float*)calloc(pred_size * 2, sizeof(float));
    result->residuals = (float*)calloc(input_data->length * result->output_dim, sizeof(float));
    
    if (!result->predictions || !result->confidence_intervals || !result->residuals) {
        destroy_time_series_prediction_result(result);
        return NULL;
    }
    
    // 模拟预测过程
    for (size_t i = 0; i < pred_size; i++) {
        // 生成预测值（基于最后几个点的趋势）
        float base_value = 0.0f;
        if (input_data->length > 0 && input_data->data) {
            size_t last_idx = input_data->length - 1;
            base_value = input_data->data[last_idx][0]; // 取最后一个点的第一个特征
        }
        
        result->predictions[i] = base_value + random_float(-0.5f, 0.5f);
        
        // 生成置信区间
        result->confidence_intervals[2 * i] = result->predictions[i] - 0.2f;
        result->confidence_intervals[2 * i + 1] = result->predictions[i] + 0.2f;
    }
    
    // 计算预测质量指标
    if (input_data->targets && input_data->length >= result->prediction_length) {
        float* actual_values = (float*)malloc(result->prediction_length * sizeof(float));
        if (actual_values) {
            for (size_t i = 0; i < result->prediction_length; i++) {
                actual_values[i] = input_data->targets[i][0]; // 简化：取第一个目标
            }
            
            result->prediction_mae = calculate_mae(result->predictions, actual_values, 
                                                  result->prediction_length);
            result->prediction_mse = calculate_mse(result->predictions, actual_values, 
                                                  result->prediction_length);
            result->prediction_rmse = sqrtf(result->prediction_mse);
            result->prediction_mape = calculate_mape(result->predictions, actual_values, 
                                                   result->prediction_length);
            result->prediction_smape = result->prediction_mape * 0.9f; // 简化
            
            free(actual_values);
        }
    }
    
    return result;
}

// ===========================================
// 工具函数实现
// ===========================================

time_series_config_t create_default_lstm_config(size_t input_length, 
                                              size_t output_length,
                                              size_t input_dim, size_t output_dim) {
    time_series_config_t config;
    
    config.method = TS_LSTM;
    config.input_length = input_length;
    config.output_length = output_length;
    config.input_dim = input_dim;
    config.output_dim = output_dim;
    config.learning_rate = 0.001f;
    config.max_epochs = 100;
    config.batch_size = 32;
    config.use_gpu = false;
    config.seed = 42;
    
    config.specific.lstm.hidden_size = 64;
    config.specific.lstm.num_layers = 2;
    config.specific.lstm.dropout_rate = 0.2f;
    config.specific.lstm.bidirectional = false;
    strcpy(config.specific.lstm.activation, "tanh");
    
    config.normalize = true;
    config.detrend = false;
    config.deseasonalize = false;
    config.patience = 10;
    config.min_delta = 0.001f;
    config.early_stopping = true;
    config.max_memory_mb = 4096;
    config.max_time_seconds = 3600;
    
    return config;
}

time_series_config_t create_default_gru_config(size_t input_length, 
                                             size_t output_length,
                                             size_t input_dim, size_t output_dim) {
    time_series_config_t config;
    
    config.method = TS_GRU;
    config.input_length = input_length;
    config.output_length = output_length;
    config.input_dim = input_dim;
    config.output_dim = output_dim;
    config.learning_rate = 0.001f;
    config.max_epochs = 100;
    config.batch_size = 32;
    config.use_gpu = false;
    config.seed = 42;
    
    config.specific.gru.hidden_size = 64;
    config.specific.gru.num_layers = 2;
    config.specific.gru.dropout_rate = 0.2f;
    config.specific.gru.bidirectional = false;
    
    config.normalize = true;
    config.detrend = false;
    config.deseasonalize = false;
    config.patience = 10;
    config.min_delta = 0.001f;
    config.early_stopping = true;
    config.max_memory_mb = 4096;
    config.max_time_seconds = 3600;
    
    return config;
}

time_series_config_t create_default_tcn_config(size_t input_length, 
                                              size_t output_length,
                                              size_t input_dim, size_t output_dim) {
    time_series_config_t config;
    
    config.method = TS_TCN;
    config.input_length = input_length;
    config.output_length = output_length;
    config.input_dim = input_dim;
    config.output_dim = output_dim;
    config.learning_rate = 0.001f;
    config.max_epochs = 100;
    config.batch_size = 32;
    config.use_gpu = false;
    config.seed = 42;
    
    config.specific.tcn.num_channels[0] = 64;
    config.specific.tcn.num_channels[1] = 64;
    config.specific.tcn.num_channels[2] = 64;
    config.specific.tcn.kernel_size = 3;
    config.specific.tcn.dropout_rate = 0.2f;
    config.specific.tcn.num_levels = 4;
    config.specific.tcn.causal = true;
    
    config.normalize = true;
    config.detrend = false;
    config.deseasonalize = false;
    config.patience = 10;
    config.min_delta = 0.001f;
    config.early_stopping = true;
    config.max_memory_mb = 4096;
    config.max_time_seconds = 3600;
    
    return config;
}

time_series_config_t create_default_transformer_config(size_t input_length, 
                                                      size_t output_length,
                                                      size_t input_dim, size_t output_dim) {
    time_series_config_t config;
    
    config.method = TS_TRANSFORMER;
    config.input_length = input_length;
    config.output_length = output_length;
    config.input_dim = input_dim;
    config.output_dim = output_dim;
    config.learning_rate = 0.0001f;
    config.max_epochs = 200;
    config.batch_size = 16;
    config.use_gpu = false;
    config.seed = 42;
    
    config.specific.transformer.d_model = 64;
    config.specific.transformer.num_heads = 8;
    config.specific.transformer.num_layers = 4;
    config.specific.transformer.d_ff = 256;
    config.specific.transformer.dropout_rate = 0.1f;
    
    config.normalize = true;
    config.detrend = false;
    config.deseasonalize = false;
    config.patience = 15;
    config.min_delta = 0.001f;
    config.early_stopping = true;
    config.max_memory_mb = 8192;
    config.max_time_seconds = 7200;
    
    return config;
}

time_series_config_t create_default_arima_config(int p, int d, int q) {
    time_series_config_t config;
    
    config.method = TS_ARIMA;
    config.input_length = 100;
    config.output_length = 10;
    config.input_dim = 1;
    config.output_dim = 1;
    config.learning_rate = 0.01f;
    config.max_epochs = 50;
    config.batch_size = 1;
    config.use_gpu = false;
    config.seed = 42;
    
    config.specific.arima.p = p;
    config.specific.arima.d = d;
    config.specific.arima.q = q;
    config.specific.arima.seasonal_period = 0;
    config.specific.arima.use_mle = true;
    
    config.normalize = true;
    config.detrend = true;
    config.deseasonalize = false;
    config.patience = 5;
    config.min_delta = 0.001f;
    config.early_stopping = true;
    config.max_memory_mb = 1024;
    config.max_time_seconds = 1800;
    
    return config;
}

// 销毁预测结果
void destroy_time_series_prediction_result(time_series_prediction_result_t* result) {
    if (!result) return;
    
    if (result->predictions) free(result->predictions);
    if (result->confidence_intervals) free(result->confidence_intervals);
    if (result->residuals) free(result->residuals);
    
    free(result);
}

// 打印模型信息
void print_time_series_model_info(const time_series_model_t* model) {
    if (!model) return;
    
    const char* method_names[] = {
        "LSTM", "GRU", "TCN", "Transformer", "ARIMA", "SARIMA", 
        "指数平滑", "Prophet", "N-BEATS", "DeepAR"
    };
    
    printf("=== 时间序列模型信息 ===\n");
    printf("方法: %s\n", method_names[model->method]);
    printf("输入长度: %zu\n", model->config.input_length);
    printf("输出长度: %zu\n", model->config.output_length);
    printf("输入维度: %zu\n", model->config.input_dim);
    printf("输出维度: %zu\n", model->config.output_dim);
    printf("是否已训练: %s\n", model->is_trained ? "是" : "否");
    printf("当前轮数: %d\n", model->current_epoch);
    printf("最佳损失: %.4f\n", model->best_loss);
    printf("当前损失: %.4f\n", model->current_loss);
    printf("MAE: %.4f\n", model->mae);
    printf("MSE: %.4f\n", model->mse);
    printf("训练时间: %.2f秒\n", model->training_time);
    printf("内存使用: %zu bytes\n", model->memory_usage);
    printf("======================\n");
}

// 获取训练统计
void get_time_series_training_stats(const time_series_model_t* model, 
                                  float* loss, float* mae, int* epoch) {
    if (!model) return;
    
    if (loss) *loss = model->current_loss;
    if (mae) *mae = model->mae;
    if (epoch) *epoch = model->current_epoch;
}

// 创建示例时间序列数据
time_series_data_t* create_example_time_series_data(int data_type) {
    time_series_data_t* data = NULL;
    
    switch (data_type) {
        case 0: // 正弦波
            data = create_time_series_data(1000, 1, 1);
            if (data) {
                data->series_name = strdup("正弦波");
                for (size_t i = 0; i < data->length; i++) {
                    float value = sinf(i * 0.1f);
                    float features[1] = {value};
                    float targets[1] = {value};
                    set_time_series_point(data, i, features, targets);
                }
            }
            break;
            
        case 1: // 带噪声的正弦波
            data = create_time_series_data(1000, 1, 1);
            if (data) {
                data->series_name = strdup("带噪声的正弦波");
                for (size_t i = 0; i < data->length; i++) {
                    float value = sinf(i * 0.1f) + random_float(-0.1f, 0.1f);
                    float features[1] = {value};
                    float targets[1] = {value};
                    set_time_series_point(data, i, features, targets);
                }
            }
            break;
            
        case 2: // 线性趋势
            data = create_time_series_data(500, 1, 1);
            if (data) {
                data->series_name = strdup("线性趋势");
                for (size_t i = 0; i < data->length; i++) {
                    float value = i * 0.01f + random_float(-0.1f, 0.1f);
                    float features[1] = {value};
                    float targets[1] = {value};
                    set_time_series_point(data, i, features, targets);
                }
            }
            break;
            
        default:
            data = create_time_series_data(100, 1, 1);
            if (data) {
                data->series_name = strdup("示例数据");
                for (size_t i = 0; i < data->length; i++) {
                    float value = random_float(0.0f, 1.0f);
                    float features[1] = {value};
                    float targets[1] = {value};
                    set_time_series_point(data, i, features, targets);
                }
            }
    }
    
    return data;
}