#include "visualization_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

// ==================== 内部工具函数 ====================

static double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static void ensure_directory_exists(const char* path) {
    // 简化实现，实际应该检查并创建目录
    printf("确保目录存在: %s\n", path);
}

static float* allocate_float_array(int size) {
    if (size <= 0) return NULL;
    float* array = malloc(sizeof(float) * size);
    if (array) {
        memset(array, 0, sizeof(float) * size);
    }
    return array;
}

static float* reallocate_float_array(float* array, int old_size, int new_size) {
    if (new_size <= 0) {
        free(array);
        return NULL;
    }
    
    float* new_array = realloc(array, sizeof(float) * new_size);
    if (new_array && new_size > old_size) {
        memset(new_array + old_size, 0, sizeof(float) * (new_size - old_size));
    }
    return new_array;
}

// ==================== 训练监控器实现 ====================

training_monitor_t* create_training_monitor(int max_epochs, const char* log_file) {
    if (max_epochs <= 0) {
        return NULL;
    }
    
    training_monitor_t* monitor = malloc(sizeof(training_monitor_t));
    if (!monitor) {
        return NULL;
    }
    
    memset(monitor, 0, sizeof(training_monitor_t));
    
    monitor->max_epochs = max_epochs;
    monitor->current_epoch = 0;
    monitor->history_capacity = max_epochs;
    monitor->history_size = 0;
    monitor->is_recording = false;
    
    // 分配历史记录数组
    monitor->loss_history = allocate_float_array(max_epochs);
    monitor->accuracy_history = allocate_float_array(max_epochs);
    monitor->learning_rate_history = allocate_float_array(max_epochs);
    
    if (!monitor->loss_history || !monitor->accuracy_history || !monitor->learning_rate_history) {
        destroy_training_monitor(monitor);
        return NULL;
    }
    
    // 设置日志文件
    if (log_file) {
        monitor->log_file = strdup(log_file);
        
        // 创建日志文件头
        FILE* log_fp = fopen(log_file, "w");
        if (log_fp) {
            fprintf(log_fp, "epoch,loss,accuracy,learning_rate,timestamp\\n");
            fclose(log_fp);
        }
    }
    
    printf("训练监控器创建成功，最大轮数: %d\n", max_epochs);
    return monitor;
}

void destroy_training_monitor(training_monitor_t* monitor) {
    if (!monitor) return;
    
    free(monitor->loss_history);
    free(monitor->accuracy_history);
    free(monitor->learning_rate_history);
    free((void*)monitor->log_file);
    free(monitor);
    
    printf("训练监控器已销毁\n");
}

int start_training_monitoring(training_monitor_t* monitor) {
    if (!monitor) {
        return -1;
    }
    
    if (monitor->is_recording) {
        printf("监控器已经在运行\n");
        return -1;
    }
    
    monitor->is_recording = true;
    monitor->current_epoch = 0;
    monitor->history_size = 0;
    
    printf("训练监控已启动\n");
    return 0;
}

int stop_training_monitoring(training_monitor_t* monitor) {
    if (!monitor) {
        return -1;
    }
    
    if (!monitor->is_recording) {
        printf("监控器未在运行\n");
        return -1;
    }
    
    monitor->is_recording = false;
    printf("训练监控已停止，记录轮数: %d\n", monitor->history_size);
    return 0;
}

int record_training_metrics(training_monitor_t* monitor, int epoch, 
                           float loss, float accuracy, float learning_rate) {
    if (!monitor || !monitor->is_recording) {
        return -1;
    }
    
    if (epoch < 0 || epoch >= monitor->max_epochs) {
        printf("无效的轮数: %d\n", epoch);
        return -1;
    }
    
    monitor->current_epoch = epoch;
    
    // 更新历史记录
    if (epoch < monitor->history_capacity) {
        monitor->loss_history[epoch] = loss;
        monitor->accuracy_history[epoch] = accuracy;
        monitor->learning_rate_history[epoch] = learning_rate;
        
        if (epoch >= monitor->history_size) {
            monitor->history_size = epoch + 1;
        }
    } else {
        // 需要扩展容量
        int new_capacity = monitor->history_capacity * 2;
        monitor->loss_history = reallocate_float_array(monitor->loss_history, 
                                                     monitor->history_capacity, new_capacity);
        monitor->accuracy_history = reallocate_float_array(monitor->accuracy_history, 
                                                         monitor->history_capacity, new_capacity);
        monitor->learning_rate_history = reallocate_float_array(monitor->learning_rate_history, 
                                                              monitor->history_capacity, new_capacity);
        
        if (!monitor->loss_history || !monitor->accuracy_history || !monitor->learning_rate_history) {
            printf("内存分配失败\n");
            return -1;
        }
        
        monitor->history_capacity = new_capacity;
        monitor->loss_history[epoch] = loss;
        monitor->accuracy_history[epoch] = accuracy;
        monitor->learning_rate_history[epoch] = learning_rate;
        monitor->history_size = epoch + 1;
    }
    
    // 记录到日志文件
    if (monitor->log_file) {
        FILE* log_fp = fopen(monitor->log_file, "a");
        if (log_fp) {
            fprintf(log_fp, "%d,%.6f,%.6f,%.6f,%ld\\n", 
                    epoch, loss, accuracy, learning_rate, time(NULL));
            fclose(log_fp);
        }
    }
    
    printf("记录训练指标: 轮数=%d, 损失=%.4f, 准确率=%.4f, 学习率=%.6f\n", 
           epoch, loss, accuracy, learning_rate);
    
    return 0;
}

// ==================== 可视化管理器实现 ====================

visualization_manager_t* create_visualization_manager(const char* dashboard_path) {
    visualization_manager_t* manager = malloc(sizeof(visualization_manager_t));
    if (!manager) {
        return NULL;
    }
    
    memset(manager, 0, sizeof(visualization_manager_t));
    
    if (dashboard_path) {
        manager->dashboard_path = strdup(dashboard_path);
        ensure_directory_exists(dashboard_path);
    }
    
    manager->real_time_update = false;
    manager->update_interval = 1000; // 默认1秒
    
    printf("可视化管理器创建成功，仪表板路径: %s\n", 
           dashboard_path ? dashboard_path : "默认");
    
    return manager;
}

void destroy_visualization_manager(visualization_manager_t* manager) {
    if (!manager) return;
    
    free((void*)manager->dashboard_path);
    free(manager);
    
    printf("可视化管理器已销毁\n");
}

int add_monitor_to_manager(visualization_manager_t* manager, training_monitor_t* monitor) {
    if (!manager || !monitor) {
        return -1;
    }
    
    if (manager->num_monitors >= 10) {
        printf("监控器数量已达上限\n");
        return -1;
    }
    
    manager->monitors[manager->num_monitors++] = monitor;
    printf("监控器已添加到管理器，当前数量: %d\n", manager->num_monitors);
    
    return 0;
}

int start_visualization_dashboard(visualization_manager_t* manager) {
    if (!manager) {
        return -1;
    }
    
    printf("启动可视化仪表板\n");
    printf("监控器数量: %d\n", manager->num_monitors);
    
    // 创建HTML仪表板文件
    if (manager->dashboard_path) {
        char html_path[256];
        snprintf(html_path, sizeof(html_path), "%s/dashboard.html", manager->dashboard_path);
        
        FILE* html_fp = fopen(html_path, "w");
        if (html_fp) {
            fprintf(html_fp, "<!DOCTYPE html>\n");
            fprintf(html_fp, "<html>\n");
            fprintf(html_fp, "<head>\n");
            fprintf(html_fp, "    <title>AI训练监控仪表板</title>\n");
            fprintf(html_fp, "    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>\n");
            fprintf(html_fp, "</head>\n");
            fprintf(html_fp, "<body>\n");
            fprintf(html_fp, "    <h1>AI训练监控仪表板</h1>\n");
            fprintf(html_fp, "    <div id=\"loss-chart\"></div>\n");
            fprintf(html_fp, "    <div id=\"accuracy-chart\"></div>\n");
            fprintf(html_fp, "    <script>\n");
            fprintf(html_fp, "        // 这里会动态加载数据\n");
            fprintf(html_fp, "    </script>\n");
            fprintf(html_fp, "</body>\n");
            fprintf(html_fp, "</html>\n");
            fclose(html_fp);
            
            printf("仪表板HTML文件已创建: %s\n", html_path);
        }
    }
    
    return 0;
}

int stop_visualization_dashboard(visualization_manager_t* manager) {
    if (!manager) {
        return -1;
    }
    
    printf("停止可视化仪表板\n");
    return 0;
}

// ==================== 基础可视化实现 ====================

int plot_loss_curve(training_monitor_t* monitor, visualization_config_t* config) {
    if (!monitor || !config) {
        return -1;
    }
    
    printf("绘制损失曲线\n");
    printf("标题: %s\n", config->title ? config->title : "默认标题");
    printf("数据点数: %d\n", monitor->history_size);
    
    // 生成SVG格式的损失曲线（简化实现）
    if (strcmp(config->output_format, "svg") == 0) {
        char svg_path[256];
        snprintf(svg_path, sizeof(svg_path), "%s/loss_curve.svg", 
                 config->output_path ? config->output_path : ".");
        
        FILE* svg_fp = fopen(svg_path, "w");
        if (svg_fp) {
            fprintf(svg_fp, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            fprintf(svg_fp, "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\">\n", 
                    config->width, config->height);
            fprintf(svg_fp, "    <rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n");
            fprintf(svg_fp, "    <text x=\"50%%\" y=\"30\" text-anchor=\"middle\" font-size=\"%d\">%s</text>\n",
                    config->font_size, config->title ? config->title : "损失曲线");
            
            // 简化绘制线条
            if (monitor->history_size > 1) {
                fprintf(svg_fp, "    <polyline points=\"");
                for (int i = 0; i < monitor->history_size; i++) {
                    int x = 50 + i * (config->width - 100) / monitor->history_size;
                    int y = config->height - 50 - (int)(monitor->loss_history[i] * (config->height - 100));
                    fprintf(svg_fp, "%d,%d ", x, y);
                }
                fprintf(svg_fp, "\" fill=\"none\" stroke=\"red\" stroke-width=\"2\"/>\n");
            }
            
            fprintf(svg_fp, "</svg>\n");
            fclose(svg_fp);
            
            printf("损失曲线SVG已保存: %s\n", svg_path);
        }
    }
    
    return 0;
}

int plot_accuracy_curve(training_monitor_t* monitor, visualization_config_t* config) {
    if (!monitor || !config) {
        return -1;
    }
    
    printf("绘制准确率曲线\n");
    printf("标题: %s\n", config->title ? config->title : "默认标题");
    printf("数据点数: %d\n", monitor->history_size);
    
    // 生成HTML格式的准确率曲线（简化实现）
    if (strcmp(config->output_format, "html") == 0) {
        char html_path[256];
        snprintf(html_path, sizeof(html_path), "%s/accuracy_curve.html", 
                 config->output_path ? config->output_path : ".");
        
        FILE* html_fp = fopen(html_path, "w");
        if (html_fp) {
            fprintf(html_fp, "<!DOCTYPE html>\n");
            fprintf(html_fp, "<html>\n");
            fprintf(html_fp, "<head>\n");
            fprintf(html_fp, "    <title>%s</title>\n", config->title ? config->title : "准确率曲线");
            fprintf(html_fp, "    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>\n");
            fprintf(html_fp, "</head>\n");
            fprintf(html_fp, "<body>\n");
            fprintf(html_fp, "    <div id=\"chart\"></div>\n");
            fprintf(html_fp, "    <script>\n");
            fprintf(html_fp, "        var epochs = [");
            for (int i = 0; i < monitor->history_size; i++) {
                fprintf(html_fp, "%d%s", i, i < monitor->history_size - 1 ? "," : "");
            }
            fprintf(html_fp, "];\n");
            fprintf(html_fp, "        var accuracy = [");
            for (int i = 0; i < monitor->history_size; i++) {
                fprintf(html_fp, "%.4f%s", monitor->accuracy_history[i], 
                        i < monitor->history_size - 1 ? "," : "");
            }
            fprintf(html_fp, "];\n");
            fprintf(html_fp, "        var trace = {\n");
            fprintf(html_fp, "            x: epochs,\n");
            fprintf(html_fp, "            y: accuracy,\n");
            fprintf(html_fp, "            type: 'scatter',\n");
            fprintf(html_fp, "            mode: 'lines+markers',\n");
            fprintf(html_fp, "            name: '准确率'\n");
            fprintf(html_fp, "        };\n");
            fprintf(html_fp, "        var layout = {\n");
            fprintf(html_fp, "            title: '%s',\n", config->title ? config->title : "准确率曲线");
            fprintf(html_fp, "            xaxis: {title: '%s'},\n", config->x_label ? config->x_label : "轮数");
            fprintf(html_fp, "            yaxis: {title: '%s'}\n", config->y_label ? config->y_label : "准确率");
            fprintf(html_fp, "        };\n");
            fprintf(html_fp, "        Plotly.newPlot('chart', [trace], layout);\n");
            fprintf(html_fp, "    </script>\n");
            fprintf(html_fp, "</body>\n");
            fprintf(html_fp, "</html>\n");
            fclose(html_fp);
            
            printf("准确率曲线HTML已保存: %s\n", html_path);
        }
    }
    
    return 0;
}

int plot_weight_distribution(nn_module_t* model, visualization_config_t* config) {
    if (!model || !config) {
        return -1;
    }
    
    printf("绘制权重分布\n");
    printf("模型层数: %d\n", model->num_layers);
    
    // 简化实现，实际需要分析模型权重
    return 0;
}

int plot_gradient_flow(nn_module_t* model, visualization_config_t* config) {
    if (!model || !config) {
        return -1;
    }
    
    printf("绘制梯度流\n");
    
    // 简化实现，实际需要分析梯度信息
    return 0;
}

// ==================== 高级可视化实现 ====================

int create_training_progress_dashboard(training_monitor_t** monitors, int num_monitors, visualization_config_t* config) {
    if (!monitors || num_monitors <= 0 || !config) {
        return -1;
    }
    
    printf("创建训练进度仪表板\n");
    printf("监控器数量: %d\n", num_monitors);
    
    // 创建综合仪表板HTML
    if (strcmp(config->output_format, "html") == 0) {
        char html_path[256];
        snprintf(html_path, sizeof(html_path), "%s/training_dashboard.html", 
                 config->output_path ? config->output_path : ".");
        
        FILE* html_fp = fopen(html_path, "w");
        if (html_fp) {
            fprintf(html_fp, "<!DOCTYPE html>\n");
            fprintf(html_fp, "<html>\n");
            fprintf(html_fp, "<head>\n");
            fprintf(html_fp, "    <title>训练进度仪表板</title>\n");
            fprintf(html_fp, "    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>\n");
            fprintf(html_fp, "    <style>\n");
            fprintf(html_fp, "        .chart { width: 45%%; height: 400px; display: inline-block; margin: 10px; }\n");
            fprintf(html_fp, "    </style>\n");
            fprintf(html_fp, "</head>\n");
            fprintf(html_fp, "<body>\n");
            fprintf(html_fp, "    <h1>训练进度仪表板</h1>\n");
            
            for (int i = 0; i < num_monitors; i++) {
                fprintf(html_fp, "    <h2>模型 %d</h2>\n", i + 1);
                fprintf(html_fp, "    <div class=\"chart\" id=\"loss-chart-%d\"></div>\n", i);
                fprintf(html_fp, "    <div class=\"chart\" id=\"accuracy-chart-%d\"></div>\n", i);
            }
            
            fprintf(html_fp, "    <script>\n");
            
            for (int i = 0; i < num_monitors; i++) {
                if (monitors[i]) {
                    fprintf(html_fp, "        // 模型 %d 数据\n", i + 1);
                    fprintf(html_fp, "        var epochs%d = [", i);
                    for (int j = 0; j < monitors[i]->history_size; j++) {
                        fprintf(html_fp, "%d%s", j, j < monitors[i]->history_size - 1 ? "," : "");
                    }
                    fprintf(html_fp, "];\n");
                    
                    fprintf(html_fp, "        var loss%d = [", i);
                    for (int j = 0; j < monitors[i]->history_size; j++) {
                        fprintf(html_fp, "%.4f%s", monitors[i]->loss_history[j], 
                                j < monitors[i]->history_size - 1 ? "," : "");
                    }
                    fprintf(html_fp, "];\n");
                    
                    fprintf(html_fp, "        var accuracy%d = [", i);
                    for (int j = 0; j < monitors[i]->history_size; j++) {
                        fprintf(html_fp, "%.4f%s", monitors[i]->accuracy_history[j], 
                                j < monitors[i]->history_size - 1 ? "," : "");
                    }
                    fprintf(html_fp, "];\n");
                }
            }
            
            fprintf(html_fp, "    </script>\n");
            fprintf(html_fp, "</body>\n");
            fprintf(html_fp, "</html>\n");
            fclose(html_fp);
            
            printf("训练进度仪表板已创建: %s\n", html_path);
        }
    }
    
    return 0;
}

// ==================== 工具函数实现 ====================

uint32_t rgb_to_uint32(uint8_t r, uint8_t g, uint8_t b) {
    return (r << 16) | (g << 8) | b;
}

void uint32_to_rgb(uint32_t color, uint8_t* r, uint8_t* g, uint8_t* b) {
    if (r) *r = (color >> 16) & 0xFF;
    if (g) *g = (color >> 8) & 0xFF;
    if (b) *b = color & 0xFF;
}

const char* get_default_colors(int index) {
    static const char* colors[] = {
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
        "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43"
    };
    
    if (index < 0 || index >= 10) {
        return "#000000";
    }
    return colors[index];
}

float* smooth_data(float* data, int length, int window_size) {
    if (!data || length <= 0 || window_size <= 0) {
        return NULL;
    }
    
    float* smoothed = allocate_float_array(length);
    if (!smoothed) {
        return NULL;
    }
    
    for (int i = 0; i < length; i++) {
        int start = i - window_size / 2;
        int end = i + window_size / 2;
        
        if (start < 0) start = 0;
        if (end >= length) end = length - 1;
        
        smoothed[i] = calculate_moving_average(data, start, end);
    }
    
    return smoothed;
}

float calculate_moving_average(float* data, int start, int end) {
    if (!data || start > end) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    int count = end - start + 1;
    
    for (int i = start; i <= end; i++) {
        sum += data[i];
    }
    
    return sum / count;
}

// ==================== 错误处理 ====================

const char* get_visualization_error_message(int error_code) {
    switch (error_code) {
        case 0: return "成功";
        case -1: return "通用错误";
        case -2: return "内存分配失败";
        case -3: return "文件操作失败";
        case -4: return "无效参数";
        case -5: return "不支持的操作";
        default: return "未知错误";
    }
}

// 其他函数实现...