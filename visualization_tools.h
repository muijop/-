#ifndef VISUALIZATION_TOOLS_H
#define VISUALIZATION_TOOLS_H

#include <stdint.h>
#include <stdbool.h>
#include "nn_module.h"
#include "ai_trainer.h"

// ==================== 可视化类型定义 ====================

typedef enum {
    VIS_LOSS_CURVE = 0,           // 损失曲线
    VIS_ACCURACY_CURVE = 1,       // 准确率曲线
    VIS_WEIGHT_DISTRIBUTION = 2,   // 权重分布
    VIS_GRADIENT_FLOW = 3,         // 梯度流
    VIS_ACTIVATION_MAPS = 4,       // 激活图
    VIS_CONFUSION_MATRIX = 5,      // 混淆矩阵
    VIS_ROC_CURVE = 6,             // ROC曲线
    VIS_PRECISION_RECALL = 7,      // 精确率-召回率曲线
    VIS_TRAINING_PROGRESS = 8,     // 训练进度
    VIS_MODEL_ARCHITECTURE = 9     // 模型架构
} visualization_type_t;

// ==================== 可视化配置结构体 ====================

typedef struct {
    visualization_type_t type;     // 可视化类型
    int width;                     // 图像宽度
    int height;                    // 图像高度
    const char* title;             // 图表标题
    const char* x_label;           // X轴标签
    const char* y_label;           // Y轴标签
    bool show_grid;                // 是否显示网格
    bool show_legend;              // 是否显示图例
    int font_size;                 // 字体大小
    const char* output_format;      // 输出格式 ("png", "svg", "html")
    const char* output_path;       // 输出路径
} visualization_config_t;

// ==================== 训练监控器结构体 ====================

typedef struct {
    int max_epochs;                // 最大训练轮数
    int current_epoch;             // 当前轮数
    float* loss_history;           // 损失历史
    float* accuracy_history;       // 准确率历史
    float* learning_rate_history;  // 学习率历史
    int history_size;              // 历史记录大小
    int history_capacity;          // 历史记录容量
    bool is_recording;             // 是否正在记录
    const char* log_file;          // 日志文件路径
} training_monitor_t;

// ==================== 可视化管理器结构体 ====================

typedef struct {
    training_monitor_t* monitors[10];  // 监控器数组
    int num_monitors;                  // 监控器数量
    const char* dashboard_path;       // 仪表板路径
    bool real_time_update;             // 是否实时更新
    int update_interval;               // 更新间隔(毫秒)
} visualization_manager_t;

// ==================== 图表数据点结构体 ====================

typedef struct {
    float x;                        // X坐标
    float y;                        // Y坐标
    const char* label;              // 数据点标签
    uint32_t color;                 // 颜色 (RGB)
} data_point_t;

// ==================== 图表数据集结构体 ====================

typedef struct {
    data_point_t* points;           // 数据点数组
    int num_points;                 // 数据点数量
    const char* series_name;        // 系列名称
    uint32_t line_color;            // 线条颜色
    uint32_t point_color;           // 点颜色
    int line_width;                 // 线条宽度
    bool show_points;               // 是否显示点
} chart_dataset_t;

// ==================== 模型可视化结构体 ====================

typedef struct {
    nn_module_t* model;             // 要可视化的模型
    const char* layer_name;         // 层名称
    int channel;                    // 通道索引
    int sample_index;               // 样本索引
    bool show_weights;              // 是否显示权重
    bool show_activations;          // 是否显示激活
    bool show_gradients;            // 是否显示梯度
} model_visualization_t;

// ==================== 可视化工具API ====================

// 训练监控器API
training_monitor_t* create_training_monitor(int max_epochs, const char* log_file);
void destroy_training_monitor(training_monitor_t* monitor);
int start_training_monitoring(training_monitor_t* monitor);
int stop_training_monitoring(training_monitor_t* monitor);
int record_training_metrics(training_monitor_t* monitor, int epoch, float loss, float accuracy, float learning_rate);

// 可视化管理器API
visualization_manager_t* create_visualization_manager(const char* dashboard_path);
void destroy_visualization_manager(visualization_manager_t* manager);
int add_monitor_to_manager(visualization_manager_t* manager, training_monitor_t* monitor);
int start_visualization_dashboard(visualization_manager_t* manager);
int stop_visualization_dashboard(visualization_manager_t* manager);

// 基础可视化API
int plot_loss_curve(training_monitor_t* monitor, visualization_config_t* config);
int plot_accuracy_curve(training_monitor_t* monitor, visualization_config_t* config);
int plot_weight_distribution(nn_module_t* model, visualization_config_t* config);
int plot_gradient_flow(nn_module_t* model, visualization_config_t* config);
int plot_activation_maps(model_visualization_t* vis_config, visualization_config_t* config);
int plot_confusion_matrix(float** matrix, int num_classes, visualization_config_t* config);
int plot_roc_curve(float* tpr, float* fpr, int num_points, visualization_config_t* config);
int plot_precision_recall_curve(float* precision, float* recall, int num_points, visualization_config_t* config);

// 高级可视化API
int create_training_progress_dashboard(training_monitor_t** monitors, int num_monitors, visualization_config_t* config);
int visualize_model_architecture(nn_module_t* model, visualization_config_t* config);
int create_comparison_chart(chart_dataset_t** datasets, int num_datasets, visualization_config_t* config);

// 实时可视化API
int start_real_time_monitoring(training_monitor_t* monitor, int update_interval);
int stop_real_time_monitoring(training_monitor_t* monitor);
int update_real_time_display(visualization_manager_t* manager);

// 导出和保存API
int save_visualization_to_file(const char* data, const char* filename, const char* format);
int export_chart_as_html(chart_dataset_t* dataset, const char* filename);
int export_chart_as_png(chart_dataset_t* dataset, const char* filename);
int export_chart_as_svg(chart_dataset_t* dataset, const char* filename);

// 工具函数
uint32_t rgb_to_uint32(uint8_t r, uint8_t g, uint8_t b);
void uint32_to_rgb(uint32_t color, uint8_t* r, uint8_t* g, uint8_t* b);
const char* get_default_colors(int index);
float* smooth_data(float* data, int length, int window_size);
float calculate_moving_average(float* data, int start, int end);

// 错误处理
const char* get_visualization_error_message(int error_code);

#endif // VISUALIZATION_TOOLS_H