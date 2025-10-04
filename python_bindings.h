#ifndef PYTHON_BINDINGS_H
#define PYTHON_BINDINGS_H

#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include "nn_module.h"
#include "ai_trainer.h"
#include "autograd.h"
#include "dataloader.h"
#include "model_zoo.h"
#include "federated_learning.h"
#include "visualization_tools.h"

// ==================== Python对象包装器 ====================

typedef struct {
    PyObject_HEAD
    nn_module_t* module;           // 神经网络模块
    bool is_initialized;           // 是否已初始化
} PyNNModule;

typedef struct {
    PyObject_HEAD
    ai_trainer_t* trainer;          // 训练器
    nn_module_t* model;             // 关联的模型
    dataloader_t* dataloader;       // 数据加载器
} PyAITrainer;

typedef struct {
    PyObject_HEAD
    tensor_t* tensor;               // 张量
    bool requires_grad;             // 是否需要梯度
} PyTensor;

typedef struct {
    PyObject_HEAD
    dataloader_t* dataloader;        // 数据加载器
    const char* dataset_path;       // 数据集路径
} PyDataLoader;

typedef struct {
    PyObject_HEAD
    model_zoo_t* model_zoo;         // 模型库
} PyModelZoo;

typedef struct {
    PyObject_HEAD
    federated_server_t* server;     // 联邦学习服务器
} PyFederatedServer;

typedef struct {
    PyObject_HEAD
    federated_client_t* client;     // 联邦学习客户端
} PyFederatedClient;

typedef struct {
    PyObject_HEAD
    training_monitor_t* monitor;    // 训练监控器
} PyTrainingMonitor;

// ==================== Python模块定义 ====================

static PyModuleDef ai_framework_module = {
    PyModuleDef_HEAD_INIT,
    "ai_framework",                 // 模块名称
    "AI框架Python绑定",             // 模块文档
    -1,                            // 模块大小
    NULL,                          // 方法定义
    NULL,                          // 槽位
    NULL,                          // 遍历函数
    NULL,                          // 清除函数
    NULL                           // 释放函数
};

// ==================== 类型定义 ====================

static PyTypeObject PyNNModuleType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ai_framework.NNModule",       // 类型名称
    sizeof(PyNNModule),            // 基本大小
    0,                             // 项目大小
    NULL,                          // 析构函数
    NULL,                          // 打印函数
    NULL,                          // 获取属性
    NULL,                          // 设置属性
    NULL,                          // 异步支持
    NULL,                          // 表示函数
    NULL,                          // 哈希函数
    NULL,                          // 调用函数
    NULL,                          // 字符串表示
    NULL,                          // 获取属性O
    NULL,                          // 设置属性O
    NULL,                          // 缓冲区协议
    Py_TPFLAGS_DEFAULT,            // 标志
    "神经网络模块",                 // 文档
    NULL,                          // 遍历
    NULL,                          // 清除
    NULL,                          // 富比较
    0,                             // 弱引用偏移
    NULL,                          // 迭代
    NULL,                          // 迭代下一个
    NULL,                          // 方法定义
    NULL,                          // 成员
    NULL,                          // 获取设置
    NULL,                          // 基类
    NULL,                          // 字典
    NULL,                          // 描述符获取
    NULL,                          // 描述符设置
    0,                             // 字典偏移
    NULL,                          // 初始化
    NULL,                          // 分配
    NULL,                          // 新建
    NULL,                          // 释放
    NULL,                          // 就绪
    NULL,                          // 访问
    NULL,                          // 异步
    NULL                           // 发送
};

// ==================== Python方法定义 ====================

// NNModule方法
static PyObject* py_nn_module_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void py_nn_module_dealloc(PyNNModule* self);
static PyObject* py_nn_module_forward(PyNNModule* self, PyObject* args);
static PyObject* py_nn_module_parameters(PyNNModule* self, PyObject* args);
static PyObject* py_nn_module_save(PyNNModule* self, PyObject* args);
static PyObject* py_nn_module_load(PyNNModule* self, PyObject* args);

// AITrainer方法
static PyObject* py_ai_trainer_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void py_ai_trainer_dealloc(PyAITrainer* self);
static PyObject* py_ai_trainer_train(PyAITrainer* self, PyObject* args);
static PyObject* py_ai_trainer_evaluate(PyAITrainer* self, PyObject* args);
static PyObject* py_ai_trainer_predict(PyAITrainer* self, PyObject* args);

// Tensor方法
static PyObject* py_tensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void py_tensor_dealloc(PyTensor* self);
static PyObject* py_tensor_shape(PyTensor* self, PyObject* args);
static PyObject* py_tensor_to_numpy(PyTensor* self, PyObject* args);
static PyObject* py_tensor_from_numpy(PyTypeObject* type, PyObject* args);

// DataLoader方法
static PyObject* py_dataloader_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void py_dataloader_dealloc(PyDataLoader* self);
static PyObject* py_dataloader_iter(PyDataLoader* self);
static PyObject* py_dataloader_next(PyDataLoader* self);
static PyObject* py_dataloader_len(PyDataLoader* self);

// ModelZoo方法
static PyObject* py_model_zoo_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void py_model_zoo_dealloc(PyModelZoo* self);
static PyObject* py_model_zoo_list_models(PyModelZoo* self, PyObject* args);
static PyObject* py_model_zoo_load_model(PyModelZoo* self, PyObject* args);
static PyObject* py_model_zoo_download_model(PyModelZoo* self, PyObject* args);

// FederatedLearning方法
static PyObject* py_federated_server_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void py_federated_server_dealloc(PyFederatedServer* self);
static PyObject* py_federated_server_start_training(PyFederatedServer* self, PyObject* args);
static PyObject* py_federated_server_stop_training(PyFederatedServer* self, PyObject* args);

// TrainingMonitor方法
static PyObject* py_training_monitor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void py_training_monitor_dealloc(PyTrainingMonitor* self);
static PyObject* py_training_monitor_plot_loss(PyTrainingMonitor* self, PyObject* args);
static PyObject* py_training_monitor_plot_accuracy(PyTrainingMonitor* self, PyObject* args);

// ==================== 工具函数 ====================

// 张量转换
static PyObject* tensor_to_pyobject(tensor_t* tensor);
static tensor_t* pyobject_to_tensor(PyObject* obj);

// 错误处理
static PyObject* set_error(const char* message);
static int check_tensor_validity(PyTensor* tensor);
static int check_nn_module_validity(PyNNModule* module);

// 配置转换
static federated_config_t* pyobject_to_federated_config(PyObject* config_dict);
static visualization_config_t* pyobject_to_visualization_config(PyObject* config_dict);

// ==================== 模块初始化函数 ====================

PyMODINIT_FUNC PyInit_ai_framework(void);

// ==================== 常量定义 ====================

#define PY_AI_FRAMEWORK_MODULE_NAME "ai_framework"
#define PY_AI_FRAMEWORK_MODULE_DOC "AI框架Python绑定模块"

// ==================== 方法表 ====================

static PyMethodDef nn_module_methods[] = {
    {"forward", (PyCFunction)py_nn_module_forward, METH_VARARGS, "前向传播"},
    {"parameters", (PyCFunction)py_nn_module_parameters, METH_NOARGS, "获取参数"},
    {"save", (PyCFunction)py_nn_module_save, METH_VARARGS, "保存模型"},
    {"load", (PyCFunction)py_nn_module_load, METH_VARARGS, "加载模型"},
    {NULL, NULL, 0, NULL}  // 哨兵
};

static PyMethodDef ai_trainer_methods[] = {
    {"train", (PyCFunction)py_ai_trainer_train, METH_VARARGS, "训练模型"},
    {"evaluate", (PyCFunction)py_ai_trainer_evaluate, METH_VARARGS, "评估模型"},
    {"predict", (PyCFunction)py_ai_trainer_predict, METH_VARARGS, "预测"},
    {NULL, NULL, 0, NULL}  // 哨兵
};

static PyMethodDef tensor_methods[] = {
    {"shape", (PyCFunction)py_tensor_shape, METH_NOARGS, "获取形状"},
    {"to_numpy", (PyCFunction)py_tensor_to_numpy, METH_NOARGS, "转换为numpy数组"},
    {"from_numpy", (PyCFunction)py_tensor_from_numpy, METH_CLASS | METH_VARARGS, "从numpy数组创建"},
    {NULL, NULL, 0, NULL}  // 哨兵
};

static PyMethodDef dataloader_methods[] = {
    {"__iter__", (PyCFunction)py_dataloader_iter, METH_NOARGS, "获取迭代器"},
    {"__next__", (PyCFunction)py_dataloader_next, METH_NOARGS, "下一个批次"},
    {"__len__", (PyCFunction)py_dataloader_len, METH_NOARGS, "数据批次数量"},
    {NULL, NULL, 0, NULL}  // 哨兵
};

static PyMethodDef model_zoo_methods[] = {
    {"list_models", (PyCFunction)py_model_zoo_list_models, METH_NOARGS, "列出可用模型"},
    {"load_model", (PyCFunction)py_model_zoo_load_model, METH_VARARGS, "加载模型"},
    {"download_model", (PyCFunction)py_model_zoo_download_model, METH_VARARGS, "下载模型"},
    {NULL, NULL, 0, NULL}  // 哨兵
};

static PyMethodDef federated_server_methods[] = {
    {"start_training", (PyCFunction)py_federated_server_start_training, METH_NOARGS, "开始训练"},
    {"stop_training", (PyCFunction)py_federated_server_stop_training, METH_NOARGS, "停止训练"},
    {NULL, NULL, 0, NULL}  // 哨兵
};

static PyMethodDef training_monitor_methods[] = {
    {"plot_loss", (PyCFunction)py_training_monitor_plot_loss, METH_VARARGS, "绘制损失曲线"},
    {"plot_accuracy", (PyCFunction)py_training_monitor_plot_accuracy, METH_VARARGS, "绘制准确率曲线"},
    {NULL, NULL, 0, NULL}  // 哨兵
};

#endif // PYTHON_BINDINGS_H