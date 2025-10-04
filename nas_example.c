#include "neural_architecture_search.h"
#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 进度回调函数
void progress_callback(int progress, const char* message) {
    printf("进度: %d%% - %s\n", progress, message);
}

// 生成示例数据
void generate_sample_data(training_data_t* train_data, training_data_t* val_data) {
    printf("生成示例数据...\n");
    
    // 简化示例：创建一些随机数据
    // 在实际应用中，这里应该加载真实数据集
    
    // 训练数据
    train_data->num_samples = 1000;
    train_data->input_dim = 784;  // MNIST图像大小
    train_data->output_dim = 10;  // 10个类别
    
    // 验证数据
    val_data->num_samples = 200;
    val_data->input_dim = 784;
    val_data->output_dim = 10;
    
    printf("示例数据生成完成: 训练样本=%d, 验证样本=%d\n", 
           train_data->num_samples, val_data->num_samples);
}

// 演示随机搜索
void demo_random_search(neural_architecture_search_manager_t* nas_manager,
                       const training_data_t* train_data,
                       const training_data_t* val_data) {
    printf("\n=== 演示随机搜索 ===\n");
    
    // 设置搜索方法
    set_nas_search_method(nas_manager, NAS_SEARCH_RANDOM);
    
    // 配置搜索参数
    nas_config_t config = nas_manager->config;
    config.num_episodes = 20;  // 减少评估次数以加快演示
    configure_nas_search(nas_manager, &config);
    
    // 执行搜索
    nas_search_result_t result = perform_architecture_search(nas_manager, train_data, val_data);
    
    // 显示结果
    printf("随机搜索结果:\n");
    printf("最佳适应度: %.3f\n", result.best_fitness);
    printf("搜索时间: %.2f秒\n", result.search_time);
    printf("评估架构数: %d\n", result.total_evaluations);
    
    // 显示最佳架构
    printf("最佳架构:\n");
    print_architecture(&result.best_architecture);
    
    // 分析架构复杂度
    analyze_architecture_complexity(&result.best_architecture);
}

// 演示进化算法搜索
void demo_evolutionary_search(neural_architecture_search_manager_t* nas_manager,
                             const training_data_t* train_data,
                             const training_data_t* val_data) {
    printf("\n=== 演示进化算法搜索 ===\n");
    
    // 设置搜索方法
    set_nas_search_method(nas_manager, NAS_SEARCH_EVOLUTIONARY);
    
    // 配置搜索参数
    nas_config_t config = nas_manager->config;
    config.population_size = 10;  // 小种群以加快演示
    config.num_generations = 5;   // 少代数以加快演示
    configure_nas_search(nas_manager, &config);
    
    // 执行搜索
    nas_search_result_t result = perform_architecture_search(nas_manager, train_data, val_data);
    
    // 显示结果
    printf("进化算法搜索结果:\n");
    printf("最佳适应度: %.3f\n", result.best_fitness);
    printf("搜索时间: %.2f秒\n", result.search_time);
    printf("评估架构数: %d\n", result.total_evaluations);
    
    // 显示最佳架构
    printf("最佳架构:\n");
    print_architecture(&result.best_architecture);
    
    // 可视化架构结构
    visualize_architecture_structure(&result.best_architecture);
}

// 演示多目标搜索
void demo_multi_objective_search(neural_architecture_search_manager_t* nas_manager,
                                const training_data_t* train_data,
                                const training_data_t* val_data) {
    printf("\n=== 演示多目标搜索 ===\n");
    
    // 执行多目标搜索
    perform_multi_objective_nas(nas_manager, train_data, val_data, 
                               0.7f,  // 准确率权重
                               0.2f,  // 复杂度权重
                               0.1f); // 延迟权重
    
    // 获取搜索结果
    const nas_search_result_t* result = get_nas_search_result(nas_manager);
    
    if (result) {
        printf("多目标搜索结果:\n");
        printf("最佳适应度: %.3f\n", result->best_fitness);
        printf("搜索时间: %.2f秒\n", result->search_time);
        
        // 显示最佳架构
        printf("多目标优化后的最佳架构:\n");
        print_architecture(&result->best_architecture);
    }
}

// 演示架构比较
void demo_architecture_comparison(void) {
    printf("\n=== 演示架构比较 ===\n");
    
    // 创建几个不同的架构
    architecture_encoding_t arch1 = create_simple_architecture(3, 128);
    architecture_encoding_t arch2 = create_simple_architecture(5, 64);
    architecture_encoding_t arch3 = create_simple_architecture(4, 256);
    
    printf("架构1 (3层, 每层128单元):\n");
    print_architecture(&arch1);
    
    printf("架构2 (5层, 每层64单元):\n");
    print_architecture(&arch2);
    
    printf("架构3 (4层, 每层256单元):\n");
    print_architecture(&arch3);
    
    // 比较架构
    printf("架构比较结果:\n");
    
    int cmp12 = compare_architectures(&arch1, &arch2);
    if (cmp12 < 0) {
        printf("架构1 比 架构2 简单\n");
    } else if (cmp12 > 0) {
        printf("架构1 比 架构2 复杂\n");
    } else {
        printf("架构1 和 架构2 复杂度相同\n");
    }
    
    int cmp13 = compare_architectures(&arch1, &arch3);
    if (cmp13 < 0) {
        printf("架构1 比 架构3 简单\n");
    } else if (cmp13 > 0) {
        printf("架构1 比 架构3 复杂\n");
    } else {
        printf("架构1 和 架构3 复杂度相同\n");
    }
    
    // 清理内存
    if (arch1.layers) free(arch1.layers);
    if (arch2.layers) free(arch2.layers);
    if (arch3.layers) free(arch3.layers);
}

// 演示搜索方法基准测试
void demo_search_benchmark(neural_architecture_search_manager_t* nas_manager,
                          const training_data_t* train_data,
                          const training_data_t* val_data) {
    printf("\n=== 演示搜索方法基准测试 ===\n");
    
    // 配置快速基准测试
    nas_config_t config = nas_manager->config;
    config.num_episodes = 10;      // 少量评估
    config.population_size = 5;    // 小种群
    config.num_generations = 3;    // 少代数
    configure_nas_search(nas_manager, &config);
    
    // 执行基准测试
    benchmark_nas_search_methods(nas_manager, train_data, val_data);
}

int main(void) {
    printf("神经网络架构搜索(NAS)示例程序\n");
    printf("==============================\n\n");
    
    // 设置随机种子
    srand((unsigned int)time(NULL));
    
    // 创建NAS管理器
    neural_architecture_search_manager_t* nas_manager = create_nas_manager();
    if (!nas_manager) {
        printf("创建NAS管理器失败!\n");
        return -1;
    }
    
    // 设置进度回调
    set_nas_progress_callback(nas_manager, progress_callback);
    
    // 生成示例数据
    training_data_t train_data, val_data;
    generate_sample_data(&train_data, &val_data);
    
    // 演示各种功能
    demo_random_search(nas_manager, &train_data, &val_data);
    demo_evolutionary_search(nas_manager, &train_data, &val_data);
    demo_multi_objective_search(nas_manager, &train_data, &val_data);
    demo_architecture_comparison();
    demo_search_benchmark(nas_manager, &train_data, &val_data);
    
    // 演示架构保存和加载
    printf("\n=== 演示架构保存和加载 ===\n");
    
    // 创建一个简单架构
    architecture_encoding_t test_arch = create_simple_architecture(4, 100);
    printf("原始架构:\n");
    print_architecture(&test_arch);
    
    // 保存架构
    save_nas_architecture(&test_arch, "test_architecture.nas");
    
    // 加载架构
    architecture_encoding_t loaded_arch = load_nas_architecture("test_architecture.nas");
    printf("加载的架构:\n");
    print_architecture(&loaded_arch);
    
    // 清理内存
    if (test_arch.layers) free(test_arch.layers);
    if (loaded_arch.layers) free(loaded_arch.layers);
    
    // 演示性能优化
    printf("\n=== 演示性能优化 ===\n");
    optimize_nas_search_performance(nas_manager, 4);  // 设置最大并行评估数为4
    printf("NAS搜索性能优化配置完成\n");
    
    // 检查搜索状态
    printf("\n=== 检查搜索状态 ===\n");
    int status = get_nas_search_status(nas_manager);
    switch (status) {
        case NAS_STATUS_READY:
            printf("NAS管理器状态: 准备就绪\n");
            break;
        case NAS_STATUS_SEARCHING:
            printf("NAS管理器状态: 正在搜索\n");
            break;
        case NAS_STATUS_COMPLETED:
            printf("NAS管理器状态: 搜索完成\n");
            break;
        case NAS_STATUS_ERROR:
            printf("NAS管理器状态: 错误\n");
            break;
        default:
            printf("NAS管理器状态: 未知\n");
            break;
    }
    
    // 清理资源
    destroy_nas_manager(nas_manager);
    
    printf("\n=== NAS示例程序执行完成 ===\n");
    printf("所有功能演示完毕!\n");
    
    return 0;
}