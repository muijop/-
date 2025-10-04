#ifndef AI_FRAMEWORK_LOW_LEVEL_ARCHITECTURE_H
#define AI_FRAMEWORK_LOW_LEVEL_ARCHITECTURE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

// 底层架构版本定义
#define AI_LL_ARCH_VERSION_MAJOR 1
#define AI_LL_ARCH_VERSION_MINOR 0
#define AI_LL_ARCH_VERSION_PATCH 0
#define AI_LL_ARCH_VERSION "1.0.0"

// 内存对齐定义
#define AI_CACHE_LINE_SIZE 64
#define AI_PAGE_SIZE 4096
#define AI_VECTOR_WIDTH 32  // AVX-512: 512位 = 64字节
#define AI_MAX_VECTOR_REGISTERS 32

// 架构特性检测
typedef enum {
    CPU_FEATURE_SSE     = 1 << 0,
    CPU_FEATURE_SSE2    = 1 << 1,
    CPU_FEATURE_SSE3    = 1 << 2,
    CPU_FEATURE_SSSE3   = 1 << 3,
    CPU_FEATURE_SSE41   = 1 << 4,
    CPU_FEATURE_SSE42   = 1 << 5,
    CPU_FEATURE_AVX     = 1 << 6,
    CPU_FEATURE_AVX2    = 1 << 7,
    CPU_FEATURE_AVX512F = 1 << 8,
    CPU_FEATURE_AVX512BW= 1 << 9,
    CPU_FEATURE_AVX512DQ= 1 << 10,
    CPU_FEATURE_AVX512VL= 1 << 11,
    CPU_FEATURE_FMA     = 1 << 12,
    CPU_FEATURE_BMI1    = 1 << 13,
    CPU_FEATURE_BMI2    = 1 << 14,
    CPU_FEATURE_POPCNT  = 1 << 15,
    CPU_FEATURE_LZCNT   = 1 << 16,
    CPU_FEATURE_TZCNT   = 1 << 17,
    CPU_FEATURE_NEON    = 1 << 18,  // ARM NEON
    CPU_FEATURE_SVE     = 1 << 19,  // ARM SVE
    CPU_FEATURE_CHINESE_STRING = 1 << 20  // 中文处理优化
} CpuFeatureFlags;

// 内存池配置
typedef struct {
    size_t pool_size;           // 内存池总大小
    size_t block_size;          // 块大小
    size_t alignment;           // 对齐要求
    bool use_huge_pages;        // 是否使用大页
    bool enable_numa_binding;   // NUMA绑定
    int numa_node;              // NUMA节点
    size_t max_cached_blocks;   // 最大缓存块数
    float gc_threshold;         // 垃圾回收阈值
    bool enable_chinese_optimization; // 中文优化
} MemoryPoolConfig;

// 内存池统计
typedef struct {
    size_t total_allocated;     // 总分配内存
    size_t total_used;          // 已使用内存
    size_t total_cached;        // 缓存内存
    size_t allocation_count;    // 分配次数
    size_t deallocation_count;  // 释放次数
    size_t cache_hits;          // 缓存命中
    size_t cache_misses;        // 缓存未命中
    double average_allocation_time; // 平均分配时间
    double peak_usage;          // 峰值使用
} MemoryPoolStats;

// 向量寄存器状态
typedef struct {
    __m512 vec_regs[AI_MAX_VECTOR_REGISTERS];  // AVX-512寄存器
    uint64_t reg_mask;                         // 寄存器使用掩码
    int spill_count;                           // 寄存器溢出次数
    int reload_count;                          // 寄存器重载次数
} VectorRegisterState;

// SIMD执行上下文
typedef struct {
    VectorRegisterState vec_state;
    void* aligned_buffer;                      // 对齐缓冲区
    size_t buffer_size;                        // 缓冲区大小
    bool use_gather_scatter;                   // 使用gather/scatter
    bool use_masking;                          // 使用掩码操作
    int unroll_factor;                         // 展开因子
    bool enable_fusion;                        // 启用融合
} SimdExecutionContext;

// 线程池任务
typedef struct ThreadPoolTask ThreadPoolTask;
struct ThreadPoolTask {
    void (*function)(void* arg);
    void* argument;
    ThreadPoolTask* next;
    int priority;
    bool completed;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

// 线程池配置
typedef struct {
    int num_threads;              // 线程数
    int max_tasks;               // 最大任务数
    bool bind_to_cores;          // 绑定到核心
    bool enable_work_stealing;   // 启用工作窃取
    int steal_threshold;         // 窃取阈值
    size_t task_queue_size;      // 任务队列大小
    bool enable_chinese_scheduling; // 中文调度优化
} ThreadPoolConfig;

// 缓存层次结构
typedef struct {
    size_t l1_size;              // L1缓存大小
    size_t l1_line_size;         // L1缓存行大小
    size_t l2_size;              // L2缓存大小
    size_t l2_line_size;         // L2缓存行大小
    size_t l3_size;              // L3缓存大小
    size_t l3_line_size;         // L3缓存行大小
    int l1_associativity;        // L1关联度
    int l2_associativity;        // L2关联度
    int l3_associativity;        // L3关联度
    size_t prefetch_distance;    // 预取距离
    int prefetch_degree;         // 预取度
} CacheHierarchy;

// NUMA拓扑信息
typedef struct {
    int num_nodes;               // NUMA节点数
    int* cores_per_node;         // 每节点核心数
    size_t* memory_per_node;     // 每节点内存
    int** node_distance;         // 节点距离矩阵
    bool enable_numa_balancing;    // NUMA平衡
    int preferred_node;          // 首选节点
} NumaTopology;

// 性能计数器
typedef struct {
    uint64_t cycles;             // CPU周期
    uint64_t instructions;       // 指令数
    uint64_t cache_misses;       // 缓存未命中
    uint64_t branch_misses;      // 分支预测失败
    uint64_t page_faults;        // 页错误
    uint64_t context_switches;   // 上下文切换
    double ipc;                  // 每周期指令数
    double cache_miss_rate;      // 缓存未命中率
    double branch_miss_rate;     // 分支预测失败率
    uint64_t vector_instructions; // 向量指令数
    uint64_t chinese_ops;        // 中文操作数
} PerformanceCounters;

// 硬件抽象层接口
typedef struct {
    // CPU信息
    char vendor[64];             // CPU厂商
    char brand[256];             // CPU品牌
    int family;                  // CPU家族
    int model;                   // CPU型号
    int stepping;                // CPU步进
    int physical_cores;          // 物理核心数
    int logical_cores;           // 逻辑核心数
    int threads_per_core;        // 每核心线程数
    uint64_t features;           // 特性标志
    double base_frequency;       // 基础频率
    double max_frequency;        // 最大频率
    
    // 缓存信息
    CacheHierarchy cache;
    
    // NUMA信息
    NumaTopology numa;
    
    // 性能计数器
    PerformanceCounters perf;
    
    // 中文优化
    bool chinese_string_acceleration;
    bool chinese_text_processing;
} HardwareInfo;

// 内存池接口
typedef struct MemoryPool MemoryPool;

struct MemoryPool {
    // 配置和统计
    MemoryPoolConfig config;
    MemoryPoolStats stats;
    
    // 内部结构
    void** free_blocks;          // 空闲块列表
    size_t num_free_blocks;      // 空闲块数
    void** used_blocks;          // 使用块列表
    size_t num_used_blocks;      // 使用块数
    pthread_mutex_t mutex;       // 互斥锁
    
    // 垃圾回收
    void* gc_context;
    void (*gc_callback)(void* context);
    
    // 中文优化
    void* chinese_buffer_pool;
    size_t chinese_buffer_size;
};

// 线程池接口
typedef struct ThreadPool ThreadPool;

struct ThreadPool {
    ThreadPoolConfig config;
    pthread_t* threads;          // 线程数组
    ThreadPoolTask* task_queue;  // 任务队列
    size_t queue_head;           // 队列头
    size_t queue_tail;           // 队列尾
    size_t queue_size;           // 队列大小
    pthread_mutex_t queue_mutex; // 队列锁
    pthread_cond_t queue_cond;   // 队列条件变量
    bool shutdown;               // 关闭标志
    
    // 工作窃取
    ThreadPool** steal_victims;  // 窃取目标
    int num_steal_victims;       // 窃取目标数
    
    // 统计
    size_t tasks_completed;
    size_t tasks_stolen;
    double average_task_time;
    
    // 中文优化
    bool chinese_scheduling_enabled;
    void* chinese_task_queue;
};

// SIMD内核函数指针类型
typedef void (*SimdKernelFunc)(void* dst, const void* src1, const void* src2, size_t n);
typedef void (*SimdUnaryKernelFunc)(void* dst, const void* src, size_t n);
typedef void (*SimdReduceKernelFunc)(void* result, const void* src, size_t n);

// SIMD内核集合
typedef struct {
    // 二元运算
    SimdKernelFunc add_f32;
    SimdKernelFunc sub_f32;
    SimdKernelFunc mul_f32;
    SimdKernelFunc div_f32;
    SimdKernelFunc max_f32;
    SimdKernelFunc min_f32;
    
    // 一元运算
    SimdUnaryKernelFunc relu_f32;
    SimdUnaryKernelFunc sigmoid_f32;
    SimdUnaryKernelFunc tanh_f32;
    SimdUnaryKernelFunc exp_f32;
    SimdUnaryKernelFunc log_f32;
    SimdUnaryKernelFunc sqrt_f32;
    
    // 归约运算
    SimdReduceKernelFunc sum_f32;
    SimdReduceKernelFunc max_reduce_f32;
    SimdReduceKernelFunc min_reduce_f32;
    SimdReduceKernelFunc dot_product_f32;
    
    // 中文处理
    SimdUnaryKernelFunc chinese_normalize;
    SimdKernelFunc chinese_embedding;
} SimdKernels;

// 底层架构管理器
typedef struct {
    // 硬件信息
    HardwareInfo hw_info;
    
    // 内存池
    MemoryPool** memory_pools;
    int num_memory_pools;
    
    // 线程池
    ThreadPool** thread_pools;
    int num_thread_pools;
    
    // SIMD内核
    SimdKernels simd_kernels;
    SimdExecutionContext simd_context;
    
    // 性能监控
    bool performance_monitoring_enabled;
    PerformanceCounters* perf_history;
    size_t perf_history_size;
    size_t perf_history_capacity;
    
    // 中文优化
    bool chinese_optimization_enabled;
    void* chinese_text_cache;
    size_t chinese_cache_size;
    
    // 初始化状态
    bool initialized;
    pthread_mutex_t init_mutex;
    
} LowLevelArchitecture;

// 全局架构管理器
extern LowLevelArchitecture* g_ll_arch;

// 底层架构初始化与清理
bool ll_arch_initialize(const MemoryPoolConfig* mem_config, 
                       const ThreadPoolConfig* thread_config);
void ll_arch_cleanup(void);
bool ll_arch_is_initialized(void);

// 硬件信息获取
const HardwareInfo* ll_arch_get_hardware_info(void);
bool ll_arch_detect_cpu_features(void);
bool ll_arch_detect_cache_hierarchy(void);
bool ll_arch_detect_numa_topology(void);

// 内存池管理
MemoryPool* ll_arch_create_memory_pool(const MemoryPoolConfig* config);
void ll_arch_destroy_memory_pool(MemoryPool* pool);
void* ll_arch_memory_pool_alloc(MemoryPool* pool, size_t size);
void ll_arch_memory_pool_free(MemoryPool* pool, void* ptr);
void* ll_arch_memory_pool_realloc(MemoryPool* pool, void* ptr, size_t new_size);
MemoryPoolStats ll_arch_get_memory_pool_stats(const MemoryPool* pool);
void ll_arch_memory_pool_gc(MemoryPool* pool);

// 线程池管理
ThreadPool* ll_arch_create_thread_pool(const ThreadPoolConfig* config);
void ll_arch_destroy_thread_pool(ThreadPool* pool);
bool ll_arch_thread_pool_submit(ThreadPool* pool, void (*func)(void*), void* arg);
bool ll_arch_thread_pool_submit_priority(ThreadPool* pool, void (*func)(void*), void* arg, int priority);
void ll_arch_thread_pool_wait(ThreadPool* pool);
void ll_arch_thread_pool_wait_all(ThreadPool** pools, int num_pools);
size_t ll_arch_get_thread_pool_tasks_completed(const ThreadPool* pool);

// SIMD优化
bool ll_arch_simd_initialize(SimdExecutionContext* context);
void ll_arch_simd_cleanup(SimdExecutionContext* context);
void ll_arch_simd_select_kernels(SimdKernels* kernels, uint64_t cpu_features);

// 高性能内核
void ll_arch_vec_add_f32(float* dst, const float* src1, const float* src2, size_t n);
void ll_arch_vec_mul_f32(float* dst, const float* src1, const float* src2, size_t n);
void ll_arch_vec_relu_f32(float* dst, const float* src, size_t n);
void ll_arch_vec_sigmoid_f32(float* dst, const float* src, size_t n);
float ll_arch_vec_sum_f32(const float* src, size_t n);
float ll_arch_vec_dot_product_f32(const float* src1, const float* src2, size_t n);

// 矩阵运算优化
void ll_arch_matmul_f32(const float* A, const float* B, float* C, 
                         size_t M, size_t N, size_t K);
void ll_arch_matmul_blocked_f32(const float* A, const float* B, float* C,
                                size_t M, size_t N, size_t K, size_t block_size);
void ll_arch_conv2d_f32(const float* input, const float* kernel, float* output,
                       int batch, int in_channels, int out_channels,
                       int height, int width, int kernel_size,
                       int stride, int padding);

// 缓存优化
void ll_arch_cache_prefetch(const void* ptr, size_t size, int prefetch_type);
void ll_arch_cache_flush(const void* ptr, size_t size);
void ll_arch_cache_invalidate(const void* ptr, size_t size);
size_t ll_arch_get_cache_line_size(int cache_level);

// NUMA优化
bool ll_arch_numa_bind_memory(void* ptr, size_t size, int numa_node);
int ll_arch_numa_get_preferred_node(void);
void ll_arch_numa_set_preferred_node(int node);
size_t ll_arch_numa_get_available_memory(int node);

// 性能计数器
bool ll_arch_perf_start_counters(void);
bool ll_arch_perf_stop_counters(PerformanceCounters* counters);
bool ll_arch_perf_reset_counters(void);
void ll_arch_perf_print_counters(const PerformanceCounters* counters);
bool ll_arch_perf_enable_chinese_counters(bool enable);

// 中文优化
bool ll_arch_chinese_enable_optimization(bool enable);
void* ll_arch_chinese_alloc_buffer(size_t size);
void ll_arch_chinese_free_buffer(void* ptr);
bool ll_arch_chinese_prefetch_text(const char* text, size_t len);
size_t ll_arch_chinese_get_cache_hits(void);
size_t ll_arch_chinese_get_cache_misses(void);

// 错误处理
typedef enum {
    LL_ARCH_SUCCESS = 0,
    LL_ARCH_ERROR_NOT_INITIALIZED = -1,
    LL_ARCH_ERROR_INVALID_ARGUMENT = -2,
    LL_ARCH_ERROR_OUT_OF_MEMORY = -3,
    LL_ARCH_ERROR_THREAD_CREATION = -4,
    LL_ARCH_ERROR_SIMD_NOT_SUPPORTED = -5,
    LL_ARCH_ERROR_NUMA_NOT_AVAILABLE = -6,
    LL_ARCH_ERROR_PERFORMANCE_COUNTERS = -7,
    LL_ARCH_ERROR_CHINESE_OPTIMIZATION = -8
} LlArchError;

const char* ll_arch_error_string(LlArchError error);
LlArchError ll_arch_get_last_error(void);
void ll_arch_clear_error(void);

// 调试和诊断
void ll_arch_print_memory_layout(void);
void ll_arch_print_thread_info(void);
void ll_arch_print_cache_info(void);
void ll_arch_print_numa_topology(void);
bool ll_arch_save_profile(const char* filename);
bool ll_arch_load_profile(const char* filename);

#ifdef __cplusplus
}
#endif

#endif // AI_FRAMEWORK_LOW_LEVEL_ARCHITECTURE_H