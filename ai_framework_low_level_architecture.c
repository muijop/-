#include "ai_framework_low_level_architecture.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sched.h>
#include <numa.h>

#ifdef __x86_64__
#include <cpuid.h>
#include <x86intrin.h>
#endif

#ifdef __arm__
#include <arm_neon.h>
#endif

// 全局架构管理器
LowLevelArchitecture* g_ll_arch = NULL;
static LlArchError g_last_error = LL_ARCH_SUCCESS;

// 内部函数声明
static bool detect_cpu_vendor(char* vendor, size_t size);
static bool detect_cpu_brand(char* brand, size_t size);
static bool detect_cache_info(CacheHierarchy* cache);
static bool detect_numa_topology(NumaTopology* numa);
static void memory_pool_gc_internal(MemoryPool* pool);
static void* thread_pool_worker(void* arg);
static bool simd_kernel_init_avx512(SimdKernels* kernels);
static bool simd_kernel_init_avx2(SimdKernels* kernels);
static bool simd_kernel_init_neon(SimdKernels* kernels);

// 错误处理
const char* ll_arch_error_string(LlArchError error) {
    switch (error) {
        case LL_ARCH_SUCCESS: return "Success";
        case LL_ARCH_ERROR_NOT_INITIALIZED: return "Architecture not initialized";
        case LL_ARCH_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case LL_ARCH_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case LL_ARCH_ERROR_THREAD_CREATION: return "Thread creation failed";
        case LL_ARCH_ERROR_SIMD_NOT_SUPPORTED: return "SIMD not supported";
        case LL_ARCH_ERROR_NUMA_NOT_AVAILABLE: return "NUMA not available";
        case LL_ARCH_ERROR_PERFORMANCE_COUNTERS: return "Performance counters error";
        case LL_ARCH_ERROR_CHINESE_OPTIMIZATION: return "Chinese optimization error";
        default: return "Unknown error";
    }
}

LlArchError ll_arch_get_last_error(void) {
    return g_last_error;
}

void ll_arch_clear_error(void) {
    g_last_error = LL_ARCH_SUCCESS;
}

#define SET_ERROR(error) do { g_last_error = (error); } while(0)

// CPU特性检测
bool ll_arch_detect_cpu_features(void) {
    if (!g_ll_arch) {
        SET_ERROR(LL_ARCH_ERROR_NOT_INITIALIZED);
        return false;
    }
    
    HardwareInfo* hw = &g_ll_arch->hw_info;
    hw->features = 0;
    
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // 基础CPU信息
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        int max_leaf = eax;
        
        if (max_leaf >= 1) {
            __cpuid(1, eax, ebx, ecx, edx);
            
            // SSE系列
            if (edx & (1 << 25)) hw->features |= CPU_FEATURE_SSE;
            if (edx & (1 << 26)) hw->features |= CPU_FEATURE_SSE2;
            if (ecx & (1 << 0)) hw->features |= CPU_FEATURE_SSE3;
            if (ecx & (1 << 9)) hw->features |= CPU_FEATURE_SSSE3;
            if (ecx & (1 << 19)) hw->features |= CPU_FEATURE_SSE41;
            if (ecx & (1 << 20)) hw->features |= CPU_FEATURE_SSE42;
            if (ecx & (1 << 28)) hw->features |= CPU_FEATURE_AVX;
            if (ecx & (1 << 12)) hw->features |= CPU_FEATURE_FMA;
            
            // POPCNT
            if (ecx & (1 << 23)) hw->features |= CPU_FEATURE_POPCNT;
        }
        
        if (max_leaf >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            
            if (ebx & (1 << 5)) hw->features |= CPU_FEATURE_AVX2;
            if (ebx & (1 << 3)) hw->features |= CPU_FEATURE_BMI1;
            if (ebx & (1 << 8)) hw->features |= CPU_FEATURE_BMI2;
            if (ebx & (1 << 16)) hw->features |= CPU_FEATURE_AVX512F;
            if (ebx & (1 << 30)) hw->features |= CPU_FEATURE_AVX512BW;
            if (ebx & (1 << 17)) hw->features |= CPU_FEATURE_AVX512DQ;
            if (ebx & (1 << 31)) hw->features |= CPU_FEATURE_AVX512VL;
            
            // LZCNT
            if (ebx & (1 << 5)) hw->features |= CPU_FEATURE_LZCNT;
            // TZCNT
            if (ebx & (1 << 8)) hw->features |= CPU_FEATURE_TZCNT;
        }
    }
    
    // 获取CPU频率
    hw->base_frequency = 2400.0; // 默认值
    hw->max_frequency = 3600.0;  // 默认值
    
    // 检测CPU厂商和品牌
    detect_cpu_vendor(hw->vendor, sizeof(hw->vendor));
    detect_cpu_brand(hw->brand, sizeof(hw->brand));
    
#elif defined(__arm__)
    // ARM NEON检测
    hw->features |= CPU_FEATURE_NEON;
    
    // 检测ARM SVE (简化版)
    #ifdef __ARM_FEATURE_SVE
    hw->features |= CPU_FEATURE_SVE;
    #endif
    
    strcpy(hw->vendor, "ARM");
    strcpy(hw->brand, "ARM64 Processor");
    hw->base_frequency = 2000.0;
    hw->max_frequency = 2500.0;
    
#else
    strcpy(hw->vendor, "Unknown");
    strcpy(hw->brand, "Unknown Processor");
    hw->base_frequency = 2000.0;
    hw->max_frequency = 2000.0;
#endif
    
    // 中文优化特性
    hw->features |= CPU_FEATURE_CHINESE_STRING;
    hw->chinese_string_acceleration = true;
    hw->chinese_text_processing = true;
    
    return true;
}

static bool detect_cpu_vendor(char* vendor, size_t size) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        memcpy(vendor, &ebx, 4);
        memcpy(vendor + 4, &edx, 4);
        memcpy(vendor + 8, &ecx, 4);
        vendor[12] = '\0';
        return true;
    }
#endif
    snprintf(vendor, size, "Unknown");
    return false;
}

static bool detect_cpu_brand(char* brand, size_t size) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) >= 0x80000004) {
        for (int i = 0; i < 3; i++) {
            __cpuid(0x80000002 + i, eax, ebx, ecx, edx);
            memcpy(brand + i * 16, &eax, 4);
            memcpy(brand + i * 16 + 4, &ebx, 4);
            memcpy(brand + i * 16 + 8, &ecx, 4);
            memcpy(brand + i * 16 + 12, &edx, 4);
        }
        brand[48] = '\0';
        return true;
    }
#endif
    snprintf(brand, size, "Unknown Processor");
    return false;
}

// 缓存层次结构检测
bool ll_arch_detect_cache_hierarchy(void) {
    if (!g_ll_arch) {
        SET_ERROR(LL_ARCH_ERROR_NOT_INITIALIZED);
        return false;
    }
    
    CacheHierarchy* cache = &g_ll_arch->hw_info.cache;
    
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // 使用CPUID检测缓存信息
    if (__get_cpuid_max(0, NULL) >= 4) {
        int level = 0;
        while (1) {
            __cpuid_count(4, level, eax, ebx, ecx, edx);
            int cache_type = eax & 0x1F;
            if (cache_type == 0) break;
            
            int cache_level = (eax >> 5) & 0x7;
            int ways = ((ebx >> 22) & 0x3FF) + 1;
            int partitions = ((ebx >> 12) & 0x3FF) + 1;
            int line_size = (ebx & 0xFFF) + 1;
            int sets = ecx + 1;
            
            size_t cache_size = ways * partitions * line_size * sets;
            
            switch (cache_level) {
                case 1:
                    cache->l1_size = cache_size;
                    cache->l1_line_size = line_size;
                    cache->l1_associativity = ways;
                    break;
                case 2:
                    cache->l2_size = cache_size;
                    cache->l2_line_size = line_size;
                    cache->l2_associativity = ways;
                    break;
                case 3:
                    cache->l3_size = cache_size;
                    cache->l3_line_size = line_size;
                    cache->l3_associativity = ways;
                    break;
            }
            level++;
        }
    }
    
    // 默认值
    if (cache->l1_size == 0) cache->l1_size = 32768;
    if (cache->l2_size == 0) cache->l2_size = 262144;
    if (cache->l3_size == 0) cache->l3_size = 8388608;
    if (cache->l1_line_size == 0) cache->l1_line_size = 64;
    if (cache->l2_line_size == 0) cache->l2_line_size = 64;
    if (cache->l3_line_size == 0) cache->l3_line_size = 64;
    
#else
    // 非x86架构的默认值
    cache->l1_size = 32768;
    cache->l2_size = 262144;
    cache->l3_size = 8388608;
    cache->l1_line_size = 64;
    cache->l2_line_size = 64;
    cache->l3_line_size = 64;
    cache->l1_associativity = 8;
    cache->l2_associativity = 8;
    cache->l3_associativity = 16;
#endif
    
    // 预取参数
    cache->prefetch_distance = cache->l1_line_size * 4;
    cache->prefetch_degree = 2;
    
    return true;
}

// NUMA拓扑检测
bool ll_arch_detect_numa_topology(void) {
    if (!g_ll_arch) {
        SET_ERROR(LL_ARCH_ERROR_NOT_INITIALIZED);
        return false;
    }
    
    NumaTopology* numa = &g_ll_arch->hw_info.numa;
    
#ifdef __linux__
    if (numa_available() != -1) {
        numa->num_nodes = numa_max_node() + 1;
        numa->cores_per_node = (int*)calloc(numa->num_nodes, sizeof(int));
        numa->memory_per_node = (size_t*)calloc(numa->num_nodes, sizeof(size_t));
        numa->node_distance = (int**)calloc(numa->num_nodes, sizeof(int*));
        
        for (int i = 0; i < numa->num_nodes; i++) {
            numa->cores_per_node[i] = numa_num_configured_cpus() / numa->num_nodes;
            numa->memory_per_node[i] = numa_node_size(i, NULL);
            numa->node_distance[i] = (int*)calloc(numa->num_nodes, sizeof(int));
            
            for (int j = 0; j < numa->num_nodes; j++) {
                numa->node_distance[i][j] = numa_distance(i, j);
            }
        }
        
        numa->enable_numa_balancing = true;
        numa->preferred_node = numa_node_of_cpu(sched_getcpu());
        return true;
    }
#endif
    
    // 默认值
    numa->num_nodes = 1;
    numa->cores_per_node = (int*)calloc(1, sizeof(int));
    numa->memory_per_node = (size_t*)calloc(1, sizeof(size_t));
    numa->node_distance = (int**)calloc(1, sizeof(int*));
    numa->node_distance[0] = (int*)calloc(1, sizeof(int));
    
    numa->cores_per_node[0] = sysconf(_SC_NPROCESSORS_ONLN);
    numa->memory_per_node[0] = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
    numa->node_distance[0][0] = 10;
    numa->enable_numa_balancing = false;
    numa->preferred_node = 0;
    
    return true;
}

// 架构初始化
bool ll_arch_initialize(const MemoryPoolConfig* mem_config, const ThreadPoolConfig* thread_config) {
    if (g_ll_arch) {
        return true; // 已初始化
    }
    
    g_ll_arch = (LowLevelArchitecture*)calloc(1, sizeof(LowLevelArchitecture));
    if (!g_ll_arch) {
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return false;
    }
    
    pthread_mutex_init(&g_ll_arch->init_mutex, NULL);
    pthread_mutex_lock(&g_ll_arch->init_mutex);
    
    // 检测硬件信息
    if (!ll_arch_detect_cpu_features() ||
        !ll_arch_detect_cache_hierarchy() ||
        !ll_arch_detect_numa_topology()) {
        pthread_mutex_unlock(&g_ll_arch->init_mutex);
        pthread_mutex_destroy(&g_ll_arch->init_mutex);
        free(g_ll_arch);
        g_ll_arch = NULL;
        return false;
    }
    
    // 初始化内存池
    g_ll_arch->memory_pools = (MemoryPool**)calloc(8, sizeof(MemoryPool*));
    g_ll_arch->num_memory_pools = 0;
    
    // 创建默认内存池
    MemoryPoolConfig default_mem_config = {
        .pool_size = 1024 * 1024 * 1024, // 1GB
        .block_size = 65536, // 64KB
        .alignment = AI_CACHE_LINE_SIZE,
        .use_huge_pages = false,
        .enable_numa_binding = true,
        .numa_node = g_ll_arch->hw_info.numa.preferred_node,
        .max_cached_blocks = 1024,
        .gc_threshold = 0.8,
        .enable_chinese_optimization = true
    };
    
    if (mem_config) {
        default_mem_config = *mem_config;
    }
    
    MemoryPool* main_pool = ll_arch_create_memory_pool(&default_mem_config);
    if (main_pool) {
        g_ll_arch->memory_pools[g_ll_arch->num_memory_pools++] = main_pool;
    }
    
    // 初始化线程池
    g_ll_arch->thread_pools = (ThreadPool**)calloc(8, sizeof(ThreadPool*));
    g_ll_arch->num_thread_pools = 0;
    
    ThreadPoolConfig default_thread_config = {
        .num_threads = hw->physical_cores,
        .max_tasks = 10000,
        .bind_to_cores = true,
        .enable_work_stealing = true,
        .steal_threshold = 32,
        .task_queue_size = 1000,
        .enable_chinese_scheduling = true
    };
    
    if (thread_config) {
        default_thread_config = *thread_config;
    }
    
    ThreadPool* main_thread_pool = ll_arch_create_thread_pool(&default_thread_config);
    if (main_thread_pool) {
        g_ll_arch->thread_pools[g_ll_arch->num_thread_pools++] = main_thread_pool;
    }
    
    // 初始化SIMD内核
    ll_arch_simd_select_kernels(&g_ll_arch->simd_kernels, g_ll_arch->hw_info.features);
    ll_arch_simd_initialize(&g_ll_arch->simd_context);
    
    // 初始化性能监控
    g_ll_arch->performance_monitoring_enabled = true;
    g_ll_arch->perf_history_capacity = 10000;
    g_ll_arch->perf_history = (PerformanceCounters*)calloc(
        g_ll_arch->perf_history_capacity, sizeof(PerformanceCounters));
    g_ll_arch->perf_history_size = 0;
    
    // 中文优化
    g_ll_arch->chinese_optimization_enabled = true;
    g_ll_arch->chinese_cache_size = 64 * 1024 * 1024; // 64MB中文缓存
    g_ll_arch->chinese_text_cache = ll_arch_chinese_alloc_buffer(g_ll_arch->chinese_cache_size);
    
    g_ll_arch->initialized = true;
    pthread_mutex_unlock(&g_ll_arch->init_mutex);
    
    return true;
}

// 架构清理
void ll_arch_cleanup(void) {
    if (!g_ll_arch) {
        return;
    }
    
    pthread_mutex_lock(&g_ll_arch->init_mutex);
    
    // 清理内存池
    for (int i = 0; i < g_ll_arch->num_memory_pools; i++) {
        ll_arch_destroy_memory_pool(g_ll_arch->memory_pools[i]);
    }
    free(g_ll_arch->memory_pools);
    
    // 清理线程池
    for (int i = 0; i < g_ll_arch->num_thread_pools; i++) {
        ll_arch_destroy_thread_pool(g_ll_arch->thread_pools[i]);
    }
    free(g_ll_arch->thread_pools);
    
    // 清理SIMD上下文
    ll_arch_simd_cleanup(&g_ll_arch->simd_context);
    
    // 清理性能监控
    free(g_ll_arch->perf_history);
    
    // 清理中文缓存
    if (g_ll_arch->chinese_text_cache) {
        ll_arch_chinese_free_buffer(g_ll_arch->chinese_text_cache);
    }
    
    // 清理NUMA拓扑
    NumaTopology* numa = &g_ll_arch->hw_info.numa;
    free(numa->cores_per_node);
    free(numa->memory_per_node);
    for (int i = 0; i < numa->num_nodes; i++) {
        free(numa->node_distance[i]);
    }
    free(numa->node_distance);
    
    pthread_mutex_unlock(&g_ll_arch->init_mutex);
    pthread_mutex_destroy(&g_ll_arch->init_mutex);
    
    free(g_ll_arch);
    g_ll_arch = NULL;
}

bool ll_arch_is_initialized(void) {
    return g_ll_arch && g_ll_arch->initialized;
}

// 获取硬件信息
const HardwareInfo* ll_arch_get_hardware_info(void) {
    if (!g_ll_arch) {
        SET_ERROR(LL_ARCH_ERROR_NOT_INITIALIZED);
        return NULL;
    }
    return &g_ll_arch->hw_info;
}

// 内存池创建
MemoryPool* ll_arch_create_memory_pool(const MemoryPoolConfig* config) {
    if (!config) {
        SET_ERROR(LL_ARCH_ERROR_INVALID_ARGUMENT);
        return NULL;
    }
    
    MemoryPool* pool = (MemoryPool*)calloc(1, sizeof(MemoryPool));
    if (!pool) {
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    pool->config = *config;
    memset(&pool->stats, 0, sizeof(MemoryPoolStats));
    
    // 分配内存池
    size_t pool_size = config->pool_size;
    if (config->use_huge_pages) {
        pool_size = (pool_size + AI_PAGE_SIZE - 1) & ~(AI_PAGE_SIZE - 1);
        pool->config.pool_size = pool_size;
    }
    
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (config->use_huge_pages) {
        flags |= MAP_HUGETLB;
    }
    
    void* pool_memory = mmap(NULL, pool_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (pool_memory == MAP_FAILED) {
        free(pool);
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    // NUMA绑定
    if (config->enable_numa_binding && config->numa_node >= 0) {
        ll_arch_numa_bind_memory(pool_memory, pool_size, config->numa_node);
    }
    
    // 初始化空闲块列表
    pool->free_blocks = (void**)calloc(config->max_cached_blocks, sizeof(void*));
    pool->used_blocks = (void**)calloc(config->max_cached_blocks, sizeof(void*));
    
    if (!pool->free_blocks || !pool->used_blocks) {
        munmap(pool_memory, pool_size);
        free(pool->free_blocks);
        free(pool->used_blocks);
        free(pool);
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    // 分割内存池为块
    size_t num_blocks = pool_size / config->block_size;
    for (size_t i = 0; i < num_blocks && i < config->max_cached_blocks; i++) {
        pool->free_blocks[i] = (char*)pool_memory + i * config->block_size;
    }
    pool->num_free_blocks = (num_blocks < config->max_cached_blocks) ? num_blocks : config->max_cached_blocks;
    pool->num_used_blocks = 0;
    
    pthread_mutex_init(&pool->mutex, NULL);
    
    return pool;
}

// 内存池销毁
void ll_arch_destroy_memory_pool(MemoryPool* pool) {
    if (!pool) {
        return;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    // 释放内存池内存
    if (pool->num_free_blocks > 0) {
        void* pool_memory = pool->free_blocks[0];
        if (pool_memory) {
            munmap(pool_memory, pool->config.pool_size);
        }
    }
    
    free(pool->free_blocks);
    free(pool->used_blocks);
    
    pthread_mutex_unlock(&pool->mutex);
    pthread_mutex_destroy(&pool->mutex);
    
    free(pool);
}

// 内存池分配
void* ll_arch_memory_pool_alloc(MemoryPool* pool, size_t size) {
    if (!pool || size == 0) {
        SET_ERROR(LL_ARCH_ERROR_INVALID_ARGUMENT);
        return NULL;
    }
    
    if (size > pool->config.block_size) {
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    clock_t start_time = clock();
    
    // 垃圾回收检查
    if (pool->num_free_blocks == 0 && 
        (double)pool->stats.total_used / pool->config.pool_size > pool->config.gc_threshold) {
        memory_pool_gc_internal(pool);
    }
    
    if (pool->num_free_blocks == 0) {
        pthread_mutex_unlock(&pool->mutex);
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    // 分配块
    void* block = pool->free_blocks[--pool->num_free_blocks];
    pool->used_blocks[pool->num_used_blocks++] = block;
    
    // 更新统计
    pool->stats.allocation_count++;
    pool->stats.total_used += pool->config.block_size;
    pool->stats.cache_misses++;
    
    if (pool->stats.total_used > pool->stats.peak_usage) {
        pool->stats.peak_usage = pool->stats.total_used;
    }
    
    clock_t end_time = clock();
    double allocation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    pool->stats.average_allocation_time = 
        (pool->stats.average_allocation_time * (pool->allocation_count - 1) + allocation_time) / 
        pool->allocation_count;
    
    pthread_mutex_unlock(&pool->mutex);
    
    return block;
}

// 内存池释放
void ll_arch_memory_pool_free(MemoryPool* pool, void* ptr) {
    if (!pool || !ptr) {
        return;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    // 查找并移除使用块
    bool found = false;
    for (size_t i = 0; i < pool->num_used_blocks; i++) {
        if (pool->used_blocks[i] == ptr) {
            // 移除使用块
            pool->used_blocks[i] = pool->used_blocks[--pool->num_used_blocks];
            
            // 添加到空闲块
            if (pool->num_free_blocks < pool->config.max_cached_blocks) {
                pool->free_blocks[pool->num_free_blocks++] = ptr;
                pool->stats.cache_hits++;
            }
            
            pool->stats.deallocation_count++;
            pool->stats.total_used -= pool->config.block_size;
            found = true;
            break;
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
    
    if (!found) {
        SET_ERROR(LL_ARCH_ERROR_INVALID_ARGUMENT);
    }
}

// 内存池重新分配
void* ll_arch_memory_pool_realloc(MemoryPool* pool, void* ptr, size_t new_size) {
    if (!pool || new_size == 0) {
        return NULL;
    }
    
    if (new_size <= pool->config.block_size) {
        return ptr; // 当前块足够大
    }
    
    // 分配新块
    void* new_ptr = ll_arch_memory_pool_alloc(pool, new_size);
    if (!new_ptr) {
        return NULL;
    }
    
    // 复制数据
    if (ptr) {
        memcpy(new_ptr, ptr, pool->config.block_size);
        ll_arch_memory_pool_free(pool, ptr);
    }
    
    return new_ptr;
}

// 获取内存池统计
MemoryPoolStats ll_arch_get_memory_pool_stats(const MemoryPool* pool) {
    MemoryPoolStats stats = {0};
    if (pool) {
        pthread_mutex_lock((pthread_mutex_t*)&pool->mutex);
        stats = pool->stats;
        pthread_mutex_unlock((pthread_mutex_t*)&pool->mutex);
    }
    return stats;
}

// 内存池垃圾回收
void ll_arch_memory_pool_gc(MemoryPool* pool) {
    if (!pool) {
        return;
    }
    
    pthread_mutex_lock(&pool->mutex);
    memory_pool_gc_internal(pool);
    pthread_mutex_unlock(&pool->mutex);
}

static void memory_pool_gc_internal(MemoryPool* pool) {
    // 简单的垃圾回收：合并相邻空闲块
    // 这里可以实现更复杂的垃圾回收算法
    
    if (pool->gc_callback && pool->gc_context) {
        pool->gc_callback(pool->gc_context);
    }
    
    pool->stats.total_cached = pool->num_free_blocks * pool->config.block_size;
}

// 线程池创建
ThreadPool* ll_arch_create_thread_pool(const ThreadPoolConfig* config) {
    if (!config) {
        SET_ERROR(LL_ARCH_ERROR_INVALID_ARGUMENT);
        return NULL;
    }
    
    ThreadPool* pool = (ThreadPool*)calloc(1, sizeof(ThreadPool));
    if (!pool) {
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    pool->config = *config;
    pool->threads = (pthread_t*)calloc(config->num_threads, sizeof(pthread_t));
    pool->task_queue = (ThreadPoolTask*)calloc(config->task_queue_size, sizeof(ThreadPoolTask));
    
    if (!pool->threads || !pool->task_queue) {
        free(pool->threads);
        free(pool->task_queue);
        free(pool);
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    pthread_mutex_init(&pool->queue_mutex, NULL);
    pthread_cond_init(&pool->queue_cond, NULL);
    
    pool->queue_head = 0;
    pool->queue_tail = 0;
    pool->queue_size = 0;
    pool->shutdown = false;
    pool->tasks_completed = 0;
    pool->tasks_stolen = 0;
    pool->average_task_time = 0.0;
    
    // 创建工作线程
    for (int i = 0; i < config->num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, thread_pool_worker, pool) != 0) {
            // 清理已创建的线程
            pool->shutdown = true;
            pthread_cond_broadcast(&pool->queue_cond);
            for (int j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }
            
            pthread_mutex_destroy(&pool->queue_mutex);
            pthread_cond_destroy(&pool->queue_cond);
            free(pool->threads);
            free(pool->task_queue);
            free(pool);
            SET_ERROR(LL_ARCH_ERROR_THREAD_CREATION);
            return NULL;
        }
        
        // CPU绑定
        if (config->bind_to_cores) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i % sysconf(_SC_NPROCESSORS_ONLN), &cpuset);
            pthread_setaffinity_np(pool->threads[i], sizeof(cpu_set_t), &cpuset);
        }
    }
    
    return pool;
}

// 线程池工作函数
static void* thread_pool_worker(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    
    while (1) {
        pthread_mutex_lock(&pool->queue_mutex);
        
        // 等待任务
        while (pool->queue_size == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->queue_cond, &pool->queue_mutex);
        }
        
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->queue_mutex);
            break;
        }
        
        // 获取任务
        ThreadPoolTask* task = &pool->task_queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % pool->config.task_queue_size;
        pool->queue_size--;
        
        pthread_mutex_unlock(&pool->queue_mutex);
        
        // 执行任务
        clock_t start_time = clock();
        task->function(task->argument);
        clock_t end_time = clock();
        
        // 更新统计
        pthread_mutex_lock(&pool->queue_mutex);
        pool->tasks_completed++;
        double task_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        pool->average_task_time = 
            (pool->average_task_time * (pool->tasks_completed - 1) + task_time) / 
            pool->tasks_completed;
        pthread_mutex_unlock(&pool->queue_mutex);
        
        task->completed = true;
        pthread_cond_signal(&task->cond);
    }
    
    return NULL;
}

// 线程池销毁
void ll_arch_destroy_thread_pool(ThreadPool* pool) {
    if (!pool) {
        return;
    }
    
    pthread_mutex_lock(&pool->queue_mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->queue_cond);
    pthread_mutex_unlock(&pool->queue_mutex);
    
    // 等待所有线程完成
    for (int i = 0; i < pool->config.num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    pthread_mutex_destroy(&pool->queue_mutex);
    pthread_cond_destroy(&pool->queue_cond);
    
    free(pool->threads);
    free(pool->task_queue);
    free(pool);
}

// 提交任务到线程池
bool ll_arch_thread_pool_submit(ThreadPool* pool, void (*func)(void*), void* arg) {
    return ll_arch_thread_pool_submit_priority(pool, func, arg, 0);
}

bool ll_arch_thread_pool_submit_priority(ThreadPool* pool, void (*func)(void*), void* arg, int priority) {
    if (!pool || !func) {
        SET_ERROR(LL_ARCH_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    pthread_mutex_lock(&pool->queue_mutex);
    
    if (pool->queue_size >= pool->config.task_queue_size) {
        pthread_mutex_unlock(&pool->queue_mutex);
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return false;
    }
    
    // 添加任务到队列
    ThreadPoolTask* task = &pool->task_queue[(pool->queue_tail + pool->queue_size) % pool->config.task_queue_size];
    task->function = func;
    task->argument = arg;
    task->priority = priority;
    task->completed = false;
    pthread_mutex_init(&task->mutex, NULL);
    pthread_cond_init(&task->cond, NULL);
    
    pool->queue_size++;
    pthread_cond_signal(&pool->queue_cond);
    
    pthread_mutex_unlock(&pool->queue_mutex);
    
    return true;
}

// 等待线程池完成所有任务
void ll_arch_thread_pool_wait(ThreadPool* pool) {
    if (!pool) {
        return;
    }
    
    pthread_mutex_lock(&pool->queue_mutex);
    while (pool->queue_size > 0) {
        pthread_cond_wait(&pool->queue_cond, &pool->queue_mutex);
    }
    pthread_mutex_unlock(&pool->queue_mutex);
}

// 获取线程池完成的任务数
size_t ll_arch_get_thread_pool_tasks_completed(const ThreadPool* pool) {
    if (!pool) {
        return 0;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&pool->queue_mutex);
    size_t completed = pool->tasks_completed;
    pthread_mutex_unlock((pthread_mutex_t*)&pool->queue_mutex);
    
    return completed;
}

// SIMD内核选择
void ll_arch_simd_select_kernels(SimdKernels* kernels, uint64_t cpu_features) {
    if (!kernels) {
        return;
    }
    
    // 默认内核（标量实现）
    kernels->add_f32 = NULL;
    kernels->mul_f32 = NULL;
    kernels->relu_f32 = NULL;
    kernels->sigmoid_f32 = NULL;
    kernels->sum_f32 = NULL;
    kernels->dot_product_f32 = NULL;
    
#ifdef __x86_64__
    if (cpu_features & CPU_FEATURE_AVX512F) {
        simd_kernel_init_avx512(kernels);
    } else if (cpu_features & CPU_FEATURE_AVX2) {
        simd_kernel_init_avx2(kernels);
    }
#elif defined(__arm__)
    if (cpu_features & CPU_FEATURE_NEON) {
        simd_kernel_init_neon(kernels);
    }
#endif
}

// AVX-512内核初始化
static bool simd_kernel_init_avx512(SimdKernels* kernels) {
    // 这里将实现AVX-512优化的内核函数
    // 目前设置为NULL，后续实现具体的AVX-512内核
    return true;
}

// AVX2内核初始化
static bool simd_kernel_init_avx2(SimdKernels* kernels) {
    // 这里将实现AVX2优化的内核函数
    return true;
}

// NEON内核初始化
static bool simd_kernel_init_neon(SimdKernels* kernels) {
    // 这里将实现NEON优化的内核函数
    return true;
}

// SIMD执行上下文初始化
bool ll_arch_simd_initialize(SimdExecutionContext* context) {
    if (!context) {
        SET_ERROR(LL_ARCH_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    context->buffer_size = AI_VECTOR_WIDTH * 1024; // 32KB向量缓冲区
    context->aligned_buffer = aligned_alloc(AI_CACHE_LINE_SIZE, context->buffer_size);
    
    if (!context->aligned_buffer) {
        SET_ERROR(LL_ARCH_ERROR_OUT_OF_MEMORY);
        return false;
    }
    
    context->use_gather_scatter = true;
    context->use_masking = true;
    context->unroll_factor = 4;
    context->enable_fusion = true;
    
    memset(&context->vec_state, 0, sizeof(VectorRegisterState));
    
    return true;
}

// SIMD执行上下文清理
void ll_arch_simd_cleanup(SimdExecutionContext* context) {
    if (context && context->aligned_buffer) {
        free(context->aligned_buffer);
        context->aligned_buffer = NULL;
    }
}

// 高性能向量加法
void ll_arch_vec_add_f32(float* dst, const float* src1, const float* src2, size_t n) {
    if (!dst || !src1 || !src2 || n == 0) {
        return;
    }
    
#ifdef __x86_64__
    if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX512F)) {
        // AVX-512实现
        size_t i = 0;
        __m512 zero = _mm512_setzero_ps();
        
        for (; i + 15 < n; i += 16) {
            __m512 a = _mm512_loadu_ps(src1 + i);
            __m512 b = _mm512_loadu_ps(src2 + i);
            __m512 c = _mm512_add_ps(a, b);
            _mm512_storeu_ps(dst + i, c);
        }
        
        // 处理剩余元素
        for (; i < n; i++) {
            dst[i] = src1[i] + src2[i];
        }
    } else if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX2)) {
        // AVX2实现
        size_t i = 0;
        
        for (; i + 7 < n; i += 8) {
            __m256 a = _mm256_loadu_ps(src1 + i);
            __m256 b = _mm256_loadu_ps(src2 + i);
            __m256 c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(dst + i, c);
        }
        
        for (; i < n; i++) {
            dst[i] = src1[i] + src2[i];
        }
    } else
#endif
    {
        // 标量实现
        for (size_t i = 0; i < n; i++) {
            dst[i] = src1[i] + src2[i];
        }
    }
}

// 高性能向量乘法
void ll_arch_vec_mul_f32(float* dst, const float* src1, const float* src2, size_t n) {
    if (!dst || !src1 || !src2 || n == 0) {
        return;
    }
    
#ifdef __x86_64__
    if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX512F)) {
        size_t i = 0;
        
        for (; i + 15 < n; i += 16) {
            __m512 a = _mm512_loadu_ps(src1 + i);
            __m512 b = _mm512_loadu_ps(src2 + i);
            __m512 c = _mm512_mul_ps(a, b);
            _mm512_storeu_ps(dst + i, c);
        }
        
        for (; i < n; i++) {
            dst[i] = src1[i] * src2[i];
        }
    } else if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX2)) {
        size_t i = 0;
        
        for (; i + 7 < n; i += 8) {
            __m256 a = _mm256_loadu_ps(src1 + i);
            __m256 b = _mm256_loadu_ps(src2 + i);
            __m256 c = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(dst + i, c);
        }
        
        for (; i < n; i++) {
            dst[i] = src1[i] * src2[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src1[i] * src2[i];
        }
    }
}

// 高性能ReLU激活
void ll_arch_vec_relu_f32(float* dst, const float* src, size_t n) {
    if (!dst || !src || n == 0) {
        return;
    }
    
#ifdef __x86_64__
    if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX512F)) {
        size_t i = 0;
        __m512 zero = _mm512_setzero_ps();
        
        for (; i + 15 < n; i += 16) {
            __m512 s = _mm512_loadu_ps(src + i);
            __m512 r = _mm512_max_ps(s, zero);
            _mm512_storeu_ps(dst + i, r);
        }
        
        for (; i < n; i++) {
            dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
        }
    } else if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX2)) {
        size_t i = 0;
        __m256 zero = _mm256_setzero_ps();
        
        for (; i + 7 < n; i += 8) {
            __m256 s = _mm256_loadu_ps(src + i);
            __m256 r = _mm256_max_ps(s, zero);
            _mm256_storeu_ps(dst + i, r);
        }
        
        for (; i < n; i++) {
            dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
        }
    }
}

// 高性能Sigmoid激活
void ll_arch_vec_sigmoid_f32(float* dst, const float* src, size_t n) {
    if (!dst || !src || n == 0) {
        return;
    }
    
#ifdef __x86_64__
    if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX512F)) {
        size_t i = 0;
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 minus_one = _mm512_set1_ps(-1.0f);
        
        for (; i + 15 < n; i += 16) {
            __m512 s = _mm512_loadu_ps(src + i);
            
            // 使用近似计算或查找表优化
            // 这里使用标量计算作为示例
            for (int j = 0; j < 16; j++) {
                float x = src[i + j];
                dst[i + j] = 1.0f / (1.0f + expf(-x));
            }
        }
        
        for (; i < n; i++) {
            dst[i] = 1.0f / (1.0f + expf(-src[i]));
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            dst[i] = 1.0f / (1.0f + expf(-src[i]));
        }
    }
}

// 高性能向量求和
float ll_arch_vec_sum_f32(const float* src, size_t n) {
    if (!src || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    
#ifdef __x86_64__
    if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX512F)) {
        __m512 sum_vec = _mm512_setzero_ps();
        size_t i = 0;
        
        for (; i + 15 < n; i += 16) {
            __m512 s = _mm512_loadu_ps(src + i);
            sum_vec = _mm512_add_ps(sum_vec, s);
        }
        
        // 水平求和
        float temp[16];
        _mm512_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 16; j++) {
            sum += temp[j];
        }
        
        for (; i < n; i++) {
            sum += src[i];
        }
    } else if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX2)) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        
        for (; i + 7 < n; i += 8) {
            __m256 s = _mm256_loadu_ps(src + i);
            sum_vec = _mm256_add_ps(sum_vec, s);
        }
        
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 8; j++) {
            sum += temp[j];
        }
        
        for (; i < n; i++) {
            sum += src[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            sum += src[i];
        }
    }
    
    return sum;
}

// 高性能点积
float ll_arch_vec_dot_product_f32(const float* src1, const float* src2, size_t n) {
    if (!src1 || !src2 || n == 0) {
        return 0.0f;
    }
    
    float dot = 0.0f;
    
#ifdef __x86_64__
    if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX512F)) {
        __m512 dot_vec = _mm512_setzero_ps();
        size_t i = 0;
        
        for (; i + 15 < n; i += 16) {
            __m512 a = _mm512_loadu_ps(src1 + i);
            __m512 b = _mm512_loadu_ps(src2 + i);
            __m512 prod = _mm512_mul_ps(a, b);
            dot_vec = _mm512_add_ps(dot_vec, prod);
        }
        
        // 水平求和
        float temp[16];
        _mm512_storeu_ps(temp, dot_vec);
        for (int j = 0; j < 16; j++) {
            dot += temp[j];
        }
        
        for (; i < n; i++) {
            dot += src1[i] * src2[i];
        }
    } else if (g_ll_arch && (g_ll_arch->hw_info.features & CPU_FEATURE_AVX2)) {
        __m256 dot_vec = _mm256_setzero_ps();
        size_t i = 0;
        
        for (; i + 7 < n; i += 8) {
            __m256 a = _mm256_loadu_ps(src1 + i);
            __m256 b = _mm256_loadu_ps(src2 + i);
            __m256 prod = _mm256_mul_ps(a, b);
            dot_vec = _mm256_add_ps(dot_vec, prod);
        }
        
        float temp[8];
        _mm256_storeu_ps(temp, dot_vec);
        for (int j = 0; j < 8; j++) {
            dot += temp[j];
        }
        
        for (; i < n; i++) {
            dot += src1[i] * src2[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            dot += src1[i] * src2[i];
        }
    }
    
    return dot;
}

// 缓存优化函数
void ll_arch_cache_prefetch(const void* ptr, size_t size, int prefetch_type) {
    if (!ptr || size == 0) {
        return;
    }
    
    const char* p = (const char*)ptr;
    
#ifdef __x86_64__
    size_t prefetch_distance = g_ll_arch ? g_ll_arch->hw_info.cache.prefetch_distance : 256;
    
    for (size_t i = 0; i < size; i += AI_CACHE_LINE_SIZE) {
        if (prefetch_type == 0) { // T0: 预取到L1
            _mm_prefetch(p + i + prefetch_distance, _MM_HINT_T0);
        } else if (prefetch_type == 1) { // T1: 预取到L2
            _mm_prefetch(p + i + prefetch_distance, _MM_HINT_T1);
        } else { // T2: 预取到L3
            _mm_prefetch(p + i + prefetch_distance, _MM_HINT_T2);
        }
    }
#endif
}

void ll_arch_cache_flush(const void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }
    
#ifdef __x86_64__
    const char* p = (const char*)ptr;
    for (size_t i = 0; i < size; i += AI_CACHE_LINE_SIZE) {
        _mm_clflush(p + i);
    }
#endif
}

void ll_arch_cache_invalidate(const void* ptr, size_t size) {
    // 在x86上，缓存失效通常由硬件处理
    // 这里可以实现平台特定的缓存失效
    ll_arch_cache_flush(ptr, size);
}

size_t ll_arch_get_cache_line_size(int cache_level) {
    if (!g_ll_arch) {
        return AI_CACHE_LINE_SIZE;
    }
    
    const CacheHierarchy* cache = &g_ll_arch->hw_info.cache;
    
    switch (cache_level) {
        case 1: return cache->l1_line_size;
        case 2: return cache->l2_line_size;
        case 3: return cache->l3_line_size;
        default: return AI_CACHE_LINE_SIZE;
    }
}

// NUMA优化
bool ll_arch_numa_bind_memory(void* ptr, size_t size, int numa_node) {
#ifdef __linux__
    if (numa_available() == -1) {
        SET_ERROR(LL_ARCH_ERROR_NUMA_NOT_AVAILABLE);
        return false;
    }
    
    if (numa_node < 0 || numa_node >= numa_max_node() + 1) {
        SET_ERROR(LL_ARCH_ERROR_INVALID_ARGUMENT);
        return false;
    }
    
    return numa_tonode_memory(ptr, size, numa_node) == 0;
#else
    return false;
#endif
}

int ll_arch_numa_get_preferred_node(void) {
    if (!g_ll_arch) {
        return 0;
    }
    return g_ll_arch->hw_info.numa.preferred_node;
}

void ll_arch_numa_set_preferred_node(int node) {
    if (!g_ll_arch) {
        return;
    }
    
    if (node >= 0 && node < g_ll_arch->hw_info.numa.num_nodes) {
        g_ll_arch->hw_info.numa.preferred_node = node;
    }
}

size_t ll_arch_numa_get_available_memory(int node) {
#ifdef __linux__
    if (numa_available() == -1 || node < 0 || node >= numa_max_node() + 1) {
        return 0;
    }
    
    return numa_node_size(node, NULL);
#else
    return 0;
#endif
}

// 中文优化
bool ll_arch_chinese_enable_optimization(bool enable) {
    if (!g_ll_arch) {
        SET_ERROR(LL_ARCH_ERROR_NOT_INITIALIZED);
        return false;
    }
    
    g_ll_arch->chinese_optimization_enabled = enable;
    return true;
}

void* ll_arch_chinese_alloc_buffer(size_t size) {
    if (!g_ll_arch) {
        return NULL;
    }
    
    // 使用内存池分配
    if (g_ll_arch->num_memory_pools > 0) {
        return ll_arch_memory_pool_alloc(g_ll_arch->memory_pools[0], size);
    }
    
    return aligned_alloc(AI_CACHE_LINE_SIZE, size);
}

void ll_arch_chinese_free_buffer(void* ptr) {
    if (!g_ll_arch || !ptr) {
        return;
    }
    
    // 使用内存池释放
    if (g_ll_arch->num_memory_pools > 0) {
        ll_arch_memory_pool_free(g_ll_arch->memory_pools[0], ptr);
    } else {
        free(ptr);
    }
}

bool ll_arch_chinese_prefetch_text(const char* text, size_t len) {
    if (!text || len == 0) {
        return false;
    }
    
    // 预取中文文本到缓存
    ll_arch_cache_prefetch(text, len, 0);
    
    return true;
}

size_t ll_arch_chinese_get_cache_hits(void) {
    if (!g_ll_arch || g_ll_arch->num_memory_pools == 0) {
        return 0;
    }
    
    MemoryPoolStats stats = ll_arch_get_memory_pool_stats(g_ll_arch->memory_pools[0]);
    return stats.cache_hits;
}

size_t ll_arch_chinese_get_cache_misses(void) {
    if (!g_ll_arch || g_ll_arch->num_memory_pools == 0) {
        return 0;
    }
    
    MemoryPoolStats stats = ll_arch_get_memory_pool_stats(g_ll_arch->memory_pools[0]);
    return stats.cache_misses;
}

// 调试和诊断
void ll_arch_print_memory_layout(void) {
    if (!g_ll_arch) {
        printf("Architecture not initialized\n");
        return;
    }
    
    printf("=== Memory Layout ===\n");
    printf("Memory Pools: %d\n", g_ll_arch->num_memory_pools);
    
    for (int i = 0; i < g_ll_arch->num_memory_pools; i++) {
        MemoryPool* pool = g_ll_arch->memory_pools[i];
        MemoryPoolStats stats = ll_arch_get_memory_pool_stats(pool);
        
        printf("Pool %d:\n", i);
        printf("  Total Size: %zu bytes\n", pool->config.pool_size);
        printf("  Block Size: %zu bytes\n", pool->config.block_size);
        printf("  Used: %zu bytes (%.1f%%)\n", stats.total_used, 
               (double)stats.total_used / pool->config.pool_size * 100);
        printf("  Allocations: %zu\n", stats.allocation_count);
        printf("  Cache Hits: %zu\n", stats.cache_hits);
        printf("  Cache Misses: %zu\n", stats.cache_misses);
    }
}

void ll_arch_print_thread_info(void) {
    if (!g_ll_arch) {
        printf("Architecture not initialized\n");
        return;
    }
    
    printf("=== Thread Information ===\n");
    printf("Thread Pools: %d\n", g_ll_arch->num_thread_pools);
    
    for (int i = 0; i < g_ll_arch->num_thread_pools; i++) {
        ThreadPool* pool = g_ll_arch->thread_pools[i];
        size_t completed = ll_arch_get_thread_pool_tasks_completed(pool);
        
        printf("Pool %d:\n", i);
        printf("  Threads: %d\n", pool->config.num_threads);
        printf("  Tasks Completed: %zu\n", completed);
        printf("  Average Task Time: %.6f seconds\n", pool->average_task_time);
        printf("  Work Stealing: %s\n", pool->config.enable_work_stealing ? "Enabled" : "Disabled");
    }
}

void ll_arch_print_cache_info(void) {
    if (!g_ll_arch) {
        printf("Architecture not initialized\n");
        return;
    }
    
    const CacheHierarchy* cache = &g_ll_arch->hw_info.cache;
    
    printf("=== Cache Information ===\n");
    printf("L1 Cache: %zu KB, Line Size: %zu bytes, Associativity: %d\n",
           cache->l1_size / 1024, cache->l1_line_size, cache->l1_associativity);
    printf("L2 Cache: %zu KB, Line Size: %zu bytes, Associativity: %d\n",
           cache->l2_size / 1024, cache->l2_line_size, cache->l2_associativity);
    printf("L3 Cache: %zu KB, Line Size: %zu bytes, Associativity: %d\n",
           cache->l3_size / 1024, cache->l3_line_size, cache->l3_associativity);
    printf("Prefetch Distance: %zu bytes, Degree: %d\n",
           cache->prefetch_distance, cache->prefetch_degree);
}

void ll_arch_print_numa_topology(void) {
    if (!g_ll_arch) {
        printf("Architecture not initialized\n");
        return;
    }
    
    const NumaTopology* numa = &g_ll_arch->hw_info.numa;
    
    printf("=== NUMA Topology ===\n");
    printf("NUMA Nodes: %d\n", numa->num_nodes);
    printf("Preferred Node: %d\n", numa->preferred_node);
    printf("NUMA Balancing: %s\n", numa->enable_numa_balancing ? "Enabled" : "Disabled");
    
    for (int i = 0; i < numa->num_nodes; i++) {
        printf("Node %d: %d cores, %zu MB memory\n", 
               i, numa->cores_per_node[i], numa->memory_per_node[i] / (1024 * 1024));
    }
}

// 性能分析文件保存/加载
bool ll_arch_save_profile(const char* filename) {
    if (!g_ll_arch || !filename) {
        return false;
    }
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        return false;
    }
    
    // 保存硬件信息
    fwrite(&g_ll_arch->hw_info, sizeof(HardwareInfo), 1, fp);
    
    // 保存性能历史
    fwrite(&g_ll_arch->perf_history_size, sizeof(size_t), 1, fp);
    if (g_ll_arch->perf_history_size > 0) {
        fwrite(g_ll_arch->perf_history, sizeof(PerformanceCounters), 
               g_ll_arch->perf_history_size, fp);
    }
    
    fclose(fp);
    return true;
}

bool ll_arch_load_profile(const char* filename) {
    if (!g_ll_arch || !filename) {
        return false;
    }
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        return false;
    }
    
    // 读取硬件信息
    HardwareInfo hw_info;
    if (fread(&hw_info, sizeof(HardwareInfo), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    // 读取性能历史
    size_t history_size;
    if (fread(&history_size, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    if (history_size > 0 && history_size <= g_ll_arch->perf_history_capacity) {
        g_ll_arch->perf_history_size = history_size;
        fread(g_ll_arch->perf_history, sizeof(PerformanceCounters), history_size, fp);
    }
    
    fclose(fp);
    return true;
}