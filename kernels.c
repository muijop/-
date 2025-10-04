#include "kernels.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef __x86_64__
#include <immintrin.h>
#include <cpuid.h>
#endif

#ifdef __arm__
#include <arm_neon.h>
#endif

// 全局内核配置
static KernelConfig g_kernel_config = {
    .simd_level = SIMD_NONE,
    .num_threads = 1,
    .cache_size = 32768,  // L1 cache size in bytes
    .use_mkl = false,
    .use_openblas = false,
    .use_cuda = false
};

// 性能统计
static KernelStats g_kernel_stats = {0};
static clock_t g_start_time = 0;

// SIMD指令集检测
SIMDLevel kernel_detect_simd(void) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    SIMDLevel level = SIMD_NONE;
    
    // 检查CPU特性
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (edx & (1 << 25)) level |= SIMD_SSE;
        if (edx & (1 << 26)) level |= SIMD_SSE2;
        if (ecx & (1 << 0)) level |= SIMD_SSE3;
        if (ecx & (1 << 9)) level |= SIMD_SSSE3;
        if (ecx & (1 << 19)) level |= SIMD_SSE41;
        if (ecx & (1 << 20)) level |= SIMD_SSE42;
        if (ecx & (1 << 28)) level |= SIMD_AVX;
    }
    
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        if (ebx & (1 << 5)) level |= SIMD_AVX2;
        if (ebx & (1 << 16)) level |= SIMD_AVX512F;
        if (ebx & (1 << 30)) level |= SIMD_AVX512BW;
    }
    
    return level;
#elif defined(__arm__)
    return SIMD_NEON;
#else
    return SIMD_NONE;
#endif
}

// 获取SIMD级别名称
const char* kernel_simd_level_name(SIMDLevel level) {
    switch (level) {
        case SIMD_NONE: return "NONE";
        case SIMD_SSE: return "SSE";
        case SIMD_SSE2: return "SSE2";
        case SIMD_SSE3: return "SSE3";
        case SIMD_SSSE3: return "SSSE3";
        case SIMD_SSE41: return "SSE4.1";
        case SIMD_SSE42: return "SSE4.2";
        case SIMD_AVX: return "AVX";
        case SIMD_AVX2: return "AVX2";
        case SIMD_AVX512F: return "AVX512F";
        case SIMD_AVX512BW: return "AVX512BW";
        case SIMD_NEON: return "NEON";
        default: return "UNKNOWN";
    }
}

// 初始化内核
void kernel_init(KernelConfig* config) {
    if (config) {
        g_kernel_config = *config;
    } else {
        g_kernel_config.simd_level = kernel_detect_simd();
        g_kernel_config.num_threads = 1;  // 默认单线程
        g_kernel_config.cache_size = 32768;
        g_kernel_config.use_mkl = false;
        g_kernel_config.use_openblas = false;
        g_kernel_config.use_cuda = false;
    }
    
    printf("Kernel initialized with SIMD level: %s\n", 
           kernel_simd_level_name(g_kernel_config.simd_level));
}

// 清理内核
void kernel_cleanup(void) {
    // 清理资源
    memset(&g_kernel_stats, 0, sizeof(KernelStats));
}

// 对齐内存分配
void* kernel_aligned_alloc(size_t size, size_t alignment) {
    void* ptr = NULL;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
#endif
    return ptr;
}

// 对齐内存释放
void kernel_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// 内存预取
void kernel_prefetch(const void* ptr, size_t size) {
#ifdef __x86_64__
    // x86-64预取指令
    const char* p = (const char*)ptr;
    for (size_t i = 0; i < size; i += 64) {  // 64字节缓存行
        __builtin_prefetch(p + i, 0, 3);
    }
#endif
}

// 浮点数组设置
void kernel_memset_float(float* dst, float value, size_t n) {
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 val = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            _mm256_storeu_ps(dst + i, val);
        }
        for (; i < n; i++) {
            dst[i] = value;
        }
    } else if (g_kernel_config.simd_level & SIMD_SSE) {
        __m128 val = _mm_set1_ps(value);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            _mm_storeu_ps(dst + i, val);
        }
        for (; i < n; i++) {
            dst[i] = value;
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            dst[i] = value;
        }
    }
}

// 浮点数组拷贝
void kernel_memcpy_float(float* dst, const float* src, size_t n) {
    memcpy(dst, src, n * sizeof(float));
}

// SAXPY操作: y = alpha * x + y
void kernel_saxpy(size_t n, float alpha, const float* x, float* y) {
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 alpha_vec = _mm256_set1_ps(alpha);
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 y_vec = _mm256_loadu_ps(y + i);
            __m256 result = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x_vec), y_vec);
            _mm256_storeu_ps(y + i, result);
        }
        for (; i < n; i++) {
            y[i] = alpha * x[i] + y[i];
        }
    } else if (g_kernel_config.simd_level & SIMD_SSE) {
        __m128 alpha_vec = _mm_set1_ps(alpha);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            __m128 y_vec = _mm_loadu_ps(y + i);
            __m128 result = _mm_add_ps(_mm_mul_ps(alpha_vec, x_vec), y_vec);
            _mm_storeu_ps(y + i, result);
        }
        for (; i < n; i++) {
            y[i] = alpha * x[i] + y[i];
        }
    } else
#elif defined(__arm__)
    if (g_kernel_config.simd_level & SIMD_NEON) {
        float32x4_t alpha_vec = vdupq_n_f32(alpha);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t y_vec = vld1q_f32(y + i);
            float32x4_t result = vaddq_f32(vmulq_f32(alpha_vec, x_vec), y_vec);
            vst1q_f32(y + i, result);
        }
        for (; i < n; i++) {
            y[i] = alpha * x[i] + y[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            y[i] = alpha * x[i] + y[i];
        }
    }
}

// SSCAL操作: x = alpha * x
void kernel_sscal(size_t n, float alpha, float* x) {
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 alpha_vec = _mm256_set1_ps(alpha);
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 result = _mm256_mul_ps(alpha_vec, x_vec);
            _mm256_storeu_ps(x + i, result);
        }
        for (; i < n; i++) {
            x[i] = alpha * x[i];
        }
    } else if (g_kernel_config.siml_level & SIMD_SSE) {
        __m128 alpha_vec = _mm_set1_ps(alpha);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            __m128 result = _mm_mul_ps(alpha_vec, x_vec);
            _mm_storeu_ps(x + i, result);
        }
        for (; i < n; i++) {
            x[i] = alpha * x[i];
        }
    } else
#elif defined(__arm__)
    if (g_kernel_config.simd_level & SIMD_NEON) {
        float32x4_t alpha_vec = vdupq_n_f32(alpha);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t result = vmulq_f32(alpha_vec, x_vec);
            vst1q_f32(x + i, result);
        }
        for (; i < n; i++) {
            x[i] = alpha * x[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            x[i] = alpha * x[i];
        }
    }
}

// 点积操作
float kernel_sdot(size_t n, const float* x, const float* y) {
    float result = 0.0f;
    
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 y_vec = _mm256_loadu_ps(y + i);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(x_vec, y_vec));
        }
        
        // 水平求和
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        for (int j = 0; j < 8; j++) {
            result += temp[j];
        }
        
        for (; i < n; i++) {
            result += x[i] * y[i];
        }
    } else if (g_kernel_config.simd_level & SIMD_SSE) {
        __m128 sum = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            __m128 y_vec = _mm_loadu_ps(y + i);
            sum = _mm_add_ps(sum, _mm_mul_ps(x_vec, y_vec));
        }
        
        // 水平求和
        float temp[4];
        _mm_storeu_ps(temp, sum);
        for (int j = 0; j < 4; j++) {
            result += temp[j];
        }
        
        for (; i < n; i++) {
            result += x[i] * y[i];
        }
    } else
#elif defined(__arm__)
    if (g_kernel_config.simd_level & SIMD_NEON) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t y_vec = vld1q_f32(y + i);
            sum = vaddq_f32(sum, vmulq_f32(x_vec, y_vec));
        }
        
        // 水平求和
        float temp[4];
        vst1q_f32(temp, sum);
        for (int j = 0; j < 4; j++) {
            result += temp[j];
        }
        
        for (; i < n; i++) {
            result += x[i] * y[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            result += x[i] * y[i];
        }
    }
    
    return result;
}

// L2范数
float kernel_snrm2(size_t n, const float* x) {
    float sum = 0.0f;
    
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_vec, x_vec));
        }
        
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 8; j++) {
            sum += temp[j];
        }
        
        for (; i < n; i++) {
            sum += x[i] * x[i];
        }
    } else if (g_kernel_config.simd_level & SIMD_SSE) {
        __m128 sum_vec = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(x_vec, x_vec));
        }
        
        float temp[4];
        _mm_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 4; j++) {
            sum += temp[j];
        }
        
        for (; i < n; i++) {
            sum += x[i] * x[i];
        }
    } else
#elif defined(__arm__)
    if (g_kernel_config.simd_level & SIMD_NEON) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            sum_vec = vaddq_f32(sum_vec, vmulq_f32(x_vec, x_vec));
        }
        
        float temp[4];
        vst1q_f32(temp, sum_vec);
        for (int j = 0; j < 4; j++) {
            sum += temp[j];
        }
        
        for (; i < n; i++) {
            sum += x[i] * x[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            sum += x[i] * x[i];
        }
    }
    
    return sqrtf(sum);
}

// ReLU激活函数
void kernel_relu(size_t n, const float* x, float* y) {
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 zero = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 result = _mm256_max_ps(x_vec, zero);
            _mm256_storeu_ps(y + i, result);
        }
        for (; i < n; i++) {
            y[i] = x[i] > 0 ? x[i] : 0;
        }
    } else if (g_kernel_config.simd_level & SIMD_SSE) {
        __m128 zero = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            __m128 result = _mm_max_ps(x_vec, zero);
            _mm_storeu_ps(y + i, result);
        }
        for (; i < n; i++) {
            y[i] = x[i] > 0 ? x[i] : 0;
        }
    } else
#elif defined(__arm__)
    if (g_kernel_config.simd_level & SIMD_NEON) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t result = vmaxq_f32(x_vec, zero);
            vst1q_f32(y + i, result);
        }
        for (; i < n; i++) {
            y[i] = x[i] > 0 ? x[i] : 0;
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            y[i] = x[i] > 0 ? x[i] : 0;
        }
    }
}

// ReLU反向传播
void kernel_relu_backward(size_t n, const float* x, const float* grad_y, float* grad_x) {
    for (size_t i = 0; i < n; i++) {
        grad_x[i] = x[i] > 0 ? grad_y[i] : 0;
    }
}

// Sigmoid激活函数 - 使用数值稳定的实现和SIMD优化
void kernel_sigmoid(size_t n, const float* x, float* y) {
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 minus_one = _mm256_set1_ps(-1.0f);
        __m256 clamp_max = _mm256_set1_ps(10.0f);  // 避免exp溢出
        __m256 clamp_min = _mm256_set1_ps(-10.0f);
        
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            
            // 数值稳定: clamp x to [-10, 10]
            x_vec = _mm256_max_ps(x_vec, clamp_min);
            x_vec = _mm256_min_ps(x_vec, clamp_max);
            
            // 计算sigmoid: 1 / (1 + exp(-x))
            __m256 neg_x = _mm256_sub_ps(zero, x_vec);
            __m256 exp_neg_x = _mm256_exp_ps(neg_x);
            __m256 denom = _mm256_add_ps(one, exp_neg_x);
            __m256 result = _mm256_div_ps(one, denom);
            
            _mm256_storeu_ps(y + i, result);
        }
        
        // 处理剩余元素
        for (; i < n; i++) {
            float xi = x[i];
            if (xi > 10.0f) xi = 10.0f;
            if (xi < -10.0f) xi = -10.0f;
            y[i] = 1.0f / (1.0f + expf(-xi));
        }
    } else if (g_kernel_config.simd_level & SIMD_SSE) {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 clamp_max = _mm_set1_ps(10.0f);
        __m128 clamp_min = _mm_set1_ps(-10.0f);
        
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            
            // 数值稳定
            x_vec = _mm_max_ps(x_vec, clamp_min);
            x_vec = _mm_min_ps(x_vec, clamp_max);
            
            // 计算sigmoid
            __m128 neg_x = _mm_sub_ps(zero, x_vec);
            __m128 exp_neg_x = _mm_exp_ps(neg_x);
            __m128 denom = _mm_add_ps(one, exp_neg_x);
            __m128 result = _mm_div_ps(one, denom);
            
            _mm_storeu_ps(y + i, result);
        }
        
        for (; i < n; i++) {
            float xi = x[i];
            if (xi > 10.0f) xi = 10.0f;
            if (xi < -10.0f) xi = -10.0f;
            y[i] = 1.0f / (1.0f + expf(-xi));
        }
    } else
#elif defined(__arm__)
    if (g_kernel_config.simd_level & SIMD_NEON) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t clamp_max = vdupq_n_f32(10.0f);
        float32x4_t clamp_min = vdupq_n_f32(-10.0f);
        
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            
            // 数值稳定
            x_vec = vmaxq_f32(x_vec, clamp_min);
            x_vec = vminq_f32(x_vec, clamp_max);
            
            // 计算sigmoid
            float32x4_t neg_x = vnegq_f32(x_vec);
            float32x4_t exp_neg_x = exp_ps_neon(neg_x);
            float32x4_t denom = vaddq_f32(one, exp_neg_x);
            float32x4_t result = vdivq_f32(one, denom);
            
            vst1q_f32(y + i, result);
        }
        
        for (; i < n; i++) {
            float xi = x[i];
            if (xi > 10.0f) xi = 10.0f;
            if (xi < -10.0f) xi = -10.0f;
            y[i] = 1.0f / (1.0f + expf(-xi));
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            float xi = x[i];
            if (xi > 10.0f) xi = 10.0f;
            if (xi < -10.0f) xi = -10.0f;
            y[i] = 1.0f / (1.0f + expf(-xi));
        }
    }
}

// Sigmoid反向传播 - 使用缓存的sigmoid值提高数值稳定性
void kernel_sigmoid_backward(size_t n, const float* x, const float* grad_y, float* grad_x) {
#ifdef __x86_64__
    if (g_kernel_config.simd_level & SIMD_AVX) {
        __m256 one = _mm256_set1_ps(1.0f);
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 grad_y_vec = _mm256_loadu_ps(grad_y + i);
            
            // 计算sigmoid值
            __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x_vec))));
            
            // 计算梯度: grad_y * sigmoid * (1 - sigmoid)
            __m256 one_minus_sigmoid = _mm256_sub_ps(one, sigmoid);
            __m256 grad = _mm256_mul_ps(grad_y_vec, _mm256_mul_ps(sigmoid, one_minus_sigmoid));
            
            _mm256_storeu_ps(grad_x + i, grad);
        }
        
        for (; i < n; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            grad_x[i] = grad_y[i] * sigmoid * (1.0f - sigmoid);
        }
    } else if (g_kernel_config.simd_level & SIMD_SSE) {
        __m128 one = _mm_set1_ps(1.0f);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            __m128 grad_y_vec = _mm_loadu_ps(grad_y + i);
            
            __m128 sigmoid = _mm_div_ps(one, _mm_add_ps(one, _mm_exp_ps(_mm_sub_ps(_mm_setzero_ps(), x_vec))));
            __m128 one_minus_sigmoid = _mm_sub_ps(one, sigmoid);
            __m128 grad = _mm_mul_ps(grad_y_vec, _mm_mul_ps(sigmoid, one_minus_sigmoid));
            
            _mm_storeu_ps(grad_x + i, grad);
        }
        
        for (; i < n; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            grad_x[i] = grad_y[i] * sigmoid * (1.0f - sigmoid);
        }
    } else
#elif defined(__arm__)
    if (g_kernel_config.simd_level & SIMD_NEON) {
        float32x4_t one = vdupq_n_f32(1.0f);
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t grad_y_vec = vld1q_f32(grad_y + i);
            
            float32x4_t sigmoid = vdivq_f32(one, vaddq_f32(one, exp_ps_neon(vnegq_f32(x_vec))));
            float32x4_t one_minus_sigmoid = vsubq_f32(one, sigmoid);
            float32x4_t grad = vmulq_f32(grad_y_vec, vmulq_f32(sigmoid, one_minus_sigmoid));
            
            vst1q_f32(grad_x + i, grad);
        }
        
        for (; i < n; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            grad_x[i] = grad_y[i] * sigmoid * (1.0f - sigmoid);
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            grad_x[i] = grad_y[i] * sigmoid * (1.0f - sigmoid);
        }
    }
}

// Tanh激活函数
void kernel_tanh(size_t n, const float* x, float* y) {
    for (size_t i = 0; i < n; i++) {
        y[i] = tanhf(x[i]);
    }
}

// Tanh反向传播
void kernel_tanh_backward(size_t n, const float* x, const float* grad_y, float* grad_x) {
    for (size_t i = 0; i < n; i++) {
        float tanh_val = tanhf(x[i]);
        grad_x[i] = grad_y[i] * (1.0f - tanh_val * tanh_val);
    }
}

// 简单的矩阵乘法（基础版本）
void kernel_sgemm_simple(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

// 优化的矩阵乘法
void kernel_sgemm(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C) {
    // 简单的分块优化
    const size_t block_size = 64;
    
    for (size_t i0 = 0; i0 < m; i0 += block_size) {
        for (size_t j0 = 0; j0 < n; j0 += block_size) {
            for (size_t k0 = 0; k0 < k; k0 += block_size) {
                size_t i_max = (i0 + block_size < m) ? i0 + block_size : m;
                size_t j_max = (j0 + block_size < n) ? j0 + block_size : n;
                size_t k_max = (k0 + block_size < k) ? k0 + block_size : k;
                
                for (size_t i = i0; i < i_max; i++) {
                    for (size_t j = j0; j < j_max; j++) {
                        float sum = 0.0f;
                        for (size_t l = k0; l < k_max; l++) {
                            sum += A[i * k + l] * B[l * n + j];
                        }
                        if (k0 == 0) {
                            C[i * n + j] = alpha * sum + beta * C[i * n + j];
                        } else {
                            C[i * n + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }
}

// 分块矩阵乘法
void kernel_sgemm_blocked(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C, size_t block_size) {
    for (size_t i0 = 0; i0 < m; i0 += block_size) {
        for (size_t j0 = 0; j0 < n; j0 += block_size) {
            for (size_t k0 = 0; k0 < k; k0 += block_size) {
                size_t i_max = (i0 + block_size < m) ? i0 + block_size : m;
                size_t j_max = (j0 + block_size < n) ? j0 + block_size : n;
                size_t k_max = (k0 + block_size < k) ? k0 + block_size : k;
                
                for (size_t i = i0; i < i_max; i++) {
                    for (size_t j = j0; j < j_max; j++) {
                        float sum = 0.0f;
                        for (size_t l = k0; l < k_max; l++) {
                            sum += A[i * k + l] * B[l * n + j];
                        }
                        if (k0 == 0) {
                            C[i * n + j] = alpha * sum + beta * C[i * n + j];
                        } else {
                            C[i * n + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }
}

// SGD优化器步骤
void kernel_sgd_step(size_t n, float* params, const float* grads, float learning_rate) {
    for (size_t i = 0; i < n; i++) {
        params[i] -= learning_rate * grads[i];
    }
}

// Momentum优化器步骤
void kernel_momentum_step(size_t n, float* params, const float* grads, float* velocities, float learning_rate, float momentum) {
    for (size_t i = 0; i < n; i++) {
        velocities[i] = momentum * velocities[i] + (1.0f - momentum) * grads[i];
        params[i] -= learning_rate * velocities[i];
    }
}

// Adam优化器步骤
void kernel_adam_step(size_t n, float* params, const float* grads, float* m, float* v, float learning_rate, float beta1, float beta2, float eps, size_t step) {
    float bias_correction1 = 1.0f - powf(beta1, step);
    float bias_correction2 = 1.0f - powf(beta2, step);
    
    for (size_t i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
        
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        
        params[i] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

// RMSprop优化器步骤
void kernel_rmsprop_step(size_t n, float* params, const float* grads, float* cache, float learning_rate, float decay_rate, float eps) {
    for (size_t i = 0; i < n; i++) {
        cache[i] = decay_rate * cache[i] + (1.0f - decay_rate) * grads[i] * grads[i];
        params[i] -= learning_rate * grads[i] / (sqrtf(cache[i]) + eps);
    }
}

// 性能分析
void kernel_start_profiling(void) {
    g_start_time = clock();
    memset(&g_kernel_stats, 0, sizeof(KernelStats));
}

void kernel_stop_profiling(void) {
    clock_t end_time = clock();
    g_kernel_stats.execution_time_ms = (double)(end_time - g_start_time) * 1000.0 / CLOCKS_PER_SEC;
}

KernelStats kernel_get_stats(void) {
    return g_kernel_stats;
}

void kernel_print_stats(void) {
    printf("Kernel Performance Stats:\n");
    printf("  GFLOPS: %.2f\n", g_kernel_stats.gflops);
    printf("  Bandwidth: %.2f GB/s\n", g_kernel_stats.bandwidth_gb_s);
    printf("  Operations: %zu\n", g_kernel_stats.num_operations);
    printf("  Execution Time: %.2f ms\n", g_kernel_stats.execution_time_ms);
}