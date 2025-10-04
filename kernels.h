#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include <stdint.h>

// SIMD指令集检测
typedef enum {
    SIMD_NONE = 0,
    SIMD_SSE = 1 << 0,
    SIMD_SSE2 = 1 << 1,
    SIMD_SSE3 = 1 << 2,
    SIMD_SSSE3 = 1 << 3,
    SIMD_SSE41 = 1 << 4,
    SIMD_SSE42 = 1 << 5,
    SIMD_AVX = 1 << 6,
    SIMD_AVX2 = 1 << 7,
    SIMD_AVX512F = 1 << 8,
    SIMD_AVX512BW = 1 << 9,
    SIMD_NEON = 1 << 10  // ARM NEON
} SIMDLevel;

// 内核配置
typedef struct {
    SIMDLevel simd_level;
    int num_threads;
    size_t cache_size;
    bool use_mkl;
    bool use_openblas;
    bool use_cuda;
} KernelConfig;

// 性能统计
typedef struct {
    double gflops;
    double bandwidth_gb_s;
    size_t num_operations;
    double execution_time_ms;
} KernelStats;

// 基础计算内核
void kernel_init(KernelConfig* config);
void kernel_cleanup(void);
SIMDLevel kernel_detect_simd(void);
const char* kernel_simd_level_name(SIMDLevel level);

// BLAS级别1操作（向量-向量）
void kernel_saxpy(size_t n, float alpha, const float* x, float* y);
void kernel_sscal(size_t n, float alpha, float* x);
void kernel_scopy(size_t n, const float* x, float* y);
float kernel_sdot(size_t n, const float* x, const float* y);
float kernel_snrm2(size_t n, const float* x);

// BLAS级别2操作（矩阵-向量）
void kernel_sgemv(size_t m, size_t n, float alpha, const float* A, const float* x, float beta, float* y);
void kernel_sger(size_t m, size_t n, float alpha, const float* x, const float* y, float* A);

// BLAS级别3操作（矩阵-矩阵）
void kernel_sgemm(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
void kernel_sgemm_blocked(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C, size_t block_size);

// 激活函数内核
void kernel_relu(size_t n, const float* x, float* y);
void kernel_relu_backward(size_t n, const float* x, const float* grad_y, float* grad_x);
void kernel_sigmoid(size_t n, const float* x, float* y);
void kernel_sigmoid_backward(size_t n, const float* x, const float* grad_y, float* grad_x);
void kernel_tanh(size_t n, const float* x, float* y);
void kernel_tanh_backward(size_t n, const float* x, const float* grad_y, float* grad_x);
void kernel_softmax(size_t n, const float* x, float* y);
void kernel_softmax_backward(size_t n, const float* x, const float* grad_y, float* grad_x);

// 归一化内核
void kernel_layernorm(size_t n, const float* x, float* y, float* mean, float* var, float epsilon);
void kernel_layernorm_backward(size_t n, const float* x, const float* grad_y, float* grad_x, const float* mean, const float* var, float epsilon);
void kernel_batchnorm(size_t n, const float* x, float* y, float* running_mean, float* running_var, float* gamma, float* beta, float epsilon, bool training);

// 卷积内核
void kernel_conv2d_forward(int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size, int stride, int padding, const float* input, const float* weight, const float* bias, float* output);
void kernel_conv2d_backward(int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size, int stride, int padding, const float* input, const float* weight, const float* grad_output, float* grad_input, float* grad_weight, float* grad_bias);

// 池化内核
void kernel_maxpool2d_forward(int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding, const float* input, float* output, int* indices);
void kernel_maxpool2d_backward(int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding, const float* grad_output, const int* indices, float* grad_input);
void kernel_avgpool2d_forward(int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding, const float* input, float* output);
void kernel_avgpool2d_backward(int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding, const float* grad_output, float* grad_input);

// 注意力机制内核
void kernel_scaled_dot_product_attention(size_t seq_len, size_t head_dim, const float* Q, const float* K, const float* V, float* output, float* attention_weights);
void kernel_multihead_attention_forward(int batch_size, int seq_len, int num_heads, int head_dim, const float* Q, const float* K, const float* V, const float* W_Q, const float* W_K, const float* W_V, const float* W_O, float* output);
void kernel_multihead_attention_backward(int batch_size, int seq_len, int num_heads, int head_dim, const float* Q, const float* K, const float* V, const float* W_Q, const float* W_K, const float* W_V, const float* W_O, const float* grad_output, float* grad_Q, float* grad_K, float* grad_V, float* grad_W_Q, float* grad_W_K, float* grad_W_V, float* grad_W_O);

// 优化器内核
void kernel_sgd_step(size_t n, float* params, const float* grads, float learning_rate);
void kernel_momentum_step(size_t n, float* params, const float* grads, float* velocities, float learning_rate, float momentum);
void kernel_adam_step(size_t n, float* params, const float* grads, float* m, float* v, float learning_rate, float beta1, float beta2, float eps, size_t step);
void kernel_rmsprop_step(size_t n, float* params, const float* grads, float* cache, float learning_rate, float decay_rate, float eps);

// 内存管理内核
void* kernel_aligned_alloc(size_t size, size_t alignment);
void kernel_aligned_free(void* ptr);
void kernel_prefetch(const void* ptr, size_t size);
void kernel_memset_float(float* dst, float value, size_t n);
void kernel_memcpy_float(float* dst, const float* src, size_t n);

// 性能监控
void kernel_start_profiling(void);
void kernel_stop_profiling(void);
KernelStats kernel_get_stats(void);
void kernel_print_stats(void);

// 汇编优化函数（内部使用）
#ifdef __x86_64__
// x86-64 SIMD优化
void kernel_sgemm_avx2(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
void kernel_sgemm_avx512(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
void kernel_relu_avx2(size_t n, const float* x, float* y);
void kernel_relu_avx512(size_t n, const float* x, float* y);
#endif

#ifdef __arm__
// ARM NEON优化
void kernel_sgemm_neon(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
void kernel_relu_neon(size_t n, const float* x, float* y);
#endif

// CUDA内核（如果支持）
#ifdef ENABLE_CUDA
void kernel_sgemm_cuda(size_t m, size_t n, size_t k, float alpha, const float* A, const float* B, float beta, float* C);
void kernel_relu_cuda(size_t n, const float* x, float* y);
#endif

#endif // KERNELS_H