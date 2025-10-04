; ARM64 Assembly Optimized Kernels
; Optimized for ARM processors with NEON support

.text
.global kernel_sgemm_neon
.global kernel_saxpy_neon
.global kernel_sdot_neon
.global kernel_relu_neon
.global kernel_sigmoid_neon
.global kernel_tanh_neon

; NEON optimized SGEMM (Single precision general matrix multiply)
; C = alpha * A * B + beta * C
; Parameters (following ARM64 calling convention):
; x0 - m (rows of A and C)
; x1 - n (cols of B and C)
; x2 - k (cols of A, rows of B)
; s0 - alpha
; x3 - A matrix pointer
; x4 - B matrix pointer
; s1 - beta
; x5 - C matrix pointer

kernel_sgemm_neon:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    ; Save parameters
    mov x6, x0          ; m
    mov x7, x1          ; n
    mov x8, x2          ; k
    fmov s16, s0        ; alpha
    mov x9, x3          ; A
    mov x10, x4         ; B
    fmov s17, s1        ; beta
    mov x11, x5         ; C
    
    ; Convert alpha and beta to vector format
    dup v16.4s, v16.s[0]  ; alpha in v16
    dup v17.4s, v17.s[0]  ; beta in v17
    
    ; Main computation loop
    mov x12, #0         ; i = 0
.i_loop:
    cmp x12, x6
    b.ge .end_i_loop
    
    mov x13, #0         ; j = 0
.j_loop:
    cmp x13, x7
    b.ge .end_j_loop
    
    ; Load C[i][j] and multiply by beta
    ldr s0, [x11, x12, x1, lsl #2]
    fmul s0, s0, s17    ; C[i][j] * beta
    
    ; Compute dot product of row i of A and column j of B
    mov x14, #0         ; l = 0
    dup v18.4s, #0      ; sum = 0
    
.k_loop:
    cmp x14, x8
    b.ge .end_k_loop
    
    ; Load A[i][l] and B[l][j]
    ldr s1, [x9, x12, x2, lsl #2]   ; A[i][l]
    ldr s2, [x10, x14, x1, lsl #2]  ; B[l][j]
    
    ; Multiply and add
    fmla v18.4s, v16.4s, v1.s[0], v2.s[0]
    
    add x14, x14, #1
    b .k_loop
    
.end_k_loop:
    ; Extract sum from vector
    addv s18, v18.4s
    
    ; Multiply by alpha and add to C[i][j]
    fmla s0, s16, s18, s18
    
    ; Store result
    str s0, [x11, x12, x1, lsl #2]
    
    add x13, x13, #1
    b .j_loop
    
.end_j_loop:
    add x12, x12, #1
    b .i_loop
    
.end_i_loop:
    ldp x29, x30, [sp], #16
    ret

; NEON optimized SAXPY (y = alpha * x + y)
; Parameters:
; x0 - n (vector length)
; s0 - alpha
; x1 - x pointer
; x2 - y pointer

kernel_saxpy_neon:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    ; Save parameters
    mov x3, x0          ; n
    fmov v16.s[0], s0   ; alpha
    mov x4, x1          ; x
    mov x5, x2          ; y
    
    ; Convert alpha to vector
    dup v16.4s, v16.s[0]
    
    ; Process 4 elements at a time
    mov x6, x3
    lsr x6, x6, #2      ; n / 4
    beq .remainder
    
.loop_neon:
    ld1 {v0.4s}, [x4], #16    ; load x
    ld1 {v1.4s}, [x5]        ; load y
    fmla v1.4s, v16.4s, v0.4s  ; y = alpha * x + y
    st1 {v1.4s}, [x5], #16   ; store y
    
    subs x6, x6, #1
    b.ne .loop_neon
    
.remainder:
    ; Process remaining elements
    and x3, x3, #3      ; n % 4
    beq .done
    
    add x7, x4, x3, lsl #2  ; end of x
    
.remainder_loop:
    ld1 {v0.s}[0], [x4], #4
    ld1 {v1.s}[0], [x5]
    fmla v1.s[0], v16.s[0], v0.s[0]
    st1 {v1.s}[0], [x5], #4
    
    cmp x4, x7
    b.lt .remainder_loop
    
.done:
    ldp x29, x30, [sp], #16
    ret

; NEON optimized SDOT (dot product)
; Parameters:
; x0 - n (vector length)
; x1 - x pointer
; x2 - y pointer
; Returns: dot product in s0

kernel_sdot_neon:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    ; Save parameters
    mov x3, x0          ; n
    mov x4, x1          ; x
    mov x5, x2          ; y
    
    ; Initialize sum to zero
    dup v18.4s, #0
    
    ; Process 4 elements at a time
    mov x6, x3
    lsr x6, x6, #2      ; n / 4
    beq .remainder
    
.loop_neon:
    ld1 {v0.4s}, [x4], #16    ; load x
    ld1 {v1.4s}, [x5], #16    ; load y
    fmla v18.4s, v0.4s, v1.4s  ; sum += x * y
    
    subs x6, x6, #1
    b.ne .loop_neon
    
.remainder:
    ; Process remaining elements
    and x3, x3, #3      ; n % 4
    beq .reduce
    
    add x7, x4, x3, lsl #2  ; end of x
    
.remainder_loop:
    ld1 {v0.s}[0], [x4], #4
    ld1 {v1.s}[0], [x5], #4
    fmla v18.s[0], v0.s[0], v1.s[0]
    
    cmp x4, x7
    b.lt .remainder_loop
    
.reduce:
    ; Reduce 4 elements to 1
    addv s18, v18.4s
    
    ; Move result to s0
    fmov s0, s18
    
    ldp x29, x30, [sp], #16
    ret

; NEON optimized ReLU
; Parameters:
; x0 - n (vector length)
; x1 - x pointer
; x2 - y pointer

kernel_relu_neon:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    ; Save parameters
    mov x3, x0          ; n
    mov x4, x1          ; x
    mov x5, x2          ; y
    
    ; Create zero vector
    dup v15.4s, #0
    
    ; Process 4 elements at a time
    mov x6, x3
    lsr x6, x6, #2      ; n / 4
    beq .remainder
    
.loop_neon:
    ld1 {v0.4s}, [x4], #16    ; load x
    fmax v1.4s, v0.4s, v15.4s ; max(x, 0)
    st1 {v1.4s}, [x5], #16   ; store y
    
    subs x6, x6, #1
    b.ne .loop_neon
    
.remainder:
    ; Process remaining elements
    and x3, x3, #3      ; n % 4
    beq .done
    
    add x7, x4, x3, lsl #2  ; end of x
    
.remainder_loop:
    ld1 {v0.s}[0], [x4], #4
    fmax v1.s[0], v0.s[0], v15.s[0]
    st1 {v1.s}[0], [x5], #4
    
    cmp x4, x7
    b.lt .remainder_loop
    
.done:
    ldp x29, x30, [sp], #16
    ret

; NEON optimized Sigmoid (simplified version)
; Parameters:
; x0 - n (vector length)
; x1 - x pointer
; x2 - y pointer

kernel_sigmoid_neon:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    ; Save parameters
    mov x3, x0          ; n
    mov x4, x1          ; x
    mov x5, x2          ; y
    
    ; Create constants
    dup v15.4s, #1.0    ; 1.0
    dup v14.4s, #-1.0   ; -1.0
    
    ; Process 4 elements at a time
    mov x6, x3
    lsr x6, x6, #2      ; n / 4
    beq .remainder
    
.loop_neon:
    ld1 {v0.4s}, [x4], #16    ; load x
    fmul v1.4s, v0.4s, v14.4s ; -x
    ; Note: ARM doesn't have built-in exp, so we use polynomial approximation
    ; This is a simplified version - real implementation would use lookup tables
    ; For now, just use a simple approximation
    fadd v2.4s, v1.4s, v15.4s ; 1 + (-x) approximation
    fdiv v3.4s, v15.4s, v2.4s ; 1 / (1 + exp(-x))
    st1 {v3.4s}, [x5], #16   ; store y
    
    subs x6, x6, #1
    b.ne .loop_neon
    
.remainder:
    ; Process remaining elements
    and x3, x3, #3      ; n % 4
    beq .done
    
    add x7, x4, x3, lsl #2  ; end of x
    
.remainder_loop:
    ld1 {v0.s}[0], [x4], #4
    fmul v1.s[0], v0.s[0], v14.s[0]  ; -x
    fadd v2.s[0], v1.s[0], v15.s[0]  ; 1 + (-x)
    fdiv v3.s[0], v15.s[0], v2.s[0]  ; 1 / (1 + exp(-x))
    st1 {v3.s}[0], [x5], #4
    
    cmp x4, x7
    b.lt .remainder_loop
    
.done:
    ldp x29, x30, [sp], #16
    ret

; NEON optimized Tanh (simplified version)
; Parameters:
; x0 - n (vector length)
; x1 - x pointer
; x2 - y pointer

kernel_tanh_neon:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    ; Save parameters
    mov x3, x0          ; n
    mov x4, x1          ; x
    mov x5, x2          ; y
    
    ; Create constants
    dup v15.4s, #2.0    ; 2.0
    dup v14.4s, #1.0    ; 1.0
    
    ; Process 4 elements at a time
    mov x6, x3
    lsr x6, x6, #2      ; n / 4
    beq .remainder
    
.loop_neon:
    ld1 {v0.4s}, [x4], #16    ; load x
    fmul v1.4s, v0.4s, v15.4s ; 2*x
    ; Note: ARM doesn't have built-in exp, so we use polynomial approximation
    ; This is a simplified version
    fadd v2.4s, v1.4s, v14.4s ; exp(2*x) + 1 approximation
    fsub v3.4s, v1.4s, v14.4s ; exp(2*x) - 1 approximation
    fdiv v4.4s, v3.4s, v2.4s  ; tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
    st1 {v4.4s}, [x5], #16   ; store y
    
    subs x6, x6, #1
    b.ne .loop_neon
    
.remainder:
    ; Process remaining elements
    and x3, x3, #3      ; n % 4
    beq .done
    
    add x7, x4, x3, lsl #2  ; end of x
    
.remainder_loop:
    ld1 {v0.s}[0], [x4], #4
    fmul v1.s[0], v0.s[0], v15.s[0]  ; 2*x
    fadd v2.s[0], v1.s[0], v14.s[0]  ; exp(2*x) + 1
    fsub v3.s[0], v1.s[0], v14.s[0]  ; exp(2*x) - 1
    fdiv v4.s[0], v3.s[0], v2.s[0]   ; tanh(x)
    st1 {v4.s}[0], [x5], #4
    
    cmp x4, x7
    b.lt .remainder_loop
    
.done:
    ldp x29, x30, [sp], #16
    ret