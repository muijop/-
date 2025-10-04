; x86-64 Assembly Optimized Kernels
; Optimized for Intel/AMD processors with AVX/AVX2/AVX512 support

section .text
    global kernel_sgemm_avx2
    global kernel_saxpy_avx2
    global kernel_sdot_avx2
    global kernel_relu_avx2
    global kernel_sigmoid_avx2
    global kernel_tanh_avx2
    global kernel_sgemm_avx512
    global kernel_sdot_avx512

; AVX2 optimized SGEMM (Single precision general matrix multiply)
; C = alpha * A * B + beta * C
; Parameters:
; rdi - m (rows of A and C)
; rsi - n (cols of B and C)
; rdx - k (cols of A, rows of B)
; rcx - alpha
; r8  - A matrix pointer
; r9  - B matrix pointer
; r10 - beta
; r11 - C matrix pointer

kernel_sgemm_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Save parameters
    mov r12, rdi        ; m
    mov r13, rsi        ; n
    mov r14, rdx        ; k
    movss xmm15, xmm0   ; alpha
    mov r15, r8         ; A
    mov rbx, r9         ; B
    movss xmm14, xmm1   ; beta
    mov r8, r11         ; C
    
    ; Convert alpha and beta to vector format
    vbroadcastss ymm15, xmm15  ; alpha in ymm15
    vbroadcastss ymm14, xmm14  ; beta in ymm14
    
    ; Main computation loop
    xor r9, r9          ; i = 0
.i_loop:
    cmp r9, r12
    jge .end_i_loop
    
    xor r10, r10        ; j = 0
.j_loop:
    cmp r10, r13
    jge .end_j_loop
    
    ; Load C[i][j] and multiply by beta
    vmovss xmm0, dword [r8 + r9*r13*4 + r10*4]
    vfmadd132ss xmm0, xmm0, xmm14, xmm14  ; C[i][j] * beta
    
    ; Compute dot product of row i of A and column j of B
    xor r11, r11        ; l = 0
    vxorps ymm0, ymm0, ymm0  ; sum = 0
    
.k_loop:
    cmp r11, r14
    jge .end_k_loop
    
    ; Load A[i][l] and B[l][j]
    vmovss xmm1, dword [r15 + r9*r14*4 + r11*4]  ; A[i][l]
    vmovss xmm2, dword [rbx + r11*r13*4 + r10*4]  ; B[l][j]
    
    ; Multiply and add
    vfmadd132ss xmm0, xmm0, xmm1, xmm2
    
    inc r11
    jmp .k_loop
    
.end_k_loop:
    ; Multiply by alpha and add to C[i][j]
    vfmadd132ss xmm0, xmm0, xmm15, xmm15
    
    ; Store result
    vmovss dword [r8 + r9*r13*4 + r10*4], xmm0
    
    inc r10
    jmp .j_loop
    
.end_j_loop:
    inc r9
    jmp .i_loop
    
.end_i_loop:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; AVX2 optimized SAXPY (y = alpha * x + y)
; Parameters:
; rdi - n (vector length)
; rsi - alpha
; rdx - x pointer
; rcx - y pointer

kernel_saxpy_avx2:
    push rbp
    mov rbp, rsp
    
    ; Save parameters
    mov r8, rdi         ; n
    movss xmm15, xmm0   ; alpha
    mov r9, rdx         ; x
    mov r10, rcx        ; y
    
    ; Convert alpha to vector
    vbroadcastss ymm15, xmm15
    
    ; Process 8 elements at a time
    mov rax, r8
    shr rax, 3          ; n / 8
    jz .remainder
    
.loop_avx2:
    vmovups ymm0, ymmword [r9]      ; load x
    vmovups ymm1, ymmword [r10]     ; load y
    vfmadd132ps ymm1, ymm1, ymm0, ymm15  ; y = alpha * x + y
    vmovups ymmword [r10], ymm1     ; store y
    
    add r9, 32          ; x += 8
    add r10, 32         ; y += 8
    dec rax
    jnz .loop_avx2
    
.remainder:
    ; Process remaining elements
    and r8, 7           ; n % 8
    jz .done
    
    lea rax, [r9 + r8*4] ; end of x
    lea rbx, [r10 + r8*4] ; end of y
    
.remainder_loop:
    vmovss xmm0, dword [r9]
    vmovss xmm1, dword [r10]
    vfmadd132ss xmm1, xmm1, xmm0, xmm15
    vmovss dword [r10], xmm1
    
    add r9, 4
    add r10, 4
    cmp r9, rax
    jl .remainder_loop
    
.done:
    pop rbp
    ret

; AVX2 optimized SDOT (dot product)
; Parameters:
; rdi - n (vector length)
; rsi - x pointer
; rdx - y pointer
; Returns: dot product in xmm0

kernel_sdot_avx2:
    push rbp
    mov rbp, rsp
    
    ; Save parameters
    mov r8, rdi         ; n
    mov r9, rsi         ; x
    mov r10, rdx        ; y
    
    ; Initialize sum to zero
    vxorps ymm0, ymm0, ymm0
    
    ; Process 8 elements at a time
    mov rax, r8
    shr rax, 3          ; n / 8
    jz .remainder
    
.loop_avx2:
    vmovups ymm1, ymmword [r9]      ; load x
    vmovups ymm2, ymmword [r10]     ; load y
    vfmadd132ps ymm0, ymm0, ymm1, ymm2  ; sum += x * y
    
    add r9, 32          ; x += 8
    add r10, 32         ; y += 8
    dec rax
    jnz .loop_avx2
    
.remainder:
    ; Process remaining elements
    and r8, 7           ; n % 8
    jz .reduce
    
    lea rax, [r9 + r8*4] ; end of x
    
.remainder_loop:
    vmovss xmm1, dword [r9]
    vmovss xmm2, dword [r10]
    vfmadd132ss xmm0, xmm0, xmm1, xmm2
    
    add r9, 4
    add r10, 4
    cmp r9, rax
    jl .remainder_loop
    
.reduce:
    ; Reduce 8 elements to 1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    
    pop rbp
    ret

; AVX2 optimized ReLU
; Parameters:
; rdi - n (vector length)
; rsi - x pointer
; rdx - y pointer

kernel_relu_avx2:
    push rbp
    mov rbp, rsp
    
    ; Save parameters
    mov r8, rdi         ; n
    mov r9, rsi         ; x
    mov r10, rdx        ; y
    
    ; Create zero vector
    vxorps ymm15, ymm15, ymm15
    
    ; Process 8 elements at a time
    mov rax, r8
    shr rax, 3          ; n / 8
    jz .remainder
    
.loop_avx2:
    vmovups ymm0, ymmword [r9]      ; load x
    vmaxps ymm0, ymm0, ymm15        ; max(x, 0)
    vmovups ymmword [r10], ymm0     ; store y
    
    add r9, 32          ; x += 8
    add r10, 32         ; y += 8
    dec rax
    jnz .loop_avx2
    
.remainder:
    ; Process remaining elements
    and r8, 7           ; n % 8
    jz .done
    
    lea rax, [r9 + r8*4] ; end of x
    
.remainder_loop:
    vmovss xmm0, dword [r9]
    vmaxss xmm0, xmm0, xmm15
    vmovss dword [r10], xmm0
    
    add r9, 4
    add r10, 4
    cmp r9, rax
    jl .remainder_loop
    
.done:
    pop rbp
    ret

; AVX2 optimized Sigmoid
; Parameters:
; rdi - n (vector length)
; rsi - x pointer
; rdx - y pointer

kernel_sigmoid_avx2:
    push rbp
    mov rbp, rsp
    
    ; Save parameters
    mov r8, rdi         ; n
    mov r9, rsi         ; x
    mov r10, rdx        ; y
    
    ; Create constants
    vxorps ymm15, ymm15, ymm15    ; 0
    vbroadcastss ymm14, dword [one_float]  ; 1.0
    
    ; Process 8 elements at a time
    mov rax, r8
    shr rax, 3          ; n / 8
    jz .remainder
    
.loop_avx2:
    vmovups ymm0, ymmword [r9]      ; load x
    vxorps ymm1, ymm1, ymm1
    vsubps ymm1, ymm1, ymm0         ; -x
    vexp2ps ymm2, ymm1              ; exp(-x) (approximation)
    vaddps ymm2, ymm2, ymm14        ; 1 + exp(-x)
    vdivps ymm0, ymm14, ymm2        ; 1 / (1 + exp(-x))
    vmovups ymmword [r10], ymm0     ; store y
    
    add r9, 32          ; x += 8
    add r10, 32         ; y += 8
    dec rax
    jnz .loop_avx2
    
.remainder:
    ; Process remaining elements
    and r8, 7           ; n % 8
    jz .done
    
    lea rax, [r9 + r8*4] ; end of x
    
.remainder_loop:
    vmovss xmm0, dword [r9]
    vxorps xmm1, xmm1, xmm1
    vsubss xmm1, xmm1, xmm0         ; -x
    vexp2ss xmm2, xmm1              ; exp(-x) (approximation)
    vaddss xmm2, xmm2, xmm14        ; 1 + exp(-x)
    vdivss xmm0, xmm14, xmm2        ; 1 / (1 + exp(-x))
    vmovss dword [r10], xmm0
    
    add r9, 4
    add r10, 4
    cmp r9, rax
    jl .remainder_loop
    
.done:
    pop rbp
    ret

; AVX2 optimized Tanh
; Parameters:
; rdi - n (vector length)
; rsi - x pointer
; rdx - y pointer

kernel_tanh_avx2:
    push rbp
    mov rbp, rsp
    
    ; Save parameters
    mov r8, rdi         ; n
    mov r9, rsi         ; x
    mov r10, rdx        ; y
    
    ; Create constants
    vxorps ymm15, ymm15, ymm15    ; 0
    vbroadcastss ymm14, dword [two_float]  ; 2.0
    vbroadcastss ymm13, dword [one_float]  ; 1.0
    
    ; Process 8 elements at a time
    mov rax, r8
    shr rax, 3          ; n / 8
    jz .remainder
    
.loop_avx2:
    vmovups ymm0, ymmword [r9]      ; load x
    vmulps ymm1, ymm0, ymm14        ; 2*x
    vexp2ps ymm2, ymm1              ; exp(2*x) (approximation)
    vsubps ymm3, ymm2, ymm13        ; exp(2*x) - 1
    vaddps ymm4, ymm2, ymm13        ; exp(2*x) + 1
    vdivps ymm0, ymm3, ymm4         ; tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
    vmovups ymmword [r10], ymm0     ; store y
    
    add r9, 32          ; x += 8
    add r10, 32         ; y += 8
    dec rax
    jnz .loop_avx2
    
.remainder:
    ; Process remaining elements
    and r8, 7           ; n % 8
    jz .done
    
    lea rax, [r9 + r8*4] ; end of x
    
.remainder_loop:
    vmovss xmm0, dword [r9]
    vmulss xmm1, xmm0, xmm14        ; 2*x
    vexp2ss xmm2, xmm1              ; exp(2*x) (approximation)
    vsubss xmm3, xmm2, xmm13        ; exp(2*x) - 1
    vaddss xmm4, xmm2, xmm13        ; exp(2*x) + 1
    vdivss xmm0, xmm3, xmm4         ; tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
    vmovss dword [r10], xmm0
    
    add r9, 4
    add r10, 4
    cmp r9, rax
    jl .remainder_loop
    
.done:
    pop rbp
    ret

; AVX512 optimized SGEMM (Single precision general matrix multiply)
; This is a simplified version for demonstration
; Real implementation would be much more complex

kernel_sgemm_avx512:
    ; AVX512 implementation would go here
    ; For now, just call the AVX2 version
    jmp kernel_sgemm_avx2

; AVX512 optimized SDOT (dot product)
; This processes 16 elements at a time

kernel_sdot_avx512:
    push rbp
    mov rbp, rsp
    
    ; Save parameters
    mov r8, rdi         ; n
    mov r9, rsi         ; x
    mov r10, rdx        ; y
    
    ; Initialize sum to zero
    vxorps zmm0, zmm0, zmm0
    
    ; Process 16 elements at a time
    mov rax, r8
    shr rax, 4          ; n / 16
    jz .remainder
    
.loop_avx512:
    vmovups zmm1, zmmword [r9]      ; load x
    vmovups zmm2, zmmword [r10]     ; load y
    vfmadd132ps zmm0, zmm0, zmm1, zmm2  ; sum += x * y
    
    add r9, 64          ; x += 16
    add r10, 64         ; y += 16
    dec rax
    jnz .loop_avx512
    
.remainder:
    ; Process remaining elements
    and r8, 15          ; n % 16
    jz .reduce
    
    lea rax, [r9 + r8*4] ; end of x
    
.remainder_loop:
    vmovss xmm1, dword [r9]
    vmovss xmm2, dword [r10]
    vfmadd132ss xmm0, xmm0, xmm1, xmm2
    
    add r9, 4
    add r10, 4
    cmp r9, rax
    jl .remainder_loop
    
.reduce:
    ; Reduce 16 elements to 1
    ; This is a simplified reduction
    vextractf64x4 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    
    pop rbp
    ret

section .data
    align 16
one_float: dd 1.0
two_float: dd 2.0