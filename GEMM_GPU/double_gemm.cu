#include <stdlib.h>
#include <iostream>

// CUDA kernel to perform matrix multiplication
__global__ void matrixMul(double *a, double *b, double *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 2){
        printf("Missing input arg\n");
        return -1;
    }

    int N = atoi(argv[1]);
    double *a, *b, *c; // Host matrices
    double *d_a, *d_b, *d_c; // Device matrices

    // Allocate memory for host matrices
    a = (double*)malloc(N * N * sizeof(double));
    b = (double*)malloc(N * N * sizeof(double));
    c = (double*)malloc(N * N * sizeof(double));

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate memory for device matrices
    cudaMalloc((void**)&d_a, N * N * sizeof(double));
    cudaMalloc((void**)&d_b, N * N * sizeof(double));
    cudaMalloc((void**)&d_c, N * N * sizeof(double));

    // Copy host matrices to device
    cudaMemcpy(d_a, a, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid((N + 15) / 16, (N + 15) / 16);
    dim3 dimBlock(16, 16);

    //create GPU timing events for timing the GPU
    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);        
     
    cudaEventRecord(st2);

    // Launch kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    cudaEventRecord(et2);

    //host waits until et2 has occured     
    cudaEventSynchronize(et2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);

    printf("N=%d Kernel time: %f ms\n", N, milliseconds);

    // Copy result from device to host
    cudaMemcpy(c, d_c, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Display result
    printf("Result Matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", c[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
