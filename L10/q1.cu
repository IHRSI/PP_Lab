#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    int N;
    printf("Enter the size of the square matrix (N): ");
    scanf("%d", &N);

    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    printf("Enter elements for Matrix A (%dx%d):\n", N, N);
    for (int i = 0; i < N * N; i++) {
        scanf("%f", &h_A[i]);
    }

    printf("Enter elements for Matrix B (%dx%d):\n", N, N);
    for (int i = 0; i < N * N; i++) {
        scanf("%f", &h_B[i]);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 1;

    dim3 blocksPerGrid;
    blocksPerGrid.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
    blocksPerGrid.z = 1;

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("\nResulting Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}