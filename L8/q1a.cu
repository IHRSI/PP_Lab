#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void addRows(int *A, int *B, int *C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int col = 0; col < N; col++) {
            int idx = row * N + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main() {
    int N;
    printf("Enter size N for NxN matrix addition: ");
    scanf("%d", &N);

    size_t size = N * N * sizeof(int);
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);

    printf("Enter %d elements for Matrix A:\n", N * N);
    for (int i = 0; i < N * N; i++) scanf("%d", &h_A[i]);

    printf("Enter %d elements for Matrix B:\n", N * N);
    for (int i = 0; i < N * N; i++) scanf("%d", &h_B[i]);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d threads (one per row).\n", N);
    addRows<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix C (Row-wise Addition):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_C[i * N + j]);
        }
        printf("\n");
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}