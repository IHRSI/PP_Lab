#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void replaceEvenOdd(int *A, int *B, int M, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) {
        int val = A[r * N + c];
        int sum = 0;

        if (val % 2 == 0) {
            for (int j = 0; j < N; j++) {
                sum += A[r * N + j];
            }
        } else {
            for (int i = 0; i < M; i++) {
                sum += A[i * N + c];
            }
        }
        B[r * N + c] = sum;
    }
}

int main() {
    int M, N;
    printf("Enter number of rows (M) and columns (N): ");
    scanf("%d %d", &M, &N);

    size_t size = M * N * sizeof(int);
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);

    printf("Enter the %d elements of the matrix:\n", M * N);
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    replaceEvenOdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_B[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
}