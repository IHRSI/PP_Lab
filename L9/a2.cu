#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define M 2
#define N 4

__global__ void repeat_chars(char *A, int *B, char *STR, int *offsets, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int idx = row * n + col;
        int start = offsets[idx];
        for (int k = 0; k < B[idx]; k++) {
            STR[start + k] = A[idx];
        }
    }
}

int main() {
    char h_A[M][N] = {
        {'p', 'C', 'a', 'P'},
        {'e', 'X', 'a', 'M'}
    };
    int h_B[M][N] = {
        {1, 2, 4, 3},
        {2, 4, 3, 2}
    };

    int h_offsets[M * N];
    int total = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_offsets[i * N + j] = total;
            total += h_B[i][j];
        }
    }

    printf("Total output length: %d\n", total);

    char *d_A, *d_STR;
    int  *d_B, *d_offsets;

    cudaMalloc(&d_A,       M * N * sizeof(char));
    cudaMalloc(&d_B,       M * N * sizeof(int));
    cudaMalloc(&d_offsets, M * N * sizeof(int));
    cudaMalloc(&d_STR,     total * sizeof(char));

    cudaMemcpy(d_A,       h_A,       M * N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,       h_B,       M * N * sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, M * N * sizeof(int),  cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    repeat_chars<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_STR, d_offsets, M, N);

    char *h_STR = (char *)malloc((total + 1) * sizeof(char));
    cudaMemcpy(h_STR, d_STR, total * sizeof(char), cudaMemcpyDeviceToHost);
    h_STR[total] = '\0';

    printf("Output String STR: %s\n", h_STR);

    free(h_STR);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_offsets);
    cudaFree(d_STR);
    return 0;
}