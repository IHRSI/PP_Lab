#include <stdio.h>
#include <cuda_runtime.h>

#define M 2
#define N 3

__global__ void compute_B(int *A, int *B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int row_sum = 0;
        for (int j = 0; j < n; j++)
            row_sum += A[row * n + j];

        int col_sum = 0;
        for (int i = 0; i < m; i++)
            col_sum += A[i * n + col];

        B[row * n + col] = row_sum + col_sum;
    }
}

int main() {
    int h_A[M][N] = {
        {1, 2, 3},
        {4, 5, 6}
    };
    int h_B[M][N] = {0};

    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%4d ", h_A[i][j]);
        printf("\n");
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    compute_B<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nMatrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%4d ", h_B[i][j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}