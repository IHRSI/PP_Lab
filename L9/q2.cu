#include <stdio.h>
#include <cuda_runtime.h>

#define M 3
#define N 3

__global__ void transform(float *A, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int idx = row * n + col;
        float val = A[idx];
        int power = row + 1;
        float result = 1.0f;
        for (int i = 0; i < power; i++) {
            result *= val;
        }
        A[idx] = result;
    }
}

int main() {
    float h_A[M][N] = {
        {1, 2, 3},
        {2, 3, 4},
        {2, 3, 4}
    };

    printf("Original matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%.2f ", h_A[i][j]);
        printf("\n");
    }

    float *d_A;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    transform<<<blocksPerGrid, threadsPerBlock>>>(d_A, M, N);

    cudaMemcpy(h_A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nTransformed matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%.2f ", h_A[i][j]);
        printf("\n");
    }

    cudaFree(d_A);
    return 0;
}