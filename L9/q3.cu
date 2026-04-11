#include <stdio.h>
#include <cuda_runtime.h>

#define M 4
#define N 4

__global__ void ones_complement(int *A, int *B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int idx = row * n + col;
        if (row == 0 || row == m-1 || col == 0 || col == n-1) {
            B[idx] = A[idx];
        } else {
            B[idx] = ~A[idx];
        }
    }
}

void print_binary(int n) {
    if (n == 0) { printf("0"); return; }
    int bits[32], count = 0;
    unsigned int un = (unsigned int)n;
    while (un > 0) {
        bits[count++] = un & 1;
        un >>= 1;
    }
    for (int i = count - 1; i >= 0; i--)
        printf("%d", bits[i]);
}

int main() {
    int h_A[M][N] = {
        {1,  2,  3,  4},
        {6,  5,  8,  3},
        {2,  4, 10,  1},
        {9,  1,  2,  5}
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

    ones_complement<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nMatrix B (non-border shown in binary):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == M-1 || j == 0 || j == N-1)
                printf("%4d ", h_B[i][j]);
            else {
                printf("%4d ", h_A[i][j] ^ ((1 << (int)(log2(h_A[i][j]) + 1)) - 1));
            }
        }
        printf("\n");
    }

    printf("\nMatrix B (non-border in binary format as shown in question):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == M-1 || j == 0 || j == N-1)
                printf("%5d ", h_A[i][j]);
            else {
                int comp = h_A[i][j] ^ ((1 << (int)(log2(h_A[i][j]) + 1)) - 1);
                printf("%5d ", comp);
                printf("(");
                print_binary(comp);
                printf(") ");
            }
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}