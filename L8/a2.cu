#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__device__ int getFactorial(int n) {
    if (n < 0) return 0; 
    if (n == 0) return 1;
    int res = 1;
    for (int i = 1; i <= n; i++) {
        res *= i;
    }
    return res;
}
__device__ int getSumDigits(int n) {
    int sum = 0;
    n = (n < 0) ? -n : n;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}
__global__ void processMatrix(int *A, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < N && c < N) {
        int idx = r * N + c;
        if (r == c) {
            A[idx] = 0;
        } 
        else if (r < c) {
            A[idx] = getFactorial(A[idx]);
        } 
        else {
            A[idx] = getSumDigits(A[idx]);
        }
    }
}

int main() {
    int N;
    printf("Enter the size of the NXN matrix: ");
    scanf("%d", &N);
    size_t size = N * N * sizeof(int);
    int *h_A = (int*)malloc(size);

    printf("Enter the %d elements of the matrix:\n", N * N);
    for (int i = 0; i < N * N; i++) {
        scanf("%d", &h_A[i]);
    }
    int *d_A;
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    processMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, N);
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    printf("\nResultant Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_A[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    free(h_A);
    return 0;
}