#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_MASK_SIZE 1024
__constant__ float d_mask[MAX_MASK_SIZE];

__global__ void convolution1D(float* I, float* J, int n, int mask_width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float result = 0.0f;
        int radius = mask_width / 2;

        for (int j = 0; j < mask_width; j++) {
            int cur_index = i + j - radius;
            
            if (cur_index >= 0 && cur_index < n) {
                result += I[cur_index] * d_mask[j];
            }
        }
        J[i] = result;
    }
}

int main() {
    int n, m_width;

    printf("Enter size of input array (n): ");
    scanf("%d", &n);

    printf("Enter size of mask (must be odd and <= 1024): ");
    scanf("%d", &m_width);

    size_t size_I = n * sizeof(float);
    size_t size_M = m_width * sizeof(float);

    float *h_I = (float*)malloc(size_I);
    float *h_J = (float*)malloc(size_I);
    float *h_M = (float*)malloc(size_M);

    printf("Enter %d elements for input array:\n", n);
    for (int i = 0; i < n; i++) scanf("%f", &h_I[i]);

    printf("Enter %d elements for mask:\n", m_width);
    for (int i = 0; i < m_width; i++) scanf("%f", &h_M[i]);

    float *d_I, *d_J;
    cudaMalloc((void**)&d_I, size_I);
    cudaMalloc((void**)&d_J, size_I);

    cudaMemcpy(d_I, h_I, size_I, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(d_mask, h_M, size_M);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    convolution1D<<<blocks, threads>>>(d_I, d_J, n, m_width);

    cudaMemcpy(h_J, d_J, size_I, cudaMemcpyDeviceToHost);

    printf("\nResultant Array:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", h_J[i]);
    }
    printf("\n");

    cudaFree(d_I);
    cudaFree(d_J);
    free(h_I);
    free(h_J);
    free(h_M);

    return 0;
}