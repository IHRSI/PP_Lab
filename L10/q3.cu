#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void tiledConvolution1D(float* N, float* P, float* M, int width, int mask_width) {
    extern __shared__ float N_ds[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = mask_width / 2;
    int l_idx = threadIdx.x + radius; 
    
    if (i < width) {
        N_ds[l_idx] = N[i];
    } else {
        N_ds[l_idx] = 0.0f;
    }

    if (threadIdx.x < radius) {
        int left_neighbor = i - radius;
        N_ds[threadIdx.x] = (left_neighbor >= 0) ? N[left_neighbor] : 0.0f;
    }

    if (threadIdx.x >= blockDim.x - radius) {
        int right_neighbor = i + radius;
        if (right_neighbor < width) {
            N_ds[l_idx + radius] = N[right_neighbor];
        } else {
            N_ds[l_idx + radius] = 0.0f;
        }
    }

    __syncthreads();

    if (i < width) {
        float Pvalue = 0.0f;
        for (int j = 0; j < mask_width; j++) {
            Pvalue += N_ds[threadIdx.x + j] * M[j];
        }
        P[i] = Pvalue;
    }
}

int main() {
    int width, m_width;
    
    printf("Enter array width: ");
    scanf("%d", &width);
    printf("Enter mask width (must be odd): ");
    scanf("%d", &m_width);

    size_t size_N = width * sizeof(float);
    size_t size_M = m_width * sizeof(float);

    float *h_N = (float*)malloc(size_N);
    float *h_P = (float*)malloc(size_N);
    float *h_M = (float*)malloc(size_M);

    printf("Enter %d elements for array N: ", width);
    for (int i = 0; i < width; i++) scanf("%f", &h_N[i]);
    
    printf("Enter %d elements for mask M: ", m_width);
    for (int i = 0; i < m_width; i++) scanf("%f", &h_M[i]);

    float *d_N, *d_P, *d_M;
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_N);
    cudaMalloc((void**)&d_M, size_M);

    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);

    int threads = 16;
    int blocks = (width + threads - 1) / threads;
    size_t sharedMemBytes = (threads + m_width - 1) * sizeof(float);

    tiledConvolution1D<<<blocks, threads, sharedMemBytes>>>(d_N, d_P, d_M, width, m_width);

    cudaMemcpy(h_P, d_P, size_N, cudaMemcpyDeviceToHost);

    printf("\nResult P:\n");
    for (int i = 0; i < width; i++) printf("%.2f ", h_P[i]);
    printf("\n");

    cudaFree(d_N); cudaFree(d_P); cudaFree(d_M);
    free(h_N); free(h_P); free(h_M);

    return 0;
}