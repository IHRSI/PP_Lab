#include <stdio.h>
#include <cuda_runtime.h>

#define N 3

__global__ void spmv_csr(float *data, int *col_index, int *row_ptr, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += data[j] * x[col_index[j]];
        }
        y[row] = sum;
    }
}

int main() {
    float h_data[]      = {1, 2, 3, 4, 5, 6};
    int   h_col_index[] = {0, 2, 0, 1, 2, 1};
    int   h_row_ptr[]   = {0, 2, 5, 6};
    float h_x[]         = {1, 2, 3};
    float h_y[N]        = {0};

    int nnz      = 6;
    int num_rows = N;

    float *d_data, *d_x, *d_y;
    int   *d_col_index, *d_row_ptr;

    cudaMalloc(&d_data,      nnz * sizeof(float));
    cudaMalloc(&d_col_index, nnz * sizeof(int));
    cudaMalloc(&d_row_ptr,   (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_x,         N * sizeof(float));
    cudaMalloc(&d_y,         N * sizeof(float));

    cudaMemcpy(d_data,      h_data,      nnz * sizeof(float),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, h_col_index, nnz * sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr,   h_row_ptr,   (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,         h_x,         N * sizeof(float),            cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    spmv_csr<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_col_index, d_row_ptr, d_x, d_y, num_rows);

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result vector y:\n");
    for (int i = 0; i < N; i++) {
        printf("y[%d] = %.2f\n", i, h_y[i]);
    }

    cudaFree(d_data);
    cudaFree(d_col_index);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}