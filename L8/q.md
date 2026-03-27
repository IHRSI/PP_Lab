Programs on Matrix using CUDA
Lab Exercises:
1. Write a program in CUDA to add two matrices for the following specifications:
a. Each row of resultant matrix to be computed by one thread.
b. Each column of resultant matrix to be computed by one thread.
c. Each element of resultant matrix to be computed by one thread.

2. Write a program in CUDA to multiply two matrices for the following specifications:
a. Each row of resultant matrix to be computed by one thread.
b. Each column of resultant matrix to be computed by one thread.
c. Each element of resultant matrix to be computed by one thread.

Additional Exercises:
1. Write a CUDA program that reads a MXN matrix A and produces a resultant matrix B of same size as follows: Replace all the even numbered matrix elements with their row sum and odd numbered matrix elements with their column sum.

2. Write a CUDA program to read a matrix A of size NXN. It replaces the principle diagonal elements with zero. Elements above the principle diagonal by their factorial and elements below the principle diagonal by their sum of digits.

__global__ void addRows(int *A, int *B, int *C, int N) {
    int row = blockIdx.x * block_dim.x + threadIdx.x;
    if (row < N) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

__global__ void addCols(int *A, int *B, int *C, int N) {
    int col = blockIdx.x * block_dim.x + threadIdx.x;
    if (col < N) {
        for (int row = 0; row < N; row++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

__global__ void addElements(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void mulRows(int *A, int *B, int *C, int N) {
    int row = threadIdx.x; 
    if (row < N) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[row * N + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
    }
}

__global__ void mulCols(int *A, int *B, int *C, int N) {
    int col = threadIdx.x;
    if (col < N) {
        for (int row = 0; row < N; row++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[row * N + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
    }
}

__global__ void mulElements(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void replaceElements(int *A, int *B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int val = A[row * N + col];
        int sum = 0;
        if (val % 2 == 0) { // Even: Row Sum
            for (int j = 0; j < N; j++) sum += A[row * N + j];
        } else { // Odd: Column Sum
            for (int i = 0; i < M; i++) sum += A[i * N + col];
        }
        B[row * N + col] = sum;
    }
}

__device__ int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

__device__ int sumDigits(int n) {
    int s = 0;
    while (n > 0) { s += n % 10; n /= 10; }
    return s;
}

__global__ void processMatrix(int *A, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < N && c < N) {
        int idx = r * N + c;
        if (r == c) A[idx] = 0;           // Principal Diagonal
        else if (r < c) A[idx] = factorial(A[idx]); // Above
        else A[idx] = sumDigits(A[idx]);  // Below
    }
}