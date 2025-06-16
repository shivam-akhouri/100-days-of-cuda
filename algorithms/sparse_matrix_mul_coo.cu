#include <cuda_runtime.h>
#include <iostream>

#define N 4  // Matrix size

// Kernel function for COO SpMV
__global__ void coo_spmv(int *rowIdx, int *colIdx, float *values, int nnz, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&y[rowIdx[i]], values[i] * x[colIdx[i]]);
    }
}

int main() {
    // Example sparse matrix in COO format
    int h_rowIdx[] = {0, 0, 1, 2, 2, 3};
    int h_colIdx[] = {0, 1, 1, 2, 3, 3};
    float h_values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int nnz = 6;

    // Input vector
    float h_x[] = {1, 2, 3, 4};
    float h_y[N] = {0};

    // Device memory allocation
    int *d_rowIdx, *d_colIdx;
    float *d_values, *d_x, *d_y;
    
    cudaMalloc(&d_rowIdx, nnz * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_rowIdx, h_rowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    coo_spmv<<<1, nnz>>>(d_rowIdx, d_colIdx, d_values, nnz, d_x, d_y);
    
    // Copy result back
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result: ";
    for (int i = 0; i < N; i++)
        std::cout << h_y[i] << " ";
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_rowIdx);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}