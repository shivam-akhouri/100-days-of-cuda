#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA Kernel for CSR Sparse Matrix-Vector Multiplication
__global__ void spmv_csr_kernel(int num_rows, const int *row_ptr, const int *col_idx,
                                const float *values, const float *x, float *y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int jj = row_start; jj < row_end; ++jj) {
            sum += values[jj] * x[col_idx[jj]];
        }

        y[row] = sum;
    }
}

// Host function to perform SpMV using CUDA
void spmv_csr(int num_rows, int nnz, const int *h_row_ptr, const int *h_col_idx,
              const float *h_values, const float *h_x, float *h_y) {
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    // Allocate device memory
    cudaMalloc(&d_row_ptr, sizeof(int) * (num_rows + 1));
    cudaMalloc(&d_col_idx, sizeof(int) * nnz);
    cudaMalloc(&d_values, sizeof(float) * nnz);
    cudaMalloc(&d_x, sizeof(float) * num_rows);
    cudaMalloc(&d_y, sizeof(float) * num_rows);

    // Copy data to device
    cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(float) * num_rows, cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    spmv_csr_kernel<<<grid_size, block_size>>>(num_rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_y, d_y, sizeof(float) * num_rows, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    // Example CSR matrix (3x3):
    // [ 10  0  0 ]
    // [  0 20  0 ]
    // [ 30 40 50 ]
    int num_rows = 3;
    int nnz = 5;

    int h_row_ptr[] = {0, 1, 2, 5};
    int h_col_idx[] = {0, 1, 0, 1, 2};
    float h_values[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float h_x[] = {1.0f, 2.0f, 3.0f};
    float h_y[3] = {0.0f};

    spmv_csr(num_rows, nnz, h_row_ptr, h_col_idx, h_values, h_x, h_y);

    // Print result
    std::cout << "Result y = A * x:" << std::endl;
    for (int i = 0; i < num_rows; ++i) {
        std::cout << h_y[i] << std::endl;
    }

    return 0;
}
