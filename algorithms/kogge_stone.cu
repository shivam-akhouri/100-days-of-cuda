#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_DIM 6

using namespace std;

__global__ void kogge_stone(float *input, float *output, float *partialsums, unsigned int length) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer1[BLOCK_DIM];
    __shared__ float buffer2[BLOCK_DIM];

    float *current = buffer1;
    float *next = buffer2;

    if (idx < length) {
        current[threadIdx.x] = input[idx];
    }
    __syncthreads();

    // Kogge-Stone scan with pointer swapping
    for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        if (threadIdx.x >= stride) {
            next[threadIdx.x] = current[threadIdx.x] + current[threadIdx.x - stride];
        } else {
            next[threadIdx.x] = current[threadIdx.x];
        }
        __syncthreads();

        // Swap pointers instead of copying buffers
        float *temp = current;
        current = next;
        next = temp;
        __syncthreads();
    }

    if (idx < length) {
        output[idx] = current[threadIdx.x];
    }

    // Store partial sums
    if (threadIdx.x == BLOCK_DIM - 1 && idx < length) {
        partialsums[blockIdx.x] = current[threadIdx.x];
    }
}

__global__ void add_kernel(float *output, float *partialsums, unsigned int length) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < length) {
        output[idx] += partialsums[blockIdx.x - 1];
    }
}

int main() {
    int length = 6;
    int size = sizeof(float) * length;
    int numBlocks = (length + BLOCK_DIM - 1) / BLOCK_DIM;
    
    float *input, *output, *partialsums;
    input = (float *)malloc(size);
    output = (float *)malloc(size);
    partialsums = (float *)malloc(sizeof(float) * numBlocks);

    float *dinput, *doutput, *dpartialsums;
    cudaMalloc(&dinput, size);
    cudaMalloc(&doutput, size);
    cudaMalloc(&dpartialsums, sizeof(float) * numBlocks);

    for (int i = 0; i < length; i++) {
        input[i] = i + 1;
    }

    cudaMemcpy(dinput, input, size, cudaMemcpyHostToDevice);
    cout << "Input: ";
    for (int i = 0; i < length; i++) {
        cout << input[i] << " ";
    }
    cout << endl;

    dim3 block(BLOCK_DIM);
    dim3 grid(numBlocks);

    kogge_stone<<<grid, block>>>(dinput, doutput, dpartialsums, length);
    cudaDeviceSynchronize(); // Ensure kernel execution completes

    add_kernel<<<grid, block>>>(doutput, dpartialsums, length);
    cudaDeviceSynchronize();

    cudaMemcpy(output, doutput, size, cudaMemcpyDeviceToHost);

    cout << "Output: ";
    for (int i = 0; i < length; i++) {
        cout << output[i] << " ";
    }
    cout << endl;

    // Cleanup
    free(input);
    free(output);
    free(partialsums);
    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dpartialsums);

    return 0;
}