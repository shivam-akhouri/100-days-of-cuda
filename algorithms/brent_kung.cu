#include <stdio.h>
#include <cuda.h>

#define N 16  // Size of input
#define LOG_N 4  // log2(N)

__global__ void brentKungScan(int* data) {
    __shared__ int temp[N];

    int tid = threadIdx.x;
    temp[tid] = data[tid];
    __syncthreads();

    // Upsweep (reduce) phase
    for (int offset = 1; offset < N; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < N)
            temp[index] += temp[index - offset];
        __syncthreads();
    }

    // Downsweep phase
    for (int offset = N / 2; offset > 0; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index + offset < N)
            temp[index + offset] += temp[index];
        __syncthreads();
    }

    data[tid] = temp[tid];
}

int main() {
    int h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = i + 1;  // dummy input

    int* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    brentKungScan<<<1, N>>>(d_data);

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    return 0;
}