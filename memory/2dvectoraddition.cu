#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixAddition(float* a, float* b, float* c, int width, int height, size_t pitcha, size_t pitchb, size_t pitchc){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < height && col < width){
        float* rowa = (float*)((char*)a + row * pitcha);
        float* rowb = (float*)((char*)b + row * pitchb);
        float* rowc = (float*)((char*)c + row * pitchc);
        rowc[col] = rowa[col] + rowb[col];
    }
}

int main() {
    int width = 5, height = 5;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    size_t pitcha, pitchb, pitchc;

    // Host memory allocation
    a = (float*)malloc(width * height * sizeof(float));
    b = (float*)malloc(width * height * sizeof(float));
    c = (float*)malloc(width * height * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            a[i * width + j] = i;
            b[i * width + j] = j;
            c[i * width + j] = 0;
        }
    }

    // Device memory allocation with pitch
    cudaMallocPitch(&d_a, &pitcha, width * sizeof(float), height);
    cudaMallocPitch(&d_b, &pitchb, width * sizeof(float), height);
    cudaMallocPitch(&d_c, &pitchc, width * sizeof(float), height);

    // Copy host to device
    cudaMemcpy2D(d_a, pitcha, a, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pitchb, b, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(2, 2);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    matrixAddition<<<gridDim, blockDim>>>(d_a, d_b, d_c, width, height, pitcha, pitchb, pitchc);

    // Copy result back to host
    cudaMemcpy2D(c, width * sizeof(float), d_c, pitchc, width * sizeof(float), height, cudaMemcpyDeviceToHost);

    // Display result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << c[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
