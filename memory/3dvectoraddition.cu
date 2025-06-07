#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 4
#define HEIGHT 4
#define DEPTH 4

__global__ void volumeAddKernel(cudaPitchedPtr d_A, cudaPitchedPtr d_B, cudaPitchedPtr d_C, int width, int height, int depth) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < width && y < height && z < depth) {
        char* sliceA = (char*)d_A.ptr + z * d_A.pitch * height;
        char* rowA = sliceA + y * d_A.pitch;
        float* elementA = (float*)(rowA) + x;

        char* sliceB = (char*)d_B.ptr + z * d_B.pitch * height;
        char* rowB = sliceB + y * d_B.pitch;
        float* elementB = (float*)(rowB) + x;

        char* sliceC = (char*)d_C.ptr + z * d_C.pitch * height;
        char* rowC = sliceC + y * d_C.pitch;
        float* elementC = (float*)(rowC) + x;

        *elementC = *elementA + *elementB;
    }
}

int main() {
    int width = WIDTH, height = HEIGHT, depth = DEPTH;

    // Allocate host memory (contiguous 3D arrays flattened to 1D)
    size_t volumeSize = width * height * depth * sizeof(float);
    float* h_A = (float*)malloc(volumeSize);
    float* h_B = (float*)malloc(volumeSize);
    float* h_C = (float*)malloc(volumeSize);

    // Initialize volumes
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = z * width * height + y * width + x;
                h_A[idx] = static_cast<float>(z);
                h_B[idx] = static_cast<float>(x);
                h_C[idx] = 0;
            }
        }
    }

    // Device pitched memory allocation
    cudaExtent volumeExtent = make_cudaExtent(width * sizeof(float), height, depth);
    cudaPitchedPtr d_A, d_B, d_C;

    cudaMalloc3D(&d_A, volumeExtent);
    cudaMalloc3D(&d_B, volumeExtent);
    cudaMalloc3D(&d_C, volumeExtent);

    // Host pitched memory setup
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = volumeExtent;
    copyParams.kind = cudaMemcpyHostToDevice;

    // A → d_A
    copyParams.srcPtr = make_cudaPitchedPtr(h_A, width * sizeof(float), width, height);
    copyParams.dstPtr = d_A;
    cudaMemcpy3D(&copyParams);

    // B → d_B
    copyParams.srcPtr = make_cudaPitchedPtr(h_B, width * sizeof(float), width, height);
    copyParams.dstPtr = d_B;
    cudaMemcpy3D(&copyParams);

    // Launch kernel
    dim3 blockDim(4, 4, 4);
    dim3 gridDim((width + 3) / 4, (height + 3) / 4, (depth + 3) / 4);

    volumeAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height, depth);
    cudaDeviceSynchronize();

    // Copy result back to host
    copyParams.kind = cudaMemcpyDeviceToHost;
    copyParams.srcPtr = d_C;
    copyParams.dstPtr = make_cudaPitchedPtr(h_C, width * sizeof(float), width, height);
    cudaMemcpy3D(&copyParams);

    // Display result
    std::cout << "Result (C = A + B):\n";
    for (int z = 0; z < depth; ++z) {
        std::cout << "Depth " << z << ":\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = z * width * height + y * width + x;
                std::cout << h_C[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_A.ptr);
    cudaFree(d_B.ptr);
    cudaFree(d_C.ptr);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
