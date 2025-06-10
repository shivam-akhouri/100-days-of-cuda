#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <math.h>

#define BLOCK_SIZE 4

using namespace std;

__global__ void sigmoid(float* w, float*s, int m, int n){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(row < m && col < n){
        int idx = row*n+col;
        float result = (float)1/(float)(1+expf(-w[idx]));
        s[idx] = result;
    }
}

__global__ void ReLU(float* w, float*s, int m, int n){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(row < m && col < n){
        int idx = row*n+col;
        float result = w[idx] >0 ? w[idx] : 0;
        s[idx] = result;
    }
}

__global__ void tanh_kernel(float* w, float*s, int m, int n){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(row < m && col < n){
        int idx = row*n+col;
        float result = tanhf(w[idx]);
        s[idx] = result;
    }
}

int main(){
    int width = 10, height=10;
    float* w = (float*)malloc(sizeof(float)*width*height);
    float* ans = (float*)malloc(sizeof(float)*width*height);
    float *dw, *ds;
    for(int i = 0; i< width; i++){
        for(int j = 0; j < height; j++){
            w[i*width+j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    cudaMalloc(&dw, sizeof(float)*width*height);
    cudaMalloc(&ds, sizeof(float)*width*height);
    cudaMemcpy(dw, w, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE-1)/BLOCK_SIZE, (height + BLOCK_SIZE -1)/ BLOCK_SIZE);
    tanh_kernel<<<grid, block>>>(dw, ds, width, height);
    cudaMemcpy(ans, ds, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    for(int i = 0;i < width; i++){
        for(int j  = 0; j < height; j++){
            cout << ans[i*width+j] << " ";
        }
        cout << endl;
    }
    return 0;
}