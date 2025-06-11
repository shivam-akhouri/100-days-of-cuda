#include <iostream>
#include <cuda_runtime.h>
#define MASK_WIDTH 3
#define MASK_HEIGHT 3
#define MASK_RADIUS 1
#define BLOCK_SIZE 3

using namespace std;

__constant__ float mask_c[MASK_WIDTH][MASK_HEIGHT];

__global__ void convolution_kernel(float* input, float* output, int width, int height){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(row < height && col < width){
        float sum = 0.0f;
        for(int i = 0;i < MASK_WIDTH; i++){
            for(int j = 0; j < MASK_HEIGHT; j++){
                int inrow = row - MASK_RADIUS + i;
                int incol = col - MASK_RADIUS + j;
                if(inrow >= 0 && inrow < height && incol >= 0 && incol < width){
                    sum+= mask_c[i][j]*input[inrow*width+incol]/(MASK_WIDTH*MASK_HEIGHT);
                }
            }
        }
        output[row*width+col] = sum;
    }
}

int main(){
    int width = 10, height= 10;
    int size = sizeof(float)*width*height;
    // Creating a kernel
    float mask[][MASK_HEIGHT] = {{1,1,1}, {1,1,1}, {1,1,1}};

    // Creating dummy input matrix
    float *input = (float*)malloc(size);
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            input[i*width+j] = 1;
        }
        cout << endl;
    }
    float *output = (float*)malloc(size);
    float *dinput, *doutput;
    cudaMalloc(&dinput, size);
    cudaMalloc(&doutput, size);
    cudaMemcpy(dinput, input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_c, mask, sizeof(mask));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE -1)/BLOCK_SIZE, (height + BLOCK_SIZE -1)/BLOCK_SIZE);
    convolution_kernel<<<grid, block>>>(dinput, doutput, width, height);

    cudaMemcpy(output, doutput, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            cout << output[i*width+j] << " ";
        }
        cout << endl;
    }
    return 0;
}