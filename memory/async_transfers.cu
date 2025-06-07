#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

using namespace std;

void init_vector(float* arr, int length, bool flag=false){
    for(int i = 0;i < length; i++){
        arr[i] = i;
        if(flag) arr[i] = 0;
    }
}

__global__ void vectoradd(float *a, float *b, float *c, int length){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < length) c[idx] = a[idx]+b[idx];
}

int main(){
    int length = 100000;
    float *a, *b, *c;
    float *da, *db, *dc;

    a = (float*)malloc(sizeof(float)*length);
    b = (float*)malloc(sizeof(float)*length);
    c = (float*)malloc(sizeof(float)*length);
    cudaMalloc(&da, sizeof(float)*length);
    cudaMalloc(&db, sizeof(float)*length);
    cudaMalloc(&dc, sizeof(float)*length);

    init_vector(a, length);    
    init_vector(b, length);    
    init_vector(c, length, true);
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);


    // Both A and B will be transferred at the same time and not in sequential order.
    cudaEventRecord(start);
    cudaMemcpyAsync(da, a, sizeof(float)*length, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(db, b, sizeof(float)*length, cudaMemcpyHostToDevice, stream2);
    cudaEventRecord(end);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    dim3 block(BLOCK_SIZE, 1,1);
    dim3 grid((length+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    vectoradd<<<grid, block>>>(da, db, dc, length);

    cudaMemcpyAsync(c, dc, sizeof(float)*length, cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);
    
    float duration;
    cudaEventElapsedTime(&duration, start, end);
    cout << "Copy command took: " << duration << " milliseconds" << endl;
    // for(int i =0; i < length; i++){
    //     cout << c[i] << " ";
    // }
    return 0;
}