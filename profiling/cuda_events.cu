#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixmul(float* a, float* b, float* c, int m, int n, int k){
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;
        for(int i = 0; i < k; i++){
            sum+= a[i+row*k]*b[col+ i*n];
        }
        c[row*n+col] = sum;
    }
}

int main(){
    int m = 4096, k = 3072, n = 4096;
    float *a, *b, *c;
    float *da, *db, *dc;
    a = (float*)malloc(sizeof(float)*m*k);
    b = (float*)malloc(sizeof(float)*k*n);
    c = (float*)malloc(sizeof(float)*m*n);
    float counter = 1.0f;
    for(int i = 0; i< m; i++){
        for(int j = 0; j < k ;j++){
            a[i*k+j] = 0.1*counter;
            counter++;
        }
    }
    counter = 7;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < n; j++){
            b[i*n+j] = 0.1*counter;
            counter++;
        }
    }

    cudaMalloc(&da, sizeof(float)*m*k);
    cudaMalloc(&db, sizeof(float)*k*n);
    cudaMalloc(&dc, sizeof(float)*m*n);
    cudaMemcpy(da, a, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    dim3 block(128, 128);
    dim3 grid((n+31)/32, (m+31)/32);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    matrixmul<<<grid, block>>>(da, db, dc, m, n, k);
    cudaEventRecord(end);

    float duration = 0.0f;
    cudaEventElapsedTime(&duration, start, end);

    cudaMemcpy(c, dc, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // for(int i =0; i < m; i++){
    //     for(int j = 0; j < n; j++){
    //         cout << c[i*n+j] << " ";
    //     }
    //     cout << endl;
    // }
    cout << "Naive Matrix Multipication took: " << duration << " miliseconds" << endl;
    free(a);
    free(b);
    free(c);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}