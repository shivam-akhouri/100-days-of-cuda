#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
using namespace std;

__global__ void tiled_matrix_mul(float* a, float* b, float* c, int m, int n, int k){
    __shared__ float ta[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tb[BLOCK_SIZE][BLOCK_SIZE];
    float temp = 0;

    int row = blockDim.y *blockIdx.y + threadIdx.y;
    int col = blockDim.x *blockIdx.x + threadIdx.x;

    for(int i = 0; i < (BLOCK_SIZE-1+k)/BLOCK_SIZE; i++){
        if(i*BLOCK_SIZE+threadIdx.x < k && row < m){
            ta[threadIdx.y][threadIdx.x] = a[row*k + i*BLOCK_SIZE + threadIdx.x];
        }else{
            ta[threadIdx.y][threadIdx.x] = 0;
        }

        if(i*BLOCK_SIZE+ threadIdx.y < k && col < n){
            tb[threadIdx.y][threadIdx.x] = b[(i*BLOCK_SIZE+threadIdx.y)*n + col];
        }else{
            tb[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();
        for(int j =0; j < BLOCK_SIZE; j++){
            temp+=ta[threadIdx.y][j]*tb[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < m && col < n){
        c[row*n+col] = temp;
    }

}

int main(){
    int m = 3, k = 2, n = 4;
    float *a, *b, *c;
    float *da, *db, *dc;
    a = (float*)malloc(m*k*sizeof(float));
    b = (float*)malloc(k*n*sizeof(float));
    c = (float*)malloc(m*n*sizeof(float));
    int counter = 1;
    for(int i = 0; i< m; i++){
        for(int j = 0; j < k ;j++){
            a[i*k+j] = counter;
            counter++;
        }
    }
    counter = 7;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < n; j++){
            b[i*n+j] = counter;
            counter++;
        }
    }
    cudaMalloc(&da, sizeof(float)*m*k);
    cudaMalloc(&db, sizeof(float)*k*n);
    cudaMalloc(&dc, sizeof(float)*m*n);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+BLOCK_SIZE-1)/BLOCK_SIZE, (m+BLOCK_SIZE-1)/BLOCK_SIZE);

    cudaMemcpy(da, a, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, sizeof(float)*m*k, cudaMemcpyHostToDevice);

    tiled_matrix_mul<<<grid, block>>>(da, db, dc, m, n, k);
    cudaMemcpy(c, dc, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i =0; i < m; i++){
        for(int j = 0; j < n; j++){
            cout << c[i*n+j] << " ";
        }
        cout << endl;
    }
    
    return 0;
}