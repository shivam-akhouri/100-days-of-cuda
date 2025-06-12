#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

int main(){
    int length = 10;
    size_t size = length*sizeof(float);
    cublasHandle_t handle;
    float *input;
    // Used for index based return values of cublas functions
    int index;
    // Used for reduction based operation values like sum, product etc.
    float result;
    input = (float*)malloc(size);
    for(unsigned int i = 0;i < length; i++){
        input[i] = rand()%100;
    }
    // Printing input array
    for(unsigned int i = 0;i < length; i++){
        cout << input[i] << " ";
    }
    cout << endl;


    float *da;
    cudaMalloc(&da, size);
    cudaMemcpy(da, input, size, cudaMemcpyHostToDevice);
    cublasCreate(&handle);

    cublasIsamax(handle, length, da, 1, &index);
    cout << "Maximum in the array is: " << input[index-1] << endl;
    
    cublasIsamin(handle, length, da, 1, &index);
    cout << "Manimum in the array is: " << input[index-1] << endl;

    cublasSasum(handle, length, da, 1, &result);
    cout << "Sum of the array element is: " << result << endl; 

    float *dy, *output;
    output = (float*)malloc(size);
    cudaMalloc(&dy, size);
    cudaMemcpy(dy, input, size, cudaMemcpyHostToDevice);
    float alpha = 2;

    cublasSaxpy(handle, length, &alpha, da, 1, dy, 1);
    cudaMemcpy(output, dy, size, cudaMemcpyDeviceToHost);


    cout << "Result of operation alhpa*x+y: "<< endl;
    for(int i  = 0; i < length; i++){
        cout << output[i] << " ";
    }
    cout << endl;

    float c = 0.2, s = 0.9;
    // ((c, s)   --> Rotation matrix
    // (-s, c))
    cublasStatus_t status = cublasSrot(handle, length, da, 1, dy, 1, &c, &s);
    if(status == CUBLAS_STATUS_SUCCESS) cout << "Success" << endl;
    else if(status == CUBLAS_STATUS_EXECUTION_FAILED) cout << "Failed" << endl;
    
    cudaMemcpy(input, da, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, dy, size, cudaMemcpyDeviceToHost);
    cout << "Input array 1 after rotation changes to: " << endl;
    for(int i  = 0; i < length; i++){
        cout << input[i] << " ";
    }
    cout << endl;
    cout << "Input array 2 after rotation changes to: " << endl;
    for(int i  = 0; i < length; i++){
        cout << output[i] << " ";
    }
    cout << endl;

    cublasSscal(handle, length, &alpha, da, 1);
    cudaMemcpy(input, da, size, cudaMemcpyDeviceToHost);
    cout<< "Matrix after applying scaling factor: " << endl;
    for(int i  = 0; i < length; i++){
        cout << input[i] << " ";
    }
    cout << endl;
    cout << "Element is vector 1 before swapping" << endl;
    for(unsigned int i = 0; i < length; i++){
        cout << input[i] << " ";
    }
    cout << endl;
    cout << "Element is vector 2 before swapping" << endl;
    for(unsigned int i = 0; i < length; i++){
        cout << output[i] << " ";
    }
    cout << endl;

    cudaMemcpy(da, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, output, size, cudaMemcpyHostToDevice);
    cublasSswap(handle, length, da, 1, dy, 1);
    cudaMemcpy(input, da, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, dy, size, cudaMemcpyDeviceToHost);

    cout << "Element is vector 1 after swapping" << endl;
    for(unsigned int i = 0; i < length; i++){
        cout << input[i] << " ";
    }
    cout << endl;
    cout << "Element is vector 2 after swapping" << endl;
    for(unsigned int i = 0; i < length; i++){
        cout << output[i] << " ";
    }
    cout << endl;
    
    cublasDestroy(handle);
    cudaFree(da);
    free(input);

    return 0;
}