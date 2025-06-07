#include <stdio.h>


__global__ void helloWorld(){
    printf("Hello world");
}

__global__ void vectorAdd(int* a, int* b, int* c){
    int i = threadIdx.x;
    c[i] = a[i]+b[i];
}

int main(){
    int a[] = {1,2,3,4};
    int b[] = {1,2,3,4};
    int c[] = {0,0,0,0};
    vectorAdd<<<1,4>>>(a,b,c);
    helloWorld<<<1,1>>>();
    for(int i = 0;i < 4; i++){
        printf("%d \n", c[i]);
    }
    return 0;
}