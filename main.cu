#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixmultiply(float* matrixA, float* matrixB, float* matrixResult, int width)
{
    int size = width * width * sizeof(float);
    float* matrixA_d;
    float* matrixB_d;
    float* matrixResult_d;

    float resultValue = 0;

    for (int i = 0; i < width; ++i)
    {
        float Melement = matrixA[blockIdx.x * width + threadIdx.x];
        float Nelement = matrixB[blockIdx.y * width + threadIdx.y];

        resultValue += Melement + Nelement;
    }

    matrixResult_d[blockIdx.y * width + threadIdx.x] = resultValue;

    cudaMalloc((void**) matrixA_d, size);
    cudaMalloc((void**) matrixB_d, size);
    cudaMalloc((void**) matrixResult_d, size);

    cudaMemcpy(matrixA_d, matrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_d, matrixB, size, cudaMemcpyHostToDevice);
}


int main() 
{
    float* matrixA_p;
    float* matrixB_p;
    float* resultMatrix_p;

    float matrixA[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float matrixB[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float resultMatrix[16];

    int width = 32;
    int matrixSize = width * width * sizeof(float);

    dim3 dimBlock(width,width);
    dim3 dimGrid(1,1);

    matrixmultiply<<<dimGrid, dimBlock>>>();
    
    cudaDeviceSynchronize();
    return 0;
}