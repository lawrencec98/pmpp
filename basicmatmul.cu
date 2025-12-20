#include <cstdio>

#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void MatrixMultiply(float* matrixA, float* matrixB, float* matrixResult, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float resultValue = 0;

    if (row < width && col < width)
    {
        for (int i = 0; i < width; ++i)
        {
            float Melement = matrixA[row*width + i];
            float Nelement = matrixB[i*width + col];

            resultValue += Melement * Nelement;
        }

        matrixResult[row*width + col] = resultValue;
    }

}


void InitializeMatrixCPU(float* matrix, int width)
{
    for (int i = 0; i < width*width; ++i)
    {
        matrix[i] = i;
    }
}


__global__ void InitializeMatrixGPU(float* matrix, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width)
    {
        matrix[row * width + col] = row*width + col;
    }
}


void DisplayMatrix(float* matrix, int width)
{
    for (std::size_t i = 0; i < width*width; ++i)
    {
        std::cout << matrix[i] << '\t';
        if ((i + 1) % width == 0)
        {
            std::cout << '\n';
        }
    }
}


int main() 
{
    int width = 4;

    // Declare host matrices
    float matrixA[width*width];
    float matrixB[width*width];
    float resultMatrix[width*width];

    // Cuda grid configuration
    dim3 dimBlock(4,4);
    dim3 dimGrid((width+15)/16 , (width+15)/16);

    // Declare device matrices
    float *matrixAd, *matrixBd, *resultMatrixd;

    int matrixSize = width * width * sizeof(float);

    // Allocate memory on device
    cudaMalloc(&matrixAd, matrixSize);
    cudaMalloc(&matrixBd, matrixSize);
    cudaMalloc(&resultMatrixd, matrixSize);

    // Initialize matrices using CPU
    // InitializeMatrixCPU(matrixA, width);
    // InitializeMatrixCPU(matrixB, width);

    // Initialize matrices using GPU
    InitializeMatrixGPU<<<dimGrid, dimBlock>>>(matrixAd, width);
    InitializeMatrixGPU<<<dimGrid, dimBlock>>>(matrixBd, width);
    cudaDeviceSynchronize();
    cudaMemcpy(matrixA, matrixAd, matrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixB, matrixBd, matrixSize, cudaMemcpyDeviceToHost);

    cudaMemcpy(matrixAd, matrixA, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixBd, matrixB, matrixSize, cudaMemcpyHostToDevice);

    MatrixMultiply<<<dimGrid, dimBlock>>>(matrixAd, matrixBd, resultMatrixd, width);

    cudaDeviceSynchronize();

    cudaMemcpy(resultMatrix, resultMatrixd, matrixSize, cudaMemcpyDeviceToHost);

    DisplayMatrix(resultMatrix, width);
    
    cudaFree(matrixAd);
    cudaFree(matrixBd);
    cudaFree(resultMatrixd);

    return 0;
}