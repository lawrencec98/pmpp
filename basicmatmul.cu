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


void DisplayMatrix(float* matrix, int width)
{
    for (std::size_t i = 0; i < width*width; ++i)
    {
        std::cout << matrix[i] << " ";
        if ((i + 1) % width == 0)
        {
            std::cout << '\n';
        }
    }
}


int main() 
{
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

    float *matrixAd, *matrixBd, *resultMatrixd;

    int width = 4;
    int matrixSize = width * width * sizeof(float);

    dim3 dimBlock(16,16);
    dim3 dimGrid((width+15)/16 , (width+15)/16);

    cudaMalloc(&matrixAd, matrixSize);
    cudaMalloc(&matrixBd, matrixSize);
    cudaMalloc(&resultMatrixd, matrixSize);

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