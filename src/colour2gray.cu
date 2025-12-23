#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <fstream>


__global__ void col2gray(float* img, int width, int height)
{


	// Gray Pixel = 0.21*R + 0.72*G + 0.07*B
}


int main()
{
	std::cout << "Hello, world\n";

	cv::Mat colourImg = cv::imread("sample.jpg");

	std::cout << "Width = " << colourImg.cols << ", Height = " << colourImg.rows << ", Channels = " << colourImg.channels() << std::endl;
	int width = colourImg.cols;
	int height = colourImg.rows;

	int tileWidth = 16;

	dim3 blockDim(tileWidth, tileWidth); //256 threads total
	dim3 gridDim( static_cast<unsigned int>(std::ceil(width/(float)tileWidth))
				, static_cast<unsigned int>(std::ceil(height/(float)tileWidth)) );

	cv::Mat colourImgCopy = colourImg.clone();
	cv::Mat flat = colourImgCopy.reshape((width*height),1);

	// col2gray<<<gridDim, blockDim>>>(flat, width, height);

}

/*
	TODO
	-check if the way im flattening the img matrix preserves the three channels.
	-convert cv::Mat into float*
*/