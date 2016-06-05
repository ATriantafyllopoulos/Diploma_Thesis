#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
__global__ void animateKernel(float3* positions, double offset)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	positions[index].x += offset;
}

__global__ void initializeKernel(float3* positions)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = threadIdx.x;
	int j = blockIdx.x;
	positions[index].x = (float)i;
	positions[index].y = (float)j;
	positions[index].z = -10.0;
}

void dummyInitialization(float3* positions, const int &numberOfParticles)
{
	initializeKernel << <(numberOfParticles + 1023) / 1024, 1024 >> >(positions);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA engine failed!" << std::endl;
		std::cout << "callback function: dummyInitialization" << std::endl;
		std::cout << "Error type: " << cudaStatus << std::endl;
		system("pause"); //for now pause system when an error occurs (only for debug purposes)
	}
}

void dummyAnimation(float3* positions, const double &offset, const int &numberOfParticles)
{
	animateKernel << <(numberOfParticles + 1023) / 1024, 1024 >> >(positions, offset);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA engine failed!" << std::endl;
		std::cout << "callback function: dummyAnimation" << std::endl;
		std::cout << "Error type: " << cudaStatus << std::endl;
		system("pause"); //for now pause system when an error occurs (only for debug purposes)
	}
}

