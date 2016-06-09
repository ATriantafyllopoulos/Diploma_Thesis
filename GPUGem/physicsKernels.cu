#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void animateKernel(float3* positions, float offset)
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

__global__ void meshCreationKernel(float3 *positions, cudaPitchedPtr gridCoordinates, float3 smallestCoords, float d)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float4 *g = (float4 *)gridCoordinates.ptr;
	size_t    pitch = gridCoordinates.pitch;
	size_t    slicePitch = pitch * 10;

	int xPos = (positions[index].x - smallestCoords.x) / d;
	int yPos = (positions[index].y - smallestCoords.y) / d;
	int zPos = (positions[index].z - smallestCoords.z) / d;

	//zPos * 10 * 10 + yPos * 10 + xPos
	int gridIndex = zPos * slicePitch + yPos * pitch + xPos;
	
	g[gridIndex].x = index;
}


void dummyInitialization(float3* positions, const int &numberOfParticles)
{
	int numOfThreads = 512;
	initializeKernel << <(numberOfParticles + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions);
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
	int numOfThreads = 512;
	animateKernel << <(numberOfParticles + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions, offset);
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

void dummyMeshCreation(float3 *positions, cudaPitchedPtr gridCoordinates, float3 smallestCoords, const float &d, const int &numberOfParticles)
{
	int numOfThreads = 512;
	meshCreationKernel << <(numberOfParticles + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions, gridCoordinates, smallestCoords, d);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA engine failed!" << std::endl;
		std::cout << "callback function: dummyMeshCreation" << std::endl;
		std::cout << "Error type: " << cudaStatus << std::endl;
		system("pause"); //for now pause system when an error occurs (only for debug purposes)
	}
}

