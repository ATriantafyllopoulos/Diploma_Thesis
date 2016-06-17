#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
cudaError_t cudaFail(cudaError_t cudaStatus, char *funcName);

__global__ void animateKernel(float3* positions, float offset)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	positions[index].x += offset;
}

__global__ void initializeKernel(float3* positions, float3* linearMomenta)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = threadIdx.x;
	int j = blockIdx.x;

	positions[index] = make_float3((float)i, (float)j, -10.f);

	//particles are initially static
	linearMomenta[index] = make_float3(0.f, 0.f, 0.f);
}

cudaError_t dummyInitialization(float3* positions, float3* linearMomenta, const int &numberOfPrimitives)
{
	int numOfThreads = 512;
	cudaError_t cudaStatus = cudaMalloc((void**)&linearMomenta, numberOfPrimitives * sizeof(float3));
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(linearMomenta);
		return cudaFail(cudaStatus, "dummyInitialization_malloc");
	}
	initializeKernel << <(numberOfPrimitives + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions, linearMomenta);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(linearMomenta);
		return cudaFail(cudaStatus, "dummyInitialization");
	}
	return cudaSuccess;
}

cudaError_t dummyAnimation(float3* positions, const double &offset, const int &numberOfPrimitives)
{
	int numOfThreads = 512;
	animateKernel << <(numberOfPrimitives + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions, offset);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "dummyAnimation");
	//std::cout << "CUDA animation was successful." << std::endl;
	return cudaSuccess;
}

cudaError_t cudaFail(cudaError_t cudaStatus, char *funcName)
{
	std::cout << "CUDA engine failed!" << std::endl;
	std::cout << "callback function:" << funcName << std::endl;
	std::cout << "Error code: " << cudaStatus << std::endl;
	std::cout << "Error type: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	std::cout << "Enter random character to continue..." << std::endl;
	int x;
	std::cin >> x;
	exit(1);
	return cudaStatus;
}