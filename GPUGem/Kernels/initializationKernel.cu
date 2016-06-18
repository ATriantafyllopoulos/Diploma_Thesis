#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void initializeKernel(float3* positions, float3* linearMomenta)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = threadIdx.x;
	int j = blockIdx.x;

	positions[index] = make_float3((float)i, (float)j, -10.f);

	//particles are initially static
	linearMomenta[index] = make_float3(0.f, 0.f, 0.f);
}

/*
Use double pointer because linearMomenta is stored in Physics Engine Class, and the allocation must hold for all subsequent calls.
If we used a 'local' allocation as in cudaMalloc((void**)&linearMomenta,...) we would not be able to access this variable elsewhere. 
*/
cudaError_t initialization(float3* positions, float3** linearMomenta, const int &numberOfPrimitives, const int &numberOfThreads)
{
	cudaError_t cudaStatus = cudaMalloc((void**)linearMomenta, numberOfPrimitives * sizeof(float3));
	if (cudaStatus != cudaSuccess)
		return cudaStatus;

	//if allocation was successful, launch initialization kernel
	initializeKernel << <(numberOfPrimitives + numberOfThreads - 1) / numberOfThreads, numberOfThreads >> >(positions, *linearMomenta);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	return cudaStatus;
}