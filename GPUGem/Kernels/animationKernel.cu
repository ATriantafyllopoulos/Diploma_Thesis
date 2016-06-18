#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void animateKernel(float3* positions, float offset)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	positions[index].x += offset;
}

cudaError_t animation(float3* positions, const double &offset, const int &numberOfPrimitives, const int &numberOfThreads)
{
	animateKernel << <(numberOfPrimitives + numberOfThreads - 1) / numberOfThreads, numberOfThreads >> >(positions, offset);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	return cudaStatus;
}