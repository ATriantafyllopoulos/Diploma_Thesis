#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void animateKernel(float3* positions, double offset)
{
	int i = threadIdx.y * blockDim.x + threadIdx.x;
	int j = threadIdx.y;
	positions[i].x += offset;
	positions[i].y = (float)j;
	positions[i].z = -10.0;
}

__global__ void initializeKernel(float3* positions)
{
	int i = threadIdx.y * blockDim.x + threadIdx.x;
	int j = threadIdx.y;
	positions[i].x = (float)i;
	positions[i].y = (float)j;
	positions[i].z = -10.0;
}

void dummyInitialization(float3* positions)
{
	initializeKernel << <2, 512 >> >(positions);
}

void dummyAnimation(float3* positions, double offset)
{
	animateKernel << <2, 512 >> >(positions, offset);
}