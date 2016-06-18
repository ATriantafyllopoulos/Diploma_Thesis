#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Primitives.h"
#include "cudaAuxiliary.h"

__global__ void updateStateVector(Primitive* leafNodes, const float3 &gravityVector, const float &timeStep,  const int &numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfPrimitives) 
		return;

	Primitive *current = leafNodes + index;
	
	//gravity effect
	current->linearMomentum = current->linearMomentum + gravityVector * current->mass * timeStep;
	current->centroid = current->centroid + current->linearMomentum / current->mass * timeStep;
}

cudaError_t update(Primitive* leafNodes, const float &timeStep, const int &numberOfPrimitives, const int &numberOfThreads)
{
	float3 gravityVector = make_float3(0.f, -9.81, 0.f);
	updateStateVector << <(numberOfPrimitives + numberOfThreads - 1) / numberOfThreads, numberOfThreads >> >(leafNodes, gravityVector, timeStep, numberOfPrimitives);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	return cudaStatus;
}