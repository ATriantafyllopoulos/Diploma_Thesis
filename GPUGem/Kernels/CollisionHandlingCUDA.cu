#include "auxiliaryKernels.cuh"

/*
This should be wrapped in a class. This is only for primitive - primitive collision and not generic.
Parameters include k, d, h. They should be given as input.
*/
__global__ void handleCollisionsKernel(Primitive *leafNodes, const float timeStep, const int numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfPrimitives)
		return;
	//these should not be hardcoded
	float k = 0.1;
	float d = 0.1;
	float h = 0.1;
	float kt = 0.1;
	Primitive *current = leafNodes + index; 
	while (current->collisionCounter)
	{
		Primitive *colliding = current->collisions[current->collisionCounter];
		float3 distance = current->centroid - colliding->centroid;
		
		float distanceNorm = norm(distance);

		float3 relativeVelocity = current->linearMomentum / current->mass - colliding->linearMomentum / colliding->mass;
		
		float3 repulsiveForce = distance * (-k * (d - distanceNorm) / distanceNorm);

		float3 dampingForce = relativeVelocity * h;
		float3 tangentialVelocity = relativeVelocity - distance * (dot(relativeVelocity, distance) / distanceNorm) / distanceNorm;
		float3 tangentialForce = tangentialVelocity * kt;

		float3 totalForce = repulsiveForce + dampingForce + tangentialForce;

		current->linearMomentum = current->linearMomentum + totalForce * timeStep;
		colliding->linearMomentum = colliding->linearMomentum + totalForce * timeStep;
		current->collisionCounter--;
	}
}

cudaError_t handleCollisions(Primitive *leafNodes, const float &timeStep, const int &numberOfPrimitives, const int numberOfThreads)
{
	const int numberOfBlocks = (numberOfPrimitives + numberOfThreads - 1) / numberOfThreads;
	handleCollisionsKernel << < numberOfBlocks, numberOfThreads >> >(leafNodes, timeStep, numberOfPrimitives);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return cudaSuccess;
	cudaStatus = cudaDeviceSynchronize();
	return cudaStatus;
}