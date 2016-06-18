#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Primitives.h"
#include <algorithm>
#include <iostream>

#include "cudaAuxiliary.h"

__device__ float3 operator+(const float3 &a, const float3 &b)
{

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ float3 operator-(const float3 &a, const float3 &b)
{

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

__device__ float3 operator*(const float3 &a, const float &b)
{

	return make_float3(a.x * b, a.y * b, a.z  * b);

}

__device__ float3 operator/(const float3 &a, const float &b)
{

	return b != 0 ? make_float3(a.x / b, a.y / b, a.z / b) : make_float3(0, 0, 0);

}

__device__ float norm(const float3 &a)
{
	return __fsqrt_rd(
		a.x * a.x +
		a.y * a.y +
		a.z * a.z);
}

__device__ float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/*
This should be wrapped in a class. This is only for primitive - primitive collision and not generic.
Parameters include k, d, h. They should be given as input.
*/
__global__ void handleCollisions(Primitive *leafNodes, int numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfPrimitives)
		return;
	float k = 0.1;
	float d = 0.1;
	float h = 0.1;
	float kt = 0.1;
	float dt = 0.1;
	Primitive *current = leafNodes + index; 
	while (current->collisionCounter)
	{
		Primitive *colliding = current->collisions[current->collisionCounter];
		float3 distance = current->centroid - colliding->centroid;
		
		float distanceNorm = norm(distance);

		float3 relativeVelocity = current->linearMomentum / current->mass - colliding->linearMomentum / colliding->mass;
		
		float3 repulsiveForce = distance * (-k * (d - distanceNorm) / distanceNorm);
		//float3 repulsiveForce = distance * (-k * (d - distanceNorm) / distanceNorm);

		float3 dampingForce = relativeVelocity * h;
		float3 tangentialVelocity = relativeVelocity - distance * (dot(relativeVelocity, distance) / distanceNorm) / distanceNorm;
		float3 tangentialForce = tangentialVelocity * kt;

		float3 totalForce = repulsiveForce + dampingForce + tangentialForce;

		current->linearMomentum = current->linearMomentum + totalForce * dt;
		colliding->linearMomentum = colliding->linearMomentum + totalForce * dt;
		current->collisionCounter--;
	}
}