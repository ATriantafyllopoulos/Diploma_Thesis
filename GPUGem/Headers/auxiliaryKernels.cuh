#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Primitives.h"

/*__device__ inline float3 operator+(const float3 &a, const float3 &b);
__device__ inline float3 operator-(const float3 &a, const float3 &b);
__device__ inline float3 operator*(const float3 &a, const float &b);
__device__ inline float3 operator/(const float3 &a, const float &b);
__device__ inline float norm(const float3 &a);
__device__ inline float dot(const float3 &a, const float3 &b);*/
__device__ inline float3 operator+(const float3 &a, const float3 &b)
{

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ inline float3 operator-(const float3 &a, const float3 &b)
{

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

__device__ inline float3 operator*(const float3 &a, const float &b)
{

	return make_float3(a.x * b, a.y * b, a.z  * b);

}

__device__ inline float3 operator/(const float3 &a, const float &b)
{

	return b != 0 ? make_float3(a.x / b, a.y / b, a.z / b) : make_float3(0, 0, 0);

}

__device__ inline float norm(const float3 &a)
{
	return __fsqrt_rd(
		a.x * a.x +
		a.y * a.y +
		a.z * a.z);
}

__device__ inline float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float MIN(float x, float y)
{
	return x < y ? x : y;
}

__device__ inline float MAX(float x, float y)
{
	return x > y ? x : y;
}

__device__ inline bool checkOverlap(Primitive *query, Primitive *node)
{
	float dist = __fsqrt_rd((node->centroid.x - query->centroid.x) * (node->centroid.x - query->centroid.x) +
		(node->centroid.y - query->centroid.y) * (node->centroid.y - query->centroid.y) +
		(node->centroid.z - query->centroid.z) * (node->centroid.z - query->centroid.z));
	return dist < node->radius + query->radius;
}