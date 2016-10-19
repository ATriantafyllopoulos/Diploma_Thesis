#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include "BVHcreation.h"
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"

#include "particleSystem.cuh"

#ifndef BVHAUXILIARY_CUH
#define BVHAUXILIARY_CUH
inline __device__ int intMin(int i, int j) {
	return (i > j) ? j : i;
}

inline __device__ int intMax(int i, int j) {
	return (i > j) ? i : j;
}

/**
* Longest common prefix for morton code
*/
inline __device__ int longestCommonPrefix(int i, int j, int len, unsigned int *sortedMortonCodes) {
	if (0 <= j && j < len) {
		unsigned int codeI = sortedMortonCodes[i];
		unsigned int codeJ = sortedMortonCodes[j];
		if (codeI != codeJ) //if res != 0 then the Morton codes are not identical
			return __clz(codeI ^ codeJ);
		else //Morton codes are identical
			return __clz(i^j) + 32;
	}
	else {
		return -1;
	}
}

inline __device__ void initBound(Sphere *targ, float4 &c, float &r)
{
	//targ->cm = c;
	//targ->radius = r;
	targ->augmentedCM = make_float4(c.x, c.y, c.z, r);
}


inline __device__ void initBound(AABB *targ, float4 &c, float &r)
{
	//making a single memory load to increase memory efficiency
	float x = c.x;
	float y = c.y;
	float z = c.z;

	targ->min.x = x - r;
	targ->min.y = y - r;
	targ->min.z = z - r;

	targ->max.x = x + r;
	targ->max.y = y + r;
	targ->max.z = z + r;

}

inline __device__ void mergeBounds(Sphere *targ, Sphere &b1, Sphere &b2)
{
	/*float4 cm1 = b1.cm, cm2 = b2.cm;
	float4 newCM = (cm1 + cm2) / 2;*/
	/*targ->cm = newCM;
	targ->radius = max(length(newCM - cm1) + b1.radius, length(newCM - cm2) + b2.radius);*/
	float3 cm1 = make_float3(b1.augmentedCM), cm2 = make_float3(b2.augmentedCM);
	float3 newCM = (cm1 + cm2) / 2;
	float r1 = b1.augmentedCM.w, r2 = b2.augmentedCM.w;
	float radius = max(length(newCM - cm1) + r1, length(newCM - cm2) + r2);
	targ->augmentedCM = make_float4(newCM.x, newCM.y, newCM.z, radius);
}

inline __device__ void mergeBounds(AABB *targ, AABB &a, AABB &b)
{

	targ->min.x = a.min.x < b.min.x ? a.min.x : b.min.x;
	targ->min.y = a.min.y < b.min.y ? a.min.y : b.min.y;
	targ->min.z = a.min.z < b.min.z ? a.min.z : b.min.z;

	targ->max.x = a.max.x > b.max.x ? a.max.x : b.max.x;
	targ->max.y = a.max.y > b.max.y ? a.max.y : b.max.y;
	targ->max.z = a.max.z > b.max.z ? a.max.z : b.max.z;

}

inline __device__
float3 collideSpheresBVH(float4 posA, float4 posB,
float3 velA, float3 velB,
float radiusA, float radiusB,
SimParams params)
{
	// calculate relative position
	float3 relPos = make_float3(posB - posA);

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);


	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = velB - velA;
		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		// attraction
		force += params.attraction*relPos;
	}
	return force;
}

inline __device__ bool checkOverlap(Sphere &b1, Sphere &b2)
{
	//return length(b1.cm - b2.cm) <= b1.radius + b2.radius;
	return length(make_float3(b1.augmentedCM) - make_float3(b2.augmentedCM)) <= b1.augmentedCM.w + b2.augmentedCM.w;
}


inline __device__ bool checkOverlap(AABB &a, AABB &b)
{
	/*float4 minB1 = b1.minPoint;
	float4 minB2 = b2.minPoint;
	float4 maxB1 = b1.maxPoint;
	float4 maxB2 = b2.maxPoint;

	return (minB1.x <= maxB2.x && maxB1.x >= minB2.x) &&
	(minB1.y <= maxB2.y && maxB1.y >= minB2.y) &&
	(minB1.z <= maxB2.z && maxB1.z >= minB2.z);*/
	return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
		(a.min.y <= b.max.y && a.max.y >= b.min.y) &&
		(a.min.z <= b.max.z && a.max.z >= b.min.z);
}

inline __device__ bool checkOverlap(float4 &pos, Sphere &b, float virtualRadius)
{
	return length(make_float3(pos) - make_float3(b.augmentedCM)) <= b.augmentedCM.w + virtualRadius;
}

inline __device__ bool checkOverlap(float4 &pos, AABB &b, float virtualRadius)
{

	float x = pos.x;
	float y = pos.y;
	float z = pos.z;
	// get box closest point to sphere center by clamping
	float xx = max(b.min.x, min(x, b.max.x));
	float yy = max(b.min.y, min(y, b.max.y));
	float zz = max(b.min.z, min(z, b.max.z));

	// this is the same as isPointInsideSphere
	float distance = __fsqrt_rz((xx - x) * (xx - x) +
		(yy - y) * (yy - y) +
		(zz - z) * (zz - z));

	return distance < virtualRadius;
}

#endif
