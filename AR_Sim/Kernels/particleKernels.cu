/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* CUDA particle system kernel code.
*/

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <helper_cuda.h>
#include <math.h>
#include "helper_math.h"
#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "math_constants.h"
#include "particleSystem.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;

void setParameters(SimParams *hostParams)
{
	// copy parameters to constant memory
	checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
}

struct integrate_functor
{
	float deltaTime;
	float3 minPos, maxPos;
	__host__ __device__
		integrate_functor(float delta_time, float3 min_pos, float3 max_pos) : deltaTime(delta_time), minPos(min_pos), maxPos(max_pos) {}

	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
		volatile float4 posData = thrust::get<0>(t);
		volatile float4 velData = thrust::get<1>(t);

		volatile int rigidBodyIndex = thrust::get<2>(t);
		if (rigidBodyIndex >= 0) return; //if this is a virtual particle associated with a rigid body
		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);

		vel += params.gravity * deltaTime;
		vel *= params.globalDamping;

		// new position = old position + velocity * deltaTime
		pos += vel * deltaTime;

		// set this to zero to disable collisions with cube sides
#if 1
		//add a 1cm offset to prevent false collisions
		maxPos.x = maxPos.x + 0.01;
		maxPos.y = maxPos.y + 0.01;
		maxPos.z = maxPos.z + 0.01;

		minPos.x = minPos.x - 0.01;
		minPos.y = minPos.y - 0.01;
		minPos.z = minPos.z - 0.01;

		if (pos.x > maxPos.x - params.particleRadius)
		{
			pos.x = maxPos.x - params.particleRadius;
			vel.x *= params.boundaryDamping;
		}

		if (pos.x < minPos.x + params.particleRadius)
		{
			pos.x = minPos.x + params.particleRadius;
			vel.x *= params.boundaryDamping;
		}

		if (pos.y > maxPos.y - params.particleRadius && vel.y > 0)
		{
			pos.y = maxPos.y - params.particleRadius;
			vel.y *= params.boundaryDamping;
		}

		if (pos.z > maxPos.z - params.particleRadius)
		{
			pos.z = maxPos.z - params.particleRadius;
			vel.z *= params.boundaryDamping;
		}

		if (pos.z < minPos.z + params.particleRadius)
		{
			pos.z = minPos.z + params.particleRadius;
			vel.z *= params.boundaryDamping;
		}

#endif

		if (pos.y < minPos.y + params.particleRadius)
		{
			pos.y = minPos.y + params.particleRadius;
			vel.y *= params.boundaryDamping;
		}

		// store new position and velocity
		thrust::get<0>(t) = make_float4(pos, posData.w);
		thrust::get<1>(t) = make_float4(vel, velData.w);
	}
};


void integrateSystem(float *pos,
	float *vel,
	float deltaTime,
	float3 minPos,
	float3 maxPos,
	int *rigidBodyIndices,
	uint numParticles)
{
	thrust::device_ptr<float4> d_pos4((float4 *)pos);
	thrust::device_ptr<float4> d_vel4((float4 *)vel);

	thrust::device_ptr<int> d_index((int *)rigidBodyIndices);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_index)),
		thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles, d_index + numParticles)),
		integrate_functor(deltaTime, minPos, maxPos));
}

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.gridSize.y - 1);
	gridPos.z = gridPos.z & (params.gridSize.z - 1);
	return gridPos.z * params.gridSize.y * params.gridSize.x + gridPos.y * params.gridSize.x + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
uint   *gridParticleIndex, // output
float4 *pos,               // input: positions
uint    numParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	volatile float4 p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(int *rbIndices, //index of the rigid body each particle belongs to
uint   *cellStart,        // output: cell start index
uint   *cellEnd,          // output: cell end index
float4 *sortedPos,        // output: sorted positions
float4 *sortedVel,        // output: sorted velocities
uint   *gridParticleHash, // input: sorted grid hashes
uint   *gridParticleIndex,// input: sorted particle indices
float4 *oldPos,           // input: sorted position array
float4 *oldVel,           // input: sorted velocity array
uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
		float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh

		sortedPos[index] = pos;
		sortedVel[index] = vel;
	}
}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
float3 velA, float3 velB,
float radiusA, float radiusB,
float attraction,
int *numCollisions)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		if (numCollisions)
			(*numCollisions)++;
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
		force += attraction*relPos;
	}

	return force;
}

// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
uint    index,
float3  pos,
float3  vel,
float4 *oldPos,
float4 *oldVel,
uint   *cellStart,
uint   *cellEnd,
int *rbIndices, //index of the rigid body each particle belongs to
uint *gridParticleIndex,//sorted particle indices
int rigidBodyIndex, //rigid body index corresponding to current particle
int *numCollisions)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			if (j != index && (rigidBodyIndex != rbIndices[gridParticleIndex[j]] || rigidBodyIndex == -1))// check not colliding with self and not of the same rigid body
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				float3 vel2 = make_float3(FETCH(oldVel, j));

				// collide two spheres
				force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction, numCollisions);
			}
		}
	}

	return force;
}

__global__
void collideD(float4 *pForce, //total force applied to rigid body - per particle
int *rbIndices, //index of the rigid body each particle belongs to
float4 *relativePos, //particle's relative position
float4 *pTorque,  //rigid body angular momentum - per particle
float4 *color,
float4 *newVel,               // output: new velocity
float4 *oldPos,               // input: sorted positions
float4 *oldVel,               // input: sorted velocities
uint   *gridParticleIndex,    // input: sorted particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(oldVel, index));

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);
	float3 torque = make_float3(0.0f);
	int numCollisions = 0;
	uint originalIndex = gridParticleIndex[index];
	int rigidBodyIndex = rbIndices[originalIndex];
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				float3 localForce= collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd, rbIndices, gridParticleIndex, rigidBodyIndex, &numCollisions);
				force += localForce;
				torque += cross(make_float3(relativePos[originalIndex]), localForce);
			}
		}
	}

	// write new velocity back to original unsorted location
	

	if (rigidBodyIndex == -1)
		newVel[originalIndex] = make_float4(vel + force, 0.0f);
	else
	{
//		rbForces[rigidBodyIndex] += make_float4(force, 0.0f);
//		rbTorque[rigidBodyIndex] += make_float4(torque, 0);
		pForce[originalIndex] = make_float4(force, 0.0f);
		pTorque[originalIndex] = make_float4(torque, 0);
	}

	if (numCollisions)
		color[originalIndex] = make_float4(1, 0, 0, 0);
	else
		color[originalIndex] = make_float4(0, 0, 1, 0);
}

__device__
float3 staticCollideSpheres(float3 posA, float3 posB,
float3 velA,
float radiusA, float radiusB,
float attraction, int *numCollisions)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = -velA;

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		// attraction
		force += attraction*relPos;
		*numCollisions++;
	}

	return force;
	//return - 2 * velA;
}

// collide a particle against all other particles in a given cell
__device__
float3 staticCollideCell(int3    gridPos,
uint    index,
float3  pos,
float3  vel,
float4 *oldPos,
float4 *oldVel,
float4 *staticPos,
float *r_radii, //radii of all scene particles
uint   *cellStart,
uint   *cellEnd,
int *numCollisions)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j < endIndex; j++)
		{
			float3 pos2 = make_float3(FETCH(staticPos, j));
			// collide two spheres
			force += staticCollideSpheres(pos, pos2, vel, params.particleRadius, r_radii[j], params.attraction, numCollisions);
		}
	}

	return force;
}

__global__
void staticCollideD(
		float4 *dCol,
		float4 *pForces, //total force applied to rigid body
		int *rbIndices, //index of the rigid body each particle belongs to
		float4 *relativePos, //particle's relative position
		float4 *pTorque,  //rigid body angular momentum
		float *r_radii, //radii of all scene particles
		float4 *newVel,               // output: new velocity
		float4 *oldPos,               // input: sorted positions
		float4 *oldVel,               // input: sorted velocities
		float4 *staticPos,
		uint   *gridParticleIndex,    // input: sorted particle indices
		uint   *cellStart,
		uint   *cellEnd,
		uint    numParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(oldVel, index));

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);
	float3 torque = make_float3(0.0f);
	uint originalIndex = gridParticleIndex[index];
	int rigidBodyIndex = rbIndices[originalIndex];
	int numCollisions = 0;
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				float3 localForce = staticCollideCell(neighbourPos,
						index,
						pos,
						vel,
						oldPos,
						oldVel,
						staticPos,
						r_radii,
						cellStart,
						cellEnd,
						&numCollisions);
				force += localForce;
				torque += cross(make_float3(relativePos[originalIndex]), localForce);
			}
		}
	}
	if (numCollisions)
		dCol[originalIndex] = make_float4(0, 1, 0, 0);
	if (rigidBodyIndex == -1)
		newVel[originalIndex] += make_float4(force, 0.0f);
	else
	{
		pForces[originalIndex] = make_float4(force, 0.0f);
		pTorque[originalIndex] = make_float4(torque, 0);
	}
	
}


__global__ void mapRelativePositionIndependentParticles(float4 *positions, float4 *relativePositions, int *rbIndices, int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles || rbIndices[index] != -1)
		return;

	relativePositions[index] = positions[index];
}

/*
* This function is auxiliary to the add rigid body function.
* After adding a rigid body sphere, new positions are maintained as relative positions.
* However, for the independent particles these relative positions correspond to the act
* ual particle positions and must be copied there.
*/
void mapRelativePositionIndependentParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)gridDim.x = 1;
	mapRelativePositionIndependentParticles << < gridDim, blockDim >> >(positions, //particle positions
		relativePositions, //relative particle positions
		rbIndices, //rigid body indices
		numParticles); //number of particles
}


__global__ void mapActualPositionIndependentParticles(float4 *positions, float4 *relativePositions, int *rbIndices, int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles || rbIndices[index] != -1)
		return;

	positions[index] = relativePositions[index];
}

/*
* This function is auxiliary to the add rigid body function.
* After adding a rigid body sphere, new positions are maintained as relative positions.
* However, for the independent particles these relative positions correspond to the act
* ual particle positions and must be copied there.
*/
void mapActualPositionIndependentParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)gridDim.x = 1;
	mapActualPositionIndependentParticles << < gridDim, blockDim >> >(positions, //particle positions
		relativePositions, //relative particle positions
		rbIndices, //rigid body indices
		numParticles); //number of particles
}


__global__ void mapRelativePositionRigidBodyParticles(float4 *positions,
		float4 *relativePositions,
		int *rbIndices,
		int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles || rbIndices[index] == -1)
		return;

	positions[index] = relativePositions[index];
}

/*
* This function is auxiliary to the add rigid body function.
* This is the exact opposite situation from independent virtual particles.
* Now we need to substitute the global positions of bound particles with 
* their relative positions PRIOR to adding a new sphere. This is necessary
* in order to copy over old relative positions.
*/
void mapRelativePositionRigidBodyParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)gridDim.x = 1;
	mapRelativePositionRigidBodyParticles << < gridDim, blockDim >> >(positions, //particle positions
		relativePositions, //relative particle positions
		rbIndices, //rigid body indices
		numParticles); //number of particles
}

__global__ void mapActualPositionRigidBodyParticles(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	float4 *rbPositions, //rigid body center of mass
	int *rbIndices, //rigid body indices
	int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int rbIndex = rbIndices[index];
	if (index >= numParticles || rbIndex == -1)
		return;

	positions[index] = rbPositions[rbIndex] + relativePositions[index];
}

void mapActualPositionRigidBodyParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	float4 *rbPositions, //rigid body center of mass
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)gridDim.x = 1;
	mapActualPositionRigidBodyParticles << < gridDim, blockDim >> >(
		positions, //particle positions
		relativePositions, //relative particle positions
		rbPositions, //rigid body center of mass
		rbIndices, //rigid body indices
		numParticles); //number of particles

}

__global__ void initializeRadii(float *radii, float particleRadius, int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles)
		return;
	radii[index] = particleRadius;
}

void initializeRadiiWrapper(float *radii, float particleRadius, int numParticles, int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	initializeRadii << < gridDim, blockDim >> >(radii, particleRadius, numParticles);
}
#endif
