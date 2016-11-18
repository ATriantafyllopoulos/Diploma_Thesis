#include "BVHAuxiliary.cuh"

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#define CUB_STDERR
//define _CubLog to avoid encountering error: "undefined reference"
#if !defined(_CubLog)
#if (CUB_PTX_ARCH == 0)
#define _CubLog(format, ...) printf(format,__VA_ARGS__);
#elif (CUB_PTX_ARCH >= 200)
#define _CubLog(format, ...) printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, __VA_ARGS__);
#endif
#endif
#define inf 0x7f800000
//cub headers
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
// calculate position in uniform grid
__device__ int3 calcGridPosAuxilDEM(float3 p, SimParams params)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHashAuxilDEM(int3 gridPos, SimParams params)
{
	gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.gridSize.y - 1);
	gridPos.z = gridPos.z & (params.gridSize.z - 1);
	return gridPos.z * params.gridSize.y * params.gridSize.x + gridPos.y * params.gridSize.x + gridPos.x;
}

__device__
void FindAndHandleRigidBodyCollisionsUniformGridCell(
int3 gridPos,
uint index,
uint originalIndex,
float3 pos,
float4 *sortedPos, // sorted particle positions
float4 *sortedVel, // sorted particle velocities
uint *cellStart,
uint *cellEnd,
uint *gridParticleIndex,//sorted particle indices
int rigidBodyIndex, //rigid body index corresponding to current particle
int *rbIndices, //index of the rigid body each particle belongs to
float4 *impulse, // impulse acted because of collisions with current cell
float3 velA, // current particle velocity
int *numCollisions,
SimParams params)
{
	uint gridHash = calcGridHashAuxilDEM(gridPos, params);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);
	// contact distance is unsorted

	float collisionThreshold = 2 * params.particleRadius; // assuming all particles have the same radius

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			int originalIndex_2 = gridParticleIndex[j];
			int rigidBodyIndex_2 = rbIndices[originalIndex_2];
			if (j != index && (rigidBodyIndex != rigidBodyIndex_2))// check not colliding with self and not of the same rigid body
			{
				float3 pos2 = make_float3(FETCH(sortedPos, j));
				
				float3 relPos = pos2 - pos;
				float dist = length(relPos); // distance between two particles
				if (dist < collisionThreshold)
				{
					float3 velB = make_float3(FETCH(sortedVel, j));
					// particles are colliding
					float3 norm = relPos / dist;

					// relative velocity
					float3 relVel = velB - velA;

					// relative tangential velocity
					float3 tanVel = relVel - (dot(relVel, norm) * norm);

					float3 localImpulse = make_float3(0, 0, 0);
					// spring force
					localImpulse = -params.spring*(collisionThreshold - dist) * norm;
					// dashpot (damping) force
					localImpulse += params.damping*relVel;
					// tangential shear force
					localImpulse += params.shear*tanVel;
					// attraction
					localImpulse += params.attraction*relPos;

					(*impulse) += make_float4(localImpulse, 0);
					(*numCollisions)++;
				}
			}
		}
	}
}

/*
* Kernel function to perform rigid body collision detection
*/
__global__
void FindAndHandleRigidBodyCollisionsUniformGridKernel(
int *rbIndices, // index of the rigid body each particle belongs to
float4 *pLinearImpulse, // total linear impulse acting on current particle
float4 *pAngularImpulse, // total angular impulse acting on current particle
float4 *color, // particle color
float4 *sortedPos,  // sorted particle positions
float4 *sortedVel,  // sorted particle velocities
float4 *relativePos, // unsorted relative positions
float3 minPos, // scene's smallest coordinates
float3 maxPos, // scene's largest coordinates
uint   *gridParticleIndex, // sorted particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(sortedPos, index));
	float3 vel = make_float3(FETCH(sortedVel, index));
	// get address in grid
	int3 gridPos = calcGridPosAuxilDEM(pos, params);

	// examine neighbouring cells
	int numCollisions = 0;
	uint originalIndex = gridParticleIndex[index];
	int rigidBodyIndex = rbIndices[originalIndex];

	float4 linear = make_float4(0, 0, 0, 0);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				float4 localImpulse = make_float4(0, 0, 0, 0);
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				FindAndHandleRigidBodyCollisionsUniformGridCell(
					neighbourPos,
					index,
					originalIndex,
					pos,
					sortedPos, // sorted particle positions
					sortedVel, // sorted particle velocities
					cellStart,
					cellEnd,
					gridParticleIndex,//sorted particle indices
					rigidBodyIndex, //rigid body index corresponding to current particle
					rbIndices, //index of the rigid body each particle belongs to
					&localImpulse, // impulse acted because of collisions with current cell
					vel, // current particle velocity
					&numCollisions, // counting number of collisions
					params);
				linear += localImpulse;
			}
		}
	}

	if (numCollisions)
		color[originalIndex] = make_float4(1, 0, 0, 0);
	else
		color[originalIndex] = make_float4(0, 0, 1, 0);
	//// now find and handle wall collisions
	//// NOTE: should this be moved to a separate kernel call
	//// (clarity and speed?)
	//numCollisions = 0;
	//if (pos.x < minPos.x + params.particleRadius && vel.x < 0)
	//{
	//	linear += make_float4((params.boundaryDamping - 1) * vel.x, 0, 0, 0);
	//	numCollisions++;
	//}
	//if (pos.y < minPos.y + params.particleRadius && vel.y < 0)
	//{
	//	linear += make_float4(0, (params.boundaryDamping - 1) * vel.y, 0, 0);
	//	numCollisions++;
	//}
	//if (pos.z < minPos.z + params.particleRadius && vel.z < 0)
	//{
	//	linear += make_float4(0, 0, (params.boundaryDamping - 1) * vel.z, 0);
	//	numCollisions++;
	//}

	//if (pos.x > maxPos.x - params.particleRadius && vel.x > 0)
	//{
	//	linear += make_float4((params.boundaryDamping - 1) * vel.x, 0, 0, 0);
	//	numCollisions++;
	//}
	//if (pos.y > maxPos.y - params.particleRadius && vel.y > 0)
	//{
	//	linear += make_float4(0, (params.boundaryDamping - 1) * vel.y, 0, 0);
	//	numCollisions++;
	//}
	//if (pos.z > maxPos.z - params.particleRadius && vel.z > 0)
	//{
	//	linear += make_float4(0, 0, (params.boundaryDamping - 1) * vel.z, 0);
	//	numCollisions++;
	//}
	//
	//if (numCollisions)
	//	color[originalIndex] = make_float4(1, 1, 1, 0);

	pLinearImpulse[originalIndex] += linear;
	// tau = r x F
	pAngularImpulse[originalIndex] += make_float4(cross(make_float3(relativePos[originalIndex]), make_float3(linear)), 0); 
	/**/
}

void FindAndHandleRigidBodyCollisionsUniformGridWrapper(
	int *rbIndices, // index of the rigid body each particle belongs to
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	float4 *color, // particle color
	float4 *sortedPos,  // sorted particle positions
	float4 *sortedVel,  // sorted particle velocities
	float4 *relativePos, // unsorted relative positions
	float3 minPos, // scene's smallest coordinates
	float3 maxPos, // scene's largest coordinates
	uint   *gridParticleIndex, // sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	SimParams params,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	FindAndHandleRigidBodyCollisionsUniformGridKernel << < gridDim, blockDim >> >(
		rbIndices, // index of the rigid body each particle belongs to
		pLinearImpulse, // total linear impulse acting on current particle
		pAngularImpulse, // total angular impulse acting on current particle
		color, // particle color
		sortedPos,  // sorted particle positions
		sortedVel,  // sorted particle velocities
		relativePos, // unsorted relative positions
		minPos, // scene's smallest coordinates
		maxPos, // scene's largest coordinates
		gridParticleIndex, // sorted particle indices
		cellStart,
		cellEnd,
		numParticles,
		params);
}

__device__
void FindAndHandleAugmentedRealityCollisionsUniformGridCell(
int3 gridPos, // cell to check
uint index, // sorted index of current particle
uint originalIndex, // unsorted index of current particle
float3 pos, // current particle position
float4 *ARPos, // sorted scene particle positions
float4 *ARnormals, // unsorted scene normals
uint *gridParticleIndexAR, // sorted scene particle indices
uint *cellStart, // scene cell start
uint *cellEnd, // scene cell end
float4 *impulse, // impulse acted because of collisions with current cell
float3 velA, // current particle velocity
int *numCollisions,
SimParams params)
{
	uint gridHash = calcGridHashAuxilDEM(gridPos, params);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);
	// contact distance is unsorted

	float collisionThreshold = 2 * params.particleRadius; // assuming all particles have the same radius

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			int originalIndex_2 = gridParticleIndexAR[j];
			float3 pos2 = make_float3(FETCH(ARPos, j));

			float3 relPos = pos2 - pos;
			float dist = length(relPos); // distance between two particles
			if (dist < collisionThreshold)
			{
				float3 velB = make_float3(0, 0, 0);
				// particles are colliding
				//float3 norm = relPos / dist;
				//float3 norm = make_float3(FETCH(ARnormals, originalIndex_2));
				float3 norm = make_float3(0, 1, 0);
				// relative velocity
				float3 relVel = velB - velA;

				// relative tangential velocity
				float3 tanVel = relVel - (dot(relVel, norm) * norm);

				float3 localImpulse = make_float3(0, 0, 0);
				// spring force
				localImpulse = -params.spring*(collisionThreshold - dist) * norm;
				// dashpot (damping) force
				localImpulse += params.damping*relVel;
				// tangential shear force
				localImpulse += params.shear*tanVel;
				// attraction
				localImpulse += params.attraction*relPos;

				(*impulse) += make_float4(localImpulse, 0);
				(*numCollisions)++;
			}
		}
	}
}

/*
* Perform simultaneous rigid body and AR collision detection and handling using DEM
*/
__global__
void FindAndHandleAugmentedRealityCollisionsUniformGridKernel(
float4 *pLinearImpulse, // total linear impulse acting on current particle
float4 *pAngularImpulse, // total angular impulse acting on current particle
float4 *color, // particle color
float4 *sortedPos,  // sorted particle positions
float4 *sortedVel,  // sorted particle velocities
float4 *relativePos, // unsorted relative positions
float4 *ARPos, // sorted scene particle positions
float4 *ARnormals, // unsorted scene normals
uint   *gridParticleIndex, // sorted particle indices
uint *gridParticleIndexAR, // sorted scene particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(sortedPos, index));
	float3 vel = make_float3(FETCH(sortedVel, index));
	// get address in grid
	int3 gridPos = calcGridPosAuxilDEM(pos, params);

	// examine neighbouring cells
	int numCollisions = 0;
	uint originalIndex = gridParticleIndex[index];

	float4 linear = make_float4(0, 0, 0, 0);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				float4 localImpulse = make_float4(0, 0, 0, 0);
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				FindAndHandleAugmentedRealityCollisionsUniformGridCell(
					neighbourPos, // cell to check
					index, // sorted index of current particle
					originalIndex, // unsorted index of current particle
					pos, // current particle position
					ARPos, // sorted scene particle positions
					ARnormals, // unsorted scene normals
					gridParticleIndexAR, // sorted scene particle indices
					cellStart, // scene cell start
					cellEnd, // scene cell end
					&localImpulse, // impulse acted because of collisions with current cell
					vel, // current particle velocity
					&numCollisions,
					params);
				linear += localImpulse;
			}
		}
	}

	if (numCollisions)
		color[originalIndex] = make_float4(0, 1, 0, 0);

	pLinearImpulse[originalIndex] += linear;
	// tau = r x F
	pAngularImpulse[originalIndex] += make_float4(cross(make_float3(relativePos[originalIndex]), make_float3(linear)), 0);
}

/*
* Wrapper for rigid body to AR collision handling
*/
void FindAndHandleAugmentedRealityCollisionsUniformGridWrapper(
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	float4 *color, // particle color
	float4 *sortedPos,  // sorted particle positions
	float4 *sortedVel,  // sorted particle velocities
	float4 *relativePos, // unsorted relative positions
	float4 *ARPos, // sorted scene particle positions
	float4 *ARnormals, // unsorted scene normals
	uint   *gridParticleIndex, // sorted particle indices
	uint   *gridParticleIndexAR, // sorted scene particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	SimParams params,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	FindAndHandleAugmentedRealityCollisionsUniformGridKernel << < gridDim, blockDim >> >(
		pLinearImpulse, // total linear impulse acting on current particle
		pAngularImpulse, // total angular impulse acting on current particle
		color, // particle color
		sortedPos,  // sorted particle positions
		sortedVel,  // sorted particle velocities
		relativePos, // unsorted relative positions
		ARPos, // sorted scene particle positions
		ARnormals, // unsorted scene normals
		gridParticleIndex, // sorted particle indices
		gridParticleIndexAR, // sorted scene particle indices
		cellStart,
		cellEnd,
		numParticles,
		params);
}


/* Now apply total impulse to each rigid body*/
__global__ void CombineReducedImpulses(
	float4 *reducedLinearImpulse, // total linear impulse acting on current particle
	float4 *reducedAngularImpulse, // total angular impulse acting on current particle
	float4 *rbVel, // rigid body linear velocity 
	float4 *rbAng, // rigid body angular velocity 
	float *rbMass, // rigid body mass
	glm::mat3 *rbInertia, // rigid body inverse inertia matrix
	int *particlesPerObject, // number of particles per object
	int numRigidBodies) // total number of rigid bodies)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numRigidBodies) 
		return;
	int cumulativeIndex = particlesPerObject[index] - 1;
	float4 totalLinearImpulse = reducedLinearImpulse[cumulativeIndex];
	float4 totalAngularImpulse = reducedAngularImpulse[cumulativeIndex];
	if (index > 0)
	{
		totalLinearImpulse -= reducedLinearImpulse[particlesPerObject[index - 1]];
		totalAngularImpulse -= reducedAngularImpulse[particlesPerObject[index - 1]];
	}
	glm::mat3 inertia = rbInertia[index];
	glm::vec3 angularImpulse(totalAngularImpulse.x, totalAngularImpulse.y, totalAngularImpulse.z);
	glm::vec3 w = inertia * angularImpulse;

	rbVel[index] += totalLinearImpulse / rbMass[index];
	rbAng[index] += make_float4(w.x, w.y, w.z, 0);
}

void ReducePerParticleImpulses(
	int *rbIndices, // index of the rigid body each particle belongs to
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	float4 *rbVel, // rigid body linear velocity 
	float4 *rbAng, // rigid body angular velocity 
	float *rbMass, // rigid body mass
	glm::mat3 *rbInertia, // rigid body inverse inertia matrix
	int *particlesPerObject, // number of particles per object
	int numRigidBodies, // total number of rigid bodies
	int numParticles, // total number of particles
	int numThreads) // number of threads to be used
{

	float4 *d_linear_out; // temporary storage of scanned linear impulses
	checkCudaErrors(cudaMalloc((void**)&d_linear_out, sizeof(float) * 4 * numParticles));
	float4 *d_angular_out; // temporary storage of scanned angular impulses
	checkCudaErrors(cudaMalloc((void**)&d_angular_out, sizeof(float) * 4 * numParticles));

	// Determine temporary device storage requirements for inclusive prefix sum
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	// compute CUDA scan of linear impulses
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, pLinearImpulse, d_linear_out, numParticles);
	// Allocate temporary storage for inclusive prefix sum
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run inclusive prefix sum
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, pLinearImpulse, d_linear_out, numParticles);

	checkCudaErrors(cudaFree(d_temp_storage)); d_temp_storage = NULL; temp_storage_bytes = 0; // reset auxiliaries

	// compute CUDA scan of angular impulses
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, pAngularImpulse, d_angular_out, numParticles);
	// Allocate temporary storage for inclusive prefix sum
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run inclusive prefix sum
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, pAngularImpulse, d_angular_out, numParticles);

	checkCudaErrors(cudaFree(d_temp_storage)); d_temp_storage = NULL; temp_storage_bytes = 0; // reset auxiliaries
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

//#define DEBUG_SCAN
#ifdef DEBUG_SCAN
	float4 *h_linearImpulse = new float4[numRigidBodies];
	float4 *h_angularImpulse = new float4[numRigidBodies];
	int processed = 0;
	for (int rb = 0; rb < numRigidBodies; rb++)
	{
		processed += particlesPerObject[rb];
		checkCudaErrors(cudaMemcpy(&h_linearImpulse[rb], &d_linear_out[processed - 1], sizeof(float) * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&h_angularImpulse[rb], &d_angular_out[processed - 1], sizeof(float) * 4, cudaMemcpyDeviceToHost));
		std::cout << "Total linear impulse applied to rigid body # " << rb << ": (" << h_angularImpulse[rb].x <<
			", " << h_angularImpulse[rb].y << ", " << h_angularImpulse[rb].z << ")" << std::endl;
		std::cout << "Total angular impulse applied to rigid body # " << rb << ": (" << h_angularImpulse[rb].x <<
			", " << h_angularImpulse[rb].y << ", " << h_angularImpulse[rb].z << ")" << std::endl;
		
	}
	delete h_linearImpulse;
	delete h_angularImpulse;

	float4 *h_particleLinearImpulse = new float4[numParticles];
	float4 *h_particleAngularImpulse = new float4[numParticles];

	checkCudaErrors(cudaMemcpy(h_particleLinearImpulse, pLinearImpulse, numParticles * sizeof(float) * 4, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_particleAngularImpulse, pAngularImpulse, numParticles * sizeof(float) * 4, cudaMemcpyDeviceToHost));

	float4 *h_cumulativeLinearImpulse = new float4[numParticles];
	float4 *h_cumulativeAngularImpulse = new float4[numParticles];

	checkCudaErrors(cudaMemcpy(h_cumulativeLinearImpulse, d_linear_out, numParticles * sizeof(float) * 4, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_cumulativeAngularImpulse, d_angular_out, numParticles * sizeof(float) * 4, cudaMemcpyDeviceToHost));

	/*for (int p = 0; p < numParticles; p++)
	{
		if (abs(h_particleLinearImpulse[p].x) > 0.001 ||
			abs(h_particleLinearImpulse[p].y) > 0.001 ||
			abs(h_particleLinearImpulse[p].z) > 0.001)
		{
			std::cout << "Total linear impulse applied to particle# " << p << ": (" << h_particleLinearImpulse[p].x <<
				", " << h_particleLinearImpulse[p].y << ", " << h_particleLinearImpulse[p].z << ")" << std::endl;
			std::cout << "Cumulative linear impulse applied at particle# " << p << ": (" << h_cumulativeLinearImpulse[p].x <<
				", " << h_cumulativeLinearImpulse[p].y << ", " << h_cumulativeLinearImpulse[p].z << ")" << std::endl;
		}

		if (abs(h_particleAngularImpulse[p].x) > 0.001 ||
			abs(h_particleAngularImpulse[p].y) > 0.001 ||
			abs(h_particleAngularImpulse[p].z) > 0.001)
		{
			std::cout << "Total angular impulse applied to particle# " << p << ": (" << h_particleAngularImpulse[p].x <<
				", " << h_particleAngularImpulse[p].y << ", " << h_particleAngularImpulse[p].z << ")" << std::endl;
			std::cout << "Cumulative angular impulse applied at particle# " << p << ": (" << h_cumulativeAngularImpulse[p].x <<
				", " << h_cumulativeAngularImpulse[p].y << ", " << h_cumulativeAngularImpulse[p].z << ")" << std::endl;
		}

	}*/

	float4 *CPU_linear_cumulative = new float4[numParticles];
	float4 *CPU_angular_cumulative = new float4[numParticles];
	CPU_linear_cumulative[0] = h_particleLinearImpulse[0];
	CPU_angular_cumulative[0] = h_particleAngularImpulse[0];
	for (int p = 1; p < numParticles; p++)
	{
		CPU_linear_cumulative[p] = CPU_linear_cumulative[p - 1] + h_particleLinearImpulse[p];
		CPU_angular_cumulative[p] = CPU_angular_cumulative[p - 1] + h_particleAngularImpulse[p];
		
		if (abs(CPU_linear_cumulative[p].x - h_cumulativeLinearImpulse[p].x) > 0.001 ||
			abs(CPU_linear_cumulative[p].y - h_cumulativeLinearImpulse[p].y) > 0.001 ||
			abs(CPU_linear_cumulative[p].z - h_cumulativeLinearImpulse[p].z) > 0.001)
		{
			std::cout << "CPU cumulative linear impulse applied to particle# " << p << ": (" << CPU_linear_cumulative[p].x <<
				", " << CPU_linear_cumulative[p].y << ", " << CPU_linear_cumulative[p].z << ")" << std::endl;
			std::cout << "GPU cumulative linear impulse applied at particle# " << p << ": (" << h_cumulativeLinearImpulse[p].x <<
				", " << h_cumulativeLinearImpulse[p].y << ", " << h_cumulativeLinearImpulse[p].z << ")" << std::endl;
		}

		if (abs(CPU_angular_cumulative[p].x - h_cumulativeAngularImpulse[p].x) > 0.001 ||
			abs(CPU_angular_cumulative[p].y - h_cumulativeAngularImpulse[p].y) > 0.001 ||
			abs(CPU_angular_cumulative[p].z - h_cumulativeAngularImpulse[p].z) > 0.001)
		{
			std::cout << "CPU cumulative angular impulse applied to particle# " << p << ": (" << CPU_angular_cumulative[p].x <<
				", " << CPU_angular_cumulative[p].y << ", " << CPU_angular_cumulative[p].z << ")" << std::endl;
			std::cout << "GPU cumulative angular impulse applied at particle# " << p << ": (" << h_cumulativeAngularImpulse[p].x <<
				", " << h_cumulativeAngularImpulse[p].y << ", " << h_cumulativeAngularImpulse[p].z << ")" << std::endl;
		}

	}
	delete CPU_linear_cumulative;
	delete CPU_angular_cumulative;
	delete h_particleLinearImpulse;
	delete h_particleAngularImpulse;
	delete h_cumulativeLinearImpulse;
	delete h_cumulativeAngularImpulse;
#endif


//	int *d_particlesPerObject; // copy number of particles per object to GPU
//	checkCudaErrors(cudaMalloc((void**)&d_particlesPerObject, sizeof(int) * numRigidBodies));
//	checkCudaErrors(cudaMemcpy(d_particlesPerObject, particlesPerObject, sizeof(int) * numRigidBodies, cudaMemcpyHostToDevice));
//
//	int *d_CumulativeParticlesPerObject;
//	checkCudaErrors(cudaMalloc((void**)&d_particlesPerObject, sizeof(int) * numRigidBodies));
//
//	void     *d_temp_storage_int = NULL;
//	size_t   temp_storage_bytes_int = 0;
//
//	cub::DeviceScan::InclusiveSum(d_temp_storage_int, temp_storage_bytes_int, d_particlesPerObject, d_CumulativeParticlesPerObject, numRigidBodies);
//	// Allocate temporary storage for inclusive prefix sum
//	checkCudaErrors(cudaMalloc(&d_temp_storage_int, temp_storage_bytes_int));
//	// Run inclusive prefix sum
//	cub::DeviceScan::InclusiveSum(d_temp_storage_int, temp_storage_bytes_int, d_particlesPerObject, d_CumulativeParticlesPerObject, numRigidBodies);
//
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//

//
//#ifdef DEBUG_SCAN
//	int *h_cumulativeIndices = new int[numRigidBodies];
//	checkCudaErrors(cudaMemcpy(h_cumulativeIndices, d_CumulativeParticlesPerObject, sizeof(int) * numRigidBodies, cudaMemcpyDeviceToHost));
//	for (int rb = 0; rb < numRigidBodies; rb++)
//	{
//		std::cout << "Cumulative particle index: " << h_cumulativeIndices[rb] << std::endl;
//	}
//	delete h_cumulativeIndices;
//	delete h_angularImpulse;
//#endif

	int *cumulative_particle_indices = new int[numRigidBodies];
	cumulative_particle_indices[0] = particlesPerObject[0];
	for (int rb = 1; rb < numRigidBodies; rb++)
	{
		cumulative_particle_indices[rb] = cumulative_particle_indices[rb - 1] + particlesPerObject[rb];
	}

	
	
	int *d_CumulativeParticlesPerObject;
	checkCudaErrors(cudaMalloc((void**)&d_CumulativeParticlesPerObject, sizeof(int) * numRigidBodies));
	checkCudaErrors(cudaMemcpy(d_CumulativeParticlesPerObject, cumulative_particle_indices, sizeof(int) * numRigidBodies, cudaMemcpyHostToDevice));
	delete cumulative_particle_indices;
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numRigidBodies + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
		gridDim.x = 1;

	CombineReducedImpulses << < gridDim, blockDim >> >(
		d_linear_out, // total linear impulse acting on current particle
		d_angular_out, // total angular impulse acting on current particle,
		rbVel, // rigid body linear velocity 
		rbAng, // rigid body angular velocity 
		rbMass, // rigid body mass
		rbInertia, // rigid body inverse inertia matrix
		d_CumulativeParticlesPerObject, // number of particles per object
		numRigidBodies);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//checkCudaErrors(cudaFree(d_temp_storage_int));
	//checkCudaErrors(cudaFree(d_particlesPerObject));
	checkCudaErrors(cudaFree(d_CumulativeParticlesPerObject));
	checkCudaErrors(cudaFree(d_linear_out));
	checkCudaErrors(cudaFree(d_angular_out));
}


__global__ void ResetParticleImpulseKernel(
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	pLinearImpulse[index] = make_float4(0, 0, 0, 0);
	pAngularImpulse[index] = make_float4(0, 0, 0, 0);
}

void ResetParticleImpulseWrapper(
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	int numParticles, //number of rigid bodies
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	//std::cout << "Using " << blockDim.x << " threads and " << gridDim.x << " blocks" << std::endl;
	ResetParticleImpulseKernel << < gridDim, blockDim >> >(
		pLinearImpulse, // total linear impulse acting on current particle
		pAngularImpulse, // total angular impulse acting on current particle
		numParticles); //number of rigid bodies
}