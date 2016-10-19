#define CUB_STDERR
//define _CubLog to avoid encountering error: "undefined reference"
#if !defined(_CubLog)
#if (CUB_PTX_ARCH == 0)
#define _CubLog(format, ...) printf(format,__VA_ARGS__);
#elif (CUB_PTX_ARCH >= 200)
#define _CubLog(format, ...) printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, __VA_ARGS__);
#endif
#endif

//cub headers
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "BVHcreation.h"
#include "particleSystem.cuh"

__global__ void kernelConstructRadixTreeSoA(
	int numberOfInternalNodes,
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	unsigned int *sortedMortonCodes
);

__global__ void kernelConstructLeafNodesSoA(
	int numberOfLeaves,
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	float4 *CMs, //array containing centers of mass for each leaf
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, //array containing corresponding unsorted indices for each leaf
	float *radii, //radii of all nodes - currently the same for all particles
	float4 *positions, //original positions
	float particleRadius //common radius parameter
);

__global__ void kernelConstructInternalNodesSoA(
	int numberOfLeaves,
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *nodeCounter //used by atomic operations - ensures that each 
);

__global__
void collideBVHSoA(
float4 *color, //particle's color, only used for testing purposes
float4 *vel, //particles original velocity, updated after all collisions are handled
bool *isLeaf, //array containing a flag to indicate whether node is leaf
int *leftIndices, //array containing indices of the left children of each node
int *rightIndices, //array containing indices of the right children of each node
int *minRange, //array containing minimum (sorted) leaf covered by each node
int *maxRange, //array containing maximum (sorted) leaf covered by each node
float4 *CMs, //array containing centers of mass for each leaf
AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
int *sortedIndices, //array containing corresponding unsorted indices for each leaf
float *radii, //radii of all nodes - currently the same for all particles
int numParticles, //number of virtual particles
SimParams params //simulation parameters
);

__global__
void staticCollideBVHSoA(float4 *positions, //virtual particle positions
float4 *vel, //particles original velocity, updated after all collisions are handled
float4 *normals, //normals computed for each real particle using its 8-neighborhood
bool *isLeaf, //array containing a flag to indicate whether node is leaf
int *leftIndices, //array containing indices of the left children of each node
int *rightIndices, //array containing indices of the right children of each node
int *minRange, //array containing minimum (sorted) leaf covered by each node
int *maxRange, //array containing maximum (sorted) leaf covered by each node
float4 *CMs, //array containing centers of mass for each leaf
AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
int *sortedIndices, //array containing corresponding unsorted indices for each leaf
float *radii, //radii of all nodes - currently the same for all particles
int numParticles, //number of virtual particles
int numRangeData, //number of static data
SimParams params); //simulation parameters

void wrapperConstructRadixTreeSoA(
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	unsigned int *sortedMortonCodes,
	int numThreads,
	int numParticles)
{
	int internalNodes = numParticles - 1;
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((internalNodes + numThreads - 1) / numThreads, 1);
	kernelConstructRadixTreeSoA << < gridDim, blockDim >> >(
		internalNodes,
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		sortedMortonCodes
		);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void wrapperConstructLeafNodesSoA(
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	float4 *CMs, //array containing centers of mass for each leaf
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, //array containing corresponding unsorted indices for each leaf
	float *radii, //radii of all nodes - currently the same for all particles
	float4 *positions, //original positions
	float particleRadius, //common radius parameter
	int numThreads,
	int numParticles
	)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	kernelConstructLeafNodesSoA << < gridDim, blockDim >> >(
		numParticles,
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		positions, //original positions
		particleRadius //common radius parameter
		);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void wrapperConstructInternalNodesSoA(
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures 
	int numThreads,
	int numParticles)
{
	int *nodeCounter;

	checkCudaErrors(cudaMalloc(&nodeCounter, sizeof(int) * (numParticles - 1)));
	checkCudaErrors(cudaMemset(nodeCounter, 0, sizeof(int) * (numParticles - 1)));

	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	kernelConstructInternalNodesSoA << < gridDim, blockDim >> >(
		numParticles,
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		nodeCounter //used by atomic operations - ensures that each node is accessed only once
		);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(nodeCounter));
}

void wrapperCollideBVHSoA(
float4 *color, //particle's color, only used for testing purposes
float4 *vel, //particles original velocity, updated after all collisions are handled
bool *isLeaf, //array containing a flag to indicate whether node is leaf
int *leftIndices, //array containing indices of the left children of each node
int *rightIndices, //array containing indices of the right children of each node
int *minRange, //array containing minimum (sorted) leaf covered by each node
int *maxRange, //array containing maximum (sorted) leaf covered by each node
float4 *CMs, //array containing centers of mass for each leaf
AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
int *sortedIndices, //array containing corresponding unsorted indices for each leaf
float *radii, //radii of all nodes - currently the same for all particles
int numParticles, //number of virtual particles
SimParams params, //simulation parameters
int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	collideBVHSoA << < gridDim, blockDim >> >(
		color, //particle's color, only used for testing purposes
		vel, //particles original velocity, updated after all collisions are handled
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		numParticles, //number of virtual particles
		params //simulation parameters
		);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
}


void createSoA(
	bool **isLeaf, //array containing a flag to indicate whether node is leaf
	int **parentIndices, //array containing indices of the parent of each node
	int **leftIndices, //array containing indices of the left children of each node
	int **rightIndices, //array containing indices of the right children of each node
	int **minRange, //array containing minimum (sorted) leaf covered by each node
	int **maxRange, //array containing maximum (sorted) leaf covered by each node
	float4 **CMs, //array containing centers of mass for each leaf
	AABB **bounds, //array containing bounding volume for each node - currently templated Array of Structures
	float **radii, //radii of all nodes - currently the same for all particles
	unsigned int **mortonCodes,
	int **indices,
	unsigned int **sortedMortonCodes,
	int **sortedIndices, //array containing corresponding unsorted indices for each leaf
	int numberOfPrimitives,
	int numberOfThreads)
{
    //std::cout << "Attempting to allocate memory for " << numberOfPrimitives << " particles." << std::endl;
	//malloc arrays
	checkCudaErrors(cudaMalloc((void**)mortonCodes, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMalloc((void**)indices, sizeof(int) * numberOfPrimitives));
	checkCudaErrors(cudaMalloc((void**)sortedMortonCodes, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMalloc((void**)sortedIndices, sizeof(int) * numberOfPrimitives));

	int totalNumberOfNodes = 2 * numberOfPrimitives - 1;
	//arrays used by all nodes
	checkCudaErrors(cudaMalloc((void**)isLeaf, sizeof(bool) * totalNumberOfNodes));
	checkCudaErrors(cudaMalloc((void**)parentIndices, sizeof(int) * totalNumberOfNodes));
	checkCudaErrors(cudaMalloc((void**)leftIndices, sizeof(int) * totalNumberOfNodes));
	checkCudaErrors(cudaMalloc((void**)rightIndices, sizeof(int) * totalNumberOfNodes));
	checkCudaErrors(cudaMalloc((void**)minRange, sizeof(int) * totalNumberOfNodes));
	checkCudaErrors(cudaMalloc((void**)maxRange, sizeof(int) * totalNumberOfNodes));
	checkCudaErrors(cudaMalloc((void**)bounds, sizeof(AABB) * totalNumberOfNodes));

	//arrays used by leaf nodes only
	checkCudaErrors(cudaMalloc((void**)CMs, sizeof(float4) * numberOfPrimitives));	
	checkCudaErrors(cudaMalloc((void**)radii, sizeof(float) * numberOfPrimitives));

    //std::cout << "Memory for " << numberOfPrimitives << " particles allocated successfully." << std::endl;
	//memset arrays
	/*checkCudaErrors(cudaMemset(*mortonCodes, 0, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMemset(*indices, 0, sizeof(int) * numberOfPrimitives));
	checkCudaErrors(cudaMemset(*sortedMortonCodes, 0, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMemset(*sortedIndices, 0, sizeof(int) * numberOfPrimitives));*/

}

void wrapperStaticCollideBVHSoA(float4 *positions, //virtual particle positions
float4 *vel, //particles original velocity, updated after all collisions are handled
float4 *normals, //normals computed for each real particle using its 8-neighborhood
bool *isLeaf, //array containing a flag to indicate whether node is leaf
int *leftIndices, //array containing indices of the left children of each node
int *rightIndices, //array containing indices of the right children of each node
int *minRange, //array containing minimum (sorted) leaf covered by each node
int *maxRange, //array containing maximum (sorted) leaf covered by each node
float4 *CMs, //array containing centers of mass for each leaf
AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
int *sortedIndices, //array containing corresponding unsorted indices for each leaf
float *radii, //radii of all nodes - currently the same for all particles
int numParticles, //number of virtual particles
int numRangeData, //number of static data
int numThreads,
SimParams params) //simulation parameters
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	staticCollideBVHSoA << < gridDim, blockDim >> >(positions, //virtual particle positions
		vel, //particles original velocity, updated after all collisions are handled
		normals, //normals computed for each real particle using its 8-neighborhood
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		numParticles, //number of virtual particles
		numRangeData, //number of static data
		params); //simulation parameters
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
