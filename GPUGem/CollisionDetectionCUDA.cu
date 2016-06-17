#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Primitives.h"
#include <algorithm>
#include <iostream>
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
#include <cub/test/test_util.h>

/*
This particular file is a mess. I definitely need to clean it up on the weekend.
It's decided. The weekend will be dedicated to cleaning up the code.
This will need to be part of a class.
*/
__global__ void handleCollisions(Particle *leafNodes, int numberOfPrimitives);
cudaError_t cudaFail(cudaError_t cudaStatus, char *funcName);
cudaError_t generateHierarchy(Particle *internalNodes,
	Particle* leafNodes,
	unsigned int* sortedMortonCodes,
	int           numberOfPrimitives);
__device__ inline float MIN(float x, float y)
{
	return x < y ? x : y;
}

__device__ inline float MAX(float x, float y)
{
	return x > y ? x : y;
}

//Expands a 10-bit integer into 30 bits
//by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z)
{
	x = MIN(MAX(x * 1024.0f, 0.0f), 1023.0f);
	y = MIN(MAX(y * 1024.0f, 0.0f), 1023.0f);
	z = MIN(MAX(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__device__ inline bool checkOverlap(Particle *query, Particle *node)
{
	float dist = __fsqrt_rd((node->centroid.x - query->centroid.x) * (node->centroid.x - query->centroid.x) +
		(node->centroid.y - query->centroid.y) * (node->centroid.y - query->centroid.y) +
		(node->centroid.z - query->centroid.z) * (node->centroid.z - query->centroid.z));
	return dist < node->radius + query->radius;
}

__device__ void traverseIterative(CollisionList& list,
	Particle *root,
	Particle* queryLeaf)
{
	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	Particle* stack[64]; //AT: Is 64 the correct size to use?
	Particle** stackPtr = stack;
	//when stack is empty thread will return
	*stackPtr++ = NULL; // push NULL at beginning

	// Traverse nodes starting from the root.
	Particle* node = root;
	do
	{
		// Check each child node for overlap.
		Particle* childL = node->left;
		Particle* childR = node->right;
		bool overlapL = (checkOverlap(queryLeaf, childL));
		bool overlapR = (checkOverlap(queryLeaf, childR));

		if (node->leftmost <= queryLeaf->id)
			overlapL = false;

		if (node->rightmost <= queryLeaf->id)
			overlapR = false;
		// Query overlaps a leaf node => report collision.
		if (overlapL && childL->isLeaf)
			queryLeaf->collisions[queryLeaf->collisionCounter++] = childL;
			//list.add(queryLeaf->id, childL->id);
		queryLeaf->collisionCounter = queryLeaf->collisionCounter > 7 ? 0 : queryLeaf->collisionCounter; //avoid overflow
		if (overlapR && childR->isLeaf)
			queryLeaf->collisions[queryLeaf->collisionCounter++] = childR;
		queryLeaf->collisionCounter = queryLeaf->collisionCounter > 7 ? 0 : queryLeaf->collisionCounter; //avoid overflow
		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && !childL->isLeaf);
		bool traverseR = (overlapR && !childR->isLeaf);

		if (!traverseL && !traverseR)
			node = *--stackPtr; // pop
		else
		{
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR)
				*stackPtr++ = childR; // push
		}
	} while (node != NULL);
}

__global__ void generateMortonCodes(float3 *positions, unsigned int *mortonCodes, int numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfPrimitives)
		return;

	mortonCodes[index] = morton3D(positions[index].x, positions[index].y, positions[index].z);
}

__global__ void findPotentialCollisions(CollisionList list, Particle *internalNodes, Particle *leafNodes, int numOfLeaves)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= numOfLeaves)
		return;

	Particle *leaf = leafNodes + index;
	traverseIterative(list, internalNodes, leaf);
}

/*
routine is called before BVH is created
leaf node primitives are yet unsorted
radius and mass are currently hard-coded -> 1 [OPEN]
Make radius parametric. Design interface to input parameters. [OPEN]
*/
__global__ void constructLeafNodes(Particle* leafNodes,
	float3 *positions,
	int numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfPrimitives)
		return;
	leafNodes[index].id = index; //leaf nodes are unsorted
	leafNodes[index].isLeaf = true;
	//each leaf reports rightmost leaf of left and right sutree as itself
	leafNodes[index].leftmost = index;
	leafNodes[index].rightmost = index;

	leafNodes[index].left = NULL;
	leafNodes[index].right = NULL;

	leafNodes[index].parent = NULL;

	//copying state vectors
	//too slow because of too many memory calls
	leafNodes[index].centroid = positions[index];
	leafNodes[index].radius = 1;
	leafNodes[index].mass = 1;
	leafNodes[index].collisionCounter = 0;

	//float3 a = linearMomenta[index];
	//leafNodes[index].linearMomentum = linearMomenta[index];
	//leafNodes[index].linearMomentum = *(float3*)((double*)(linearMomenta + index));
	/*float x = *(float*)(linearMomenta + index);
	float y = *(float*)(linearMomenta + index + 32);
	float z = *(float*)(linearMomenta + index + 64);

	leafNodes[index].linearMomentum = make_float3(x, y, z);*/
	/*int idx = 3 * blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float sMomenta[512 * 3];
	sMomenta[threadIdx.x] = linearMomenta[idx];
	sMomenta[threadIdx.x + 512] = linearMomenta[idx + 512];
	sMomenta[threadIdx.x + 1024] = linearMomenta[idx + 1024];
	__syncthreads();*/

	//leafNodes[idx].linearMomentum.x = (sMomenta)[threadIdx.x];
	//leafNodes[idx + 512].linearMomentum.y = (sMomenta)[threadIdx.x + 512];
	//leafNodes[idx + 1024].linearMomentum.z = (sMomenta)[threadIdx.x + 1024];
	//leafNodes[index].angularMomentum = angularMomentums[index];
	//leafNodes[index].quaternion = quaternions[index];
}

__global__ void testingMemoryAccess(float *input, float *output, int numberOfPrimitives)
{
	/*int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfPrimitives) return;
	output[index] = input[index];*/
	int idx = 3 * blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float sMomenta[512 * 3];
	sMomenta[threadIdx.x] = input[idx];
	sMomenta[threadIdx.x + 512] = input[idx + 512];
	sMomenta[threadIdx.x + 1024] = input[idx + 1024];
	__syncthreads();

	output[idx] = (sMomenta)[threadIdx.x];
	output[idx + 512] = (sMomenta)[threadIdx.x + 512];
	output[idx + 1024] = (sMomenta)[threadIdx.x + 1024];
}

__global__ void assignMomenta(Particle *leafNodes, float3 *linearMomenta, int numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfPrimitives)
		return;
	leafNodes[index].linearMomentum = linearMomenta[index];
}
/*
Potential error: sorting the leaf nodes by value, using Morton codes as keys, is not done on the code I found. [OPEN]
*/
cudaError_t detectCollisions(float3 *positions, float3 *linearMomenta, int numberOfPrimitives)
{
	unsigned int *mortonCodes;
	cudaError_t cudaStatus = cudaMalloc((void**)&mortonCodes, numberOfPrimitives * sizeof(unsigned int));
	//cudaStatus = cudaMemset(mortonCodes, 0, sizeof(unsigned int) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess) {
		cudaFree(mortonCodes);
		return cudaFail(cudaStatus, "cudaMalloc_mortonCodes");
	}

	int numOfThreads = 512; //maximum amount of threads supported by laptop
	//assign a Morton code to each primitive
	//launch all objects
	generateMortonCodes << <(numberOfPrimitives + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions, mortonCodes, numberOfPrimitives);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cudaFree(mortonCodes);
		return cudaFail(cudaStatus, "generateMortonCodes_getLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cudaFree(mortonCodes);
		return cudaFail(cudaStatus, "generateMortonCodes_cudaDeviceSynchronize");
	}

	//create leaf nodes here
	//then sort them using their morton codes as keys
	//and pass them as argument to the BVH hierarchy creation routine
	Particle *leafNodes;
	cudaStatus = cudaMalloc((void**)&leafNodes, numberOfPrimitives * sizeof(Particle));
	//cudaStatus = cudaMemset(leafNodes, 0, sizeof(Particle) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "cudaMalloc_leafNodes");
	}

	float3 *temp;
	cudaStatus = cudaMalloc((void**)&temp, numberOfPrimitives * sizeof(float3));
	//cudaStatus = cudaMemset(leafNodes, 0, sizeof(Particle) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "cudaMalloc_temp");
	}

	testingMemoryAccess << <(numberOfPrimitives + numOfThreads - 1) / numOfThreads, numOfThreads >> >((float*)linearMomenta, (float*)temp, numberOfPrimitives);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "testingMemoryAccess_cudaGetLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "testingMemoryAccess_cudaDeviceSynchronize");
	}



	constructLeafNodes << <(numberOfPrimitives + numOfThreads - 1) / numOfThreads, numOfThreads >> >(leafNodes, positions, numberOfPrimitives);	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "constructLeafNodes_cudaGetLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "constructLeafNodes_cudaDeviceSynchronize");
	}

	assignMomenta << <(numberOfPrimitives + numOfThreads - 1) / numOfThreads, numOfThreads >> >(leafNodes, linearMomenta, numberOfPrimitives);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "assignMomenta_cudaGetLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "assignMomenta_cudaDeviceSynchronize");
	}
	exit(1);
	//sorting procedure using cub (currently building)
	cub::DoubleBuffer<unsigned int> sortKeys; //keys to sort by - Morton codes
	cub::DoubleBuffer<Particle> sortValues; //also sort corresponding particles by key
	
	//presumambly, there is no need to allocate space for the current buffers
	sortKeys.d_buffers[0] = mortonCodes;
	sortValues.d_buffers[0] = leafNodes;

	cub::CachingDeviceAllocator  g_allocator(true);

	cudaStatus = g_allocator.DeviceAllocate((void**)&sortKeys.d_buffers[1], sizeof(unsigned int) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		return cudaFail(cudaStatus, "sortKeys_gAllocate");
	}
	
	cudaStatus = g_allocator.DeviceAllocate((void**)&sortValues.d_buffers[1], sizeof(Particle) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "sortValues_gAllocate");
	}

	// Allocate temporary storage
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "first call to DeviceRadixSort");
	}
	cudaStatus = g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_temp_storage);
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "cub: temporary storage alocation");
	}

	// Run sort
	//Note: why do I need to sort the particles themselves?
	//The code I found does nothing of the kind.
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_temp_storage);
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "second call to DeviceRadixSort");
	}
	
	//sort seems to be working properly

	Particle* internalNodes;	
	cudaStatus = cudaMalloc((void**)&internalNodes, (numberOfPrimitives - 1) * sizeof(Particle));
	//cudaStatus = cudaMemset(internalNodes, 0, sizeof(Particle) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess) {
		cudaFree(internalNodes);
		cudaFree(d_temp_storage);
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "cudaMalloc_internalNodes");
	}
	cudaStatus = generateHierarchy(internalNodes, sortValues.Current(), sortKeys.Current(), numberOfPrimitives);
	if (cudaStatus != cudaSuccess){
		cudaFree(internalNodes);
		cudaFree(d_temp_storage);
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "bvh_generateHierarchy");
	}

	handleCollisions << <(numberOfPrimitives + numOfThreads - 1) / numOfThreads, numOfThreads >> >(leafNodes, numberOfPrimitives);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		cudaFree(internalNodes);
		cudaFree(d_temp_storage);
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "constructLeafNodes_cudaGetLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		cudaFree(internalNodes);
		cudaFree(d_temp_storage);
		cudaFree(mortonCodes);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		return cudaFail(cudaStatus, "constructLeafNodes_cudaDeviceSynchronize");
	}
	//cleaning up
	cudaFree(internalNodes);
	cudaFree(d_temp_storage);
	cudaFree(mortonCodes);
	cudaFree(leafNodes);
	g_allocator.DeviceFree(sortKeys.d_buffers[1]);
	g_allocator.DeviceFree(sortValues.d_buffers[1]);
	//g_allocator.DeviceFree(sortKeys.d_buffers[0]);
	//g_allocator.DeviceFree(sortValues.d_buffers[0]);
	return cudaSuccess;
}

