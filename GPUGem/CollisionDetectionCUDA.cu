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

cudaError_t cudaFail(cudaError_t cudaStatus, char *funcName);
cudaError_t generateHierarchy(Particle *internalNodes,
	Particle* leafNodes,
	unsigned int* sortedMortonCodes,
	int           numObjects);
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
	*stackPtr++ = NULL; // push

	// Traverse nodes starting from the root.
	Particle* node = root;
	do
	{
		// Check each child node for overlap.
		Particle* childL = node->left;
		Particle* childR = node->right;
		bool overlapL = (checkOverlap(queryLeaf, childL));
		bool overlapR = (checkOverlap(queryLeaf, childR));

		//this is wrong
		//what exactly is a queryleaf?
		//How do I access it efficiently?
		//for now I pass it as an argument
		//Primitive* queryLeaf = bvh.getLeaf(queryObjectIdx);
		
		if (node->leftmost->id <= queryLeaf->id)
			overlapL = false;

		if (node->rightmost->id <= queryLeaf->id)
			overlapR = false;
		// Query overlaps a leaf node => report collision.
		if (overlapL && childL->isLeaf);
			//list.add(queryLeaf->id, childL->id);

		if (overlapR && childR->isLeaf);
			//list.add(queryLeaf->id, childR->id);

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

__global__ void generateMortonCodes(float3 *positions, unsigned int *mortonCodes, int numObjects)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numObjects)
	{
		mortonCodes[idx] = morton3D(positions[idx].x, positions[idx].y, positions[idx].z);
	}
}

__global__ void findPotentialCollisions(CollisionList list, Particle *internalNodes, Particle *leafNodes, int numOfLeaves)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < numOfLeaves)
	{
		Particle *leaf = leafNodes + idx;
		traverseIterative(list, internalNodes, leaf);
	}
}

/*
routine is called before BVH is created
leaf node primitives are yet unsorted
radius is currently hard-coded -> 1
*/
__global__ void constructLeafNodes(Particle* leafNodes, float3 *positions, int numObjects)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numObjects)
	{
		leafNodes[idx].id = idx; //leaf nodes are unsorted
		leafNodes[idx].isLeaf = true;
		//each leaf reports rightmost leaf of left and right sutree as itself
		leafNodes[idx].leftmost = leafNodes + idx;
		leafNodes[idx].rightmost = leafNodes + idx;

		leafNodes[idx].centroid = positions[idx];
		leafNodes[idx].radius = 1;
	}
}

cudaError_t detectCollisions(float3 *positions, int numObjects)
{
	unsigned int *mortonCodes;
	cudaError_t cudaStatus = cudaMalloc((void**)&mortonCodes, numObjects * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		cudaFree(mortonCodes);
		return cudaFail(cudaStatus, "cudaMalloc_mortonCodes");
	}

	int numOfThreads = 512; //maximum amount of threads supported by laptop
	//assign a Morton code to each primitive
	//launch all objects
	generateMortonCodes << <(numObjects + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions, mortonCodes, numObjects);

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

	/*cudaStatus = cudaMalloc((void**)&collisions, numObjects * sizeof(CollisionList));
	if (cudaStatus != cudaSuccess)  {
		cudaFree(mortonCodes);
		cudaFree(collisions);
		return cudaFail(cudaStatus, "cudaMalloc_collisions");
	}*/

	//create leaf nodes here
	//then sort them using their morton codes as keys
	//and pass them as argument to the BVH hierarchy creation routine
	Particle *leafNodes;
	cudaStatus = cudaMalloc((void**)&leafNodes, numObjects * sizeof(Particle));
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		//cudaFree(collisions);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "cudaMalloc_leafNodes");
	}
	constructLeafNodes << <(numObjects + numOfThreads - 1) / numOfThreads, numOfThreads >> >(leafNodes, positions, numObjects);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		//cudaFree(collisions);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "constructLeafNodes_cudaGetLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		//cudaFree(collisions);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "constructLeafNodes_cudaDeviceSynchronize");
	}
	//sorting morton codes using thrust (deprecated)
	//no need to store them in a different buffer
	//also sorting leaf nodes
	//each leaf node has a field containing its original address
	//thrust::device_ptr<Particle> devParticlePtr(leafNodes);
	/*thrust::device_ptr<unsigned int> devMortonCodesPtr(mortonCodes);

	thrust::device_ptr<float3> devTest(positions);
	thrust::sort_by_key(devMortonCodesPtr, devMortonCodesPtr + numObjects, devTest);
	
	float3 *test = thrust::raw_pointer_cast(devTest);
	//leafNodes = thrust::raw_pointer_cast(devParticlePtr);
	mortonCodes = thrust::raw_pointer_cast(devMortonCodesPtr);*/

	//sorting procedure using cub (currently building)
	cub::DoubleBuffer<unsigned int> sortKeys; //keys to sort by - Morton codes
	cub::DoubleBuffer<Particle> sortValues; //also sort corresponding particles by key
	
	//presumambly, there is no need to allocate space for the current buffers
	sortKeys.d_buffers[0] = mortonCodes;
	sortValues.d_buffers[0] = leafNodes;
	
	//allocate memory for alternate buffers
	//allocate memory using cub allocator
	//so many problems here
	//is it enough to allocate memory only for alternate buffers?
	//what about error checking
	/*
	cudaFree(mortonCodes);
	//cudaFree(collisions);
	g_allocator.DeviceFree(sortKeys.d_buffers[1]);
	cudaFree(leafNodes);
	*/
	cub::CachingDeviceAllocator  g_allocator(true);

	/*cudaStatus = g_allocator.DeviceAllocate((void**)&sortKeys.d_buffers[0], sizeof(unsigned int) * numObjects);
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "sortKeys_gAllocate");
	*/
	cudaStatus = g_allocator.DeviceAllocate((void**)&sortKeys.d_buffers[1], sizeof(unsigned int) * numObjects);
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "sortKeys_gAllocate");
	
	/*cudaStatus = g_allocator.DeviceAllocate((void**)&sortValues.d_buffers[0], sizeof(Particle) * numObjects);
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "sortValues_gAllocate");
	*/
	cudaStatus = g_allocator.DeviceAllocate((void**)&sortValues.d_buffers[1], sizeof(Particle) * numObjects);
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "sortValues_gAllocate");


	// Allocate temporary storage
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numObjects);
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "first call to DeviceRadixSort");
	cudaStatus = g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "first call to DeviceRadixSort");


	// Initialize device arrays
	//CubDebugExit(cudaMemcpy(sortKeys.d_buffers[sortKeys.selector], mortonCodes, sizeof(unsigned int) * numObjects, cudaMemcpyDeviceToDevice));
	//CubDebugExit(cudaMemcpy(sortValues.d_buffers[sortValues.selector], leafNodes, sizeof(Particle) * numObjects, cudaMemcpyDeviceToDevice));

	//no need to check results of sort
	/*unsigned int *h_keys = new unsigned int[numObjects];
	cudaMemcpy(h_keys, mortonCodes, sizeof(unsigned int) * numObjects, cudaMemcpyDeviceToHost);
	int compare = CompareDeviceResults(h_keys, mortonCodes, numObjects, true, true);
	if (!compare)
		std::cout << "Comparison results before sort: " << "TRUE" << std::endl;
	else
		std::cout << "Comparison results before sort: " << "FALSE" << std::endl;

	std::stable_sort(h_keys, h_keys + numObjects); //now reference keys are sorted
	*/
	// Run
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numObjects);
	if (cudaStatus != cudaSuccess)
		return cudaFail(cudaStatus, "second call to DeviceRadixSort");
	
	/*compare = CompareDeviceResults(h_keys, sortKeys.Current(), numObjects, true, false);
	if (!compare)
		std::cout << "Comparison results after sort: " << "TRUE" << std::endl;
	else
		std::cout << "Comparison results after sort: " << "FALSE" << std::endl;
	*/
	//allocate memory for internal nodes
	//DO NOT CROSS THIS BREAKPOINT UNTIL ALL OTHER ISSUES ARE RESOLVED
	Particle* internalNodes;
	//exit(1); I can now proceed below this breakpoint
	//sort seems to be working properly
	cudaStatus = cudaMalloc((void**)&internalNodes, (numObjects - 1) * sizeof(Particle));
	if (cudaStatus != cudaSuccess) {
		cudaFree(mortonCodes);
		cudaFree(internalNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "cudaMalloc_internalNodes");
	}
	cudaStatus = generateHierarchy(internalNodes, sortValues.Current(), sortKeys.Current(), numObjects);
	if (cudaStatus != cudaSuccess){
		cudaFree(mortonCodes);
		//cudaFree(collisions);
		cudaFree(leafNodes);
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
		cudaFree(leafNodes);
		return cudaFail(cudaStatus, "bvh_generateHierarchy");
	}
	/*unsigned int *sortedMortonCodes;
	cudaStatus = cudaMalloc((void**)&sortedMortonCodes, numObjects * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc_detectCollisions failed!");
		goto Error;
	}*/
	return cudaSuccess;
}

