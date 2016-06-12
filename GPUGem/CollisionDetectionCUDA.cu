#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Primitives.h"
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
Particle* generateHierarchy(Particle* leafNodes,
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

CollisionList* detectCollisions(float3 *positions, int numObjects)
{
	unsigned int *mortonCodes;
	cudaError_t cudaStatus = cudaMalloc((void**)&mortonCodes, numObjects * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc_detectCollisions failed!");
		goto Error;
	}

	int numOfThreads = 512; //maximum amount of threads supported by laptop
	//assign a Morton code to each primitive
	//launch all objects
	generateMortonCodes << <(numObjects + numOfThreads - 1) / numOfThreads, numOfThreads >> >(positions, mortonCodes, numObjects);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "generateMortonCodes launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching generateMortonCodes!\n", cudaStatus);
		goto Error;
	}

	CollisionList *collisions;
	cudaStatus = cudaMalloc((void**)&collisions, numObjects * sizeof(CollisionList));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc_collisions failed!");
		goto Error;
	}

	//create leaf nodes here
	//then sort them using their morton codes as keys
	//and pass them as argument to the BVH hierarchy creation routine
	Particle *leafNodes;
	cudaStatus = cudaMalloc((void**)&leafNodes, numObjects * sizeof(Particle));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc_leafNodes_detectCollisions failed!");
		goto Error;
	}
	constructLeafNodes << <(numObjects + numOfThreads - 1) / numOfThreads, numOfThreads >> >(leafNodes, positions, numObjects);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "constructLeafNodes launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching constructLeafNodes!\n", cudaStatus);
		goto Error;
	}
		

	//sorting morton codes
	//no need to store them in a different buffer
	//also sorting leaf nodes
	//each leaf node has a field containing its original address
	//thrust::device_ptr<Particle> devParticlePtr(leafNodes);
	thrust::device_ptr<unsigned int> devMortonCodesPtr(mortonCodes);

	thrust::device_ptr<float3> devTest(positions);
	thrust::sort_by_key(devMortonCodesPtr, devMortonCodesPtr + numObjects, devTest);
	
	float3 *test = thrust::raw_pointer_cast(devTest);
	//leafNodes = thrust::raw_pointer_cast(devParticlePtr);
	mortonCodes = thrust::raw_pointer_cast(devMortonCodesPtr);

	Particle* bvh = generateHierarchy(leafNodes, mortonCodes, numObjects);
	/*unsigned int *sortedMortonCodes;
	cudaStatus = cudaMalloc((void**)&sortedMortonCodes, numObjects * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc_detectCollisions failed!");
		goto Error;
	}*/
	return collisions;
Error:
	cudaFree(mortonCodes);
	cudaFree(collisions);
	//cudaFree(sortedMortonCodes);
	return NULL;
}

