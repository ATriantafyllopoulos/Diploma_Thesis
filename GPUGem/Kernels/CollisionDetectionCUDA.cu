#include "auxiliaryKernels.cuh"
#include <algorithm>
#include <iostream>

//function prototypes: will be moved over to PhysicsEngineCUDA in the future
cudaError_t update(Primitive* leafNodes,
	const float &timeStep,
	const int &numberOfPrimitives,
	const int &numberOfThreads);

cudaError_t handleCollisions(Primitive* leafNodes,
	const float &timeStep,
	const int &numberOfPrimitives,
	const int numberOfThreads);

cudaError_t generateHierarchy(Primitive *internalNodes,
	Primitive* leafNodes,
	unsigned int* sortedMortonCodes,
	const int& numberOfPrimitives,
	const int& numberOfThreads);

cudaError_t createMortonCodes(float3 *positions,
	unsigned int *mortonCodes,
	const int &numberOfPrimitives,
	const int &numberOfThreads);

cudaError_t sortMortonCodes(Primitive* leafNodes,
	unsigned int *mortonCodes,
	Primitive **sortedLeafNodes,
	unsigned int **sortedMortonCodes,
	const int &numberOfPrimitives);

cudaError_t constructLeafNodes(Primitive* leafNodes,
	float3 *positions,
	float3 *linearMomenta,
	const int &numberOfPrimitives,
	const int &numberOfThreads);


__device__ void traverseIterative(Primitive *root,
	Primitive* queryLeaf)
{
	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	Primitive* stack[64]; //AT: Is 64 the correct size to use?
	Primitive** stackPtr = stack;
	//when stack is empty thread will return
	*stackPtr++ = NULL; // push NULL at beginning

	// Traverse nodes starting from the root.
	Primitive* node = root;
	do
	{
		// Check each child node for overlap.
		Primitive* childL = node->left;
		Primitive* childR = node->right;
		bool overlapL = (checkOverlap(queryLeaf, childL));
		bool overlapR = (checkOverlap(queryLeaf, childR));

		if (node->leftmost <= queryLeaf->id)
			overlapL = false;

		if (node->rightmost <= queryLeaf->id)
			overlapR = false;
		
		// Query overlaps a leaf node => report collision.
		if (overlapL && childL->isLeaf)
			queryLeaf->collisions[queryLeaf->collisionCounter++] = childL;
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

__global__ void findPotentialCollisions(Primitive *internalNodes, Primitive *leafNodes, int numOfLeaves)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= numOfLeaves)
		return;

	Primitive *leaf = leafNodes + index;
	traverseIterative(internalNodes, leaf);
}

__global__ void copyBackToMemory(Primitive *leafNodes, float3 *positions, float3 *linearMomenta, const int numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfPrimitives)
		return;

	positions[index] = leafNodes[index].centroid;
	linearMomenta[index] = leafNodes[index].linearMomentum;
}

/*
Potential error: sorting the leaf nodes by value, using Morton codes as keys, is not done on the code I found. [OPEN]
*/
cudaError_t detectCollisions(float3 *positions, 
	float3 **linearMomenta, 
	const float &timeStep, 
	const int &numberOfPrimitives, 
	const int &numberOfThreads)
{
	//pre-declaring all variables so we can clean up all of them after calling goto
	unsigned int *mortonCodes;
	Primitive *leafNodes;
	Primitive* internalNodes;
	Primitive *sortedLeafNodes; //used only to get sorted leaf nodes from sort kernel
	unsigned int *sortedMortonCodes; //used only to get sorted Morton codes from sort kernel
	cudaError_t cudaStatus;
	
	int numberOfBlocks = (numberOfPrimitives + numberOfThreads - 1) / numberOfThreads;

	cudaStatus = cudaMalloc((void**)&mortonCodes, numberOfPrimitives * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "cudaMalloc_mortonCodes" << std::endl;
		goto Error;
	}
	//assign a Morton code to each primitive
	//launch all objects
	cudaStatus = createMortonCodes(positions, mortonCodes, numberOfPrimitives, numberOfThreads);
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Callback function: " << "createMortonCodes" << std::endl;
		goto Error;
	}

	//create leaf nodes here
	//then sort them using their morton codes as keys
	//and pass them as argument to the BVH hierarchy creation routine	
	cudaStatus = cudaMalloc((void**)&leafNodes, numberOfPrimitives * sizeof(Primitive));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "cudaMalloc_leafNodes" << std::endl;
		goto Error;
	}

	cudaStatus = constructLeafNodes(leafNodes, positions, *linearMomenta, numberOfPrimitives, numberOfThreads);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "constructLeafNodes" << std::endl;
		goto Error;
	}
	

	cudaStatus = sortMortonCodes(leafNodes, mortonCodes, &sortedLeafNodes, &sortedMortonCodes, numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "sortMortonCodes" << std::endl;
		goto Error;
	}
	//sort seems to be working properly

	//allocate N-1 internal nodes
	//this is the exact number of internal nodes in a BVH tree
	//according to Terro Karras of course
	cudaStatus = cudaMalloc((void**)&internalNodes, (numberOfPrimitives - 1) * sizeof(Primitive));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "cudaMalloc_internalNodes" << std::endl;
		goto Error;
	}

	//generate BVH using sorted leaves and keys
	cudaStatus = generateHierarchy(internalNodes, sortedLeafNodes, sortedMortonCodes, numberOfPrimitives, numberOfThreads);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "bvh_generateHierarchy" << std::endl;
		goto Error;
	}
	//
	//
	//
	//collision detection IS missing
	//
	//
	//
	cudaStatus = handleCollisions(leafNodes, timeStep, numberOfPrimitives, numberOfThreads);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "handleCollisions" << std::endl;
		goto Error;
	}


	cudaStatus = update(leafNodes, timeStep, numberOfPrimitives, numberOfThreads);
	if (cudaStatus != cudaSuccess){
		std::cout << "Callback function: " << "update" << std::endl;
		goto Error;
	}

	copyBackToMemory << <numberOfBlocks, numberOfThreads >> >(leafNodes, positions, *linearMomenta, numberOfPrimitives);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Callback function: " << "copyBackToMemory_cudaGetLastError" << std::endl;
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		std::cout << "Callback function: " << "copyBackToMemory_cudaDeviceSynchronize" << std::endl;
		goto Error;
	}

Error:
	//cleaning up
	if (internalNodes)
		cudaFree(internalNodes);
	if (mortonCodes)
		cudaFree(mortonCodes);
	if (leafNodes)
		cudaFree(leafNodes);
	if (sortedMortonCodes)
		cudaFree(sortedMortonCodes);
	if (sortedLeafNodes)
		cudaFree(sortedLeafNodes);
	return cudaStatus;
}
