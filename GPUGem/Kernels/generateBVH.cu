#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Primitives.h"
#include <stdio.h>
#include <iostream>

__device__ inline int MIN(int x, int y)
{
	return x < y ? x : y;
}

__device__ inline int MAX(int x, int y)
{
	return x > y ? x : y;
}

__device__ inline int longestCommonPrefix(int i, int j, int len) {
	if (0 <= j && j < len) {
		return __clz(i ^ j);
	}
	else {
		return -1;
	}
}

__device__ int2 determineRange(int numberOfPrimitives, int idx)
{
	//int d1 = __clz(idx ^ (idx + 1));
	//int d2 = __clz(idx ^ (idx - 1));
	//int d = d1 > d2 ? 1 : -1;
	int d = longestCommonPrefix(idx, idx + 1, numberOfPrimitives + 1) -
		longestCommonPrefix(idx, idx - 1, numberOfPrimitives + 1) > 0 ? 1 : -1;


	//int dmin = __clz(idx ^ (idx-d));
	int dmin = longestCommonPrefix(idx, idx - d, numberOfPrimitives + 1);
	int lmax = 2;
	//while (__clz(idx ^ (idx + lmax * d)) > dmin) lmax <<= 1;
	while (longestCommonPrefix(idx, idx + lmax * d, numberOfPrimitives + 1) > dmin) lmax <<= 1;
	
	int l = 0;
	int t = lmax >> 1;
	while (t >= 1) //at last iteration 1 >> 1 = 0 (integers)
	{
		//if (__clz(idx ^ (idx + (l + t) * d)) > dmin)
		if (longestCommonPrefix(idx, idx + (l + t) * d, numberOfPrimitives + 1) > dmin)
			l += t;
		t >>= 1;
	}

	int j = idx + l*d; //found range's other end (I think)
	//this is the point where we need to check for duplicate keys
	//we use findSplit function as a guide
	//corresponding to (first, last)
	//is this all that's necessary to deal with duplicate keys?
	/*if (sortedMortonCodes[idx] == sortedMortonCodes[j])
	{
	int2 range;
	range.x = d > 0 ? idx : (idx + j) / 2; //if d positive then start at idx
	range.y = d < 0 ? idx : (idx + j) / 2; //if d negative then stop at idx
	return range; //is a return needed?
	}*/

	//from now on we search for the split position
	//int dnode = __clz(idx ^ j);
	int dnode = longestCommonPrefix(idx, j, numberOfPrimitives + 1);

	int s = 0;
	int divider = 2; //now l is not an integer so we need proper integer arithmetic
	t = (l + (divider - 1)) / divider;
	while (t >= 1) //at last iteration 1 >> 2 = 0 (integers)
	{

		//if (__clz(idx ^ (idx + (s + t) * d)) > dnode)
		if (longestCommonPrefix(idx, idx + (s + t) * d, numberOfPrimitives + 1) > dnode)
			s += t;
		divider <<= 1;
		t = (l + (divider - 1)) / divider;
	}

	int gamma = idx + s * d + MIN(d, 0);
	int2 range;
	range.x = j;
	range.y = gamma;
	return range;

}

__global__ void constructInternalNodes(Primitive* internalNodes, Primitive* leafNodes, unsigned int* sortedMortonCodes, int numberOfPrimitives)
{
	//root is located at internalNodes[0]
	//The indices of each particular node's children are assigned
	//according to its respective split position
	//left child is L[split] if it is a leaf or I[split] otherwise
	//similarly right child is L[split+1] if it is a leaf or I[split+1] otherwise
	//This way the index of every internal node coincides with either its first
	//or its last key
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numberOfPrimitives - 1)
		return;

	int2 range = determineRange(numberOfPrimitives, idx);
	int j = range.x;
	int gamma = range.y;

	Primitive *leftChild = (MIN(idx, j) == gamma ? (leafNodes + gamma) : (internalNodes + gamma));
	(internalNodes + idx)->left = leftChild;
	leftChild->parent = (internalNodes + idx);

	Primitive *rightChild = (MAX(idx, j) == gamma + 1 ? (leafNodes + gamma + 1) : (internalNodes + gamma + 1));
	(internalNodes + idx)->right = rightChild;
	rightChild->parent = (internalNodes + idx);

	internalNodes[idx].isLeaf = false;

	(internalNodes + idx)->leftmost = MIN(idx, j);
	(internalNodes + idx)->rightmost = MAX(idx, j);
}

/*
[SOLVED]
Known issue: this kernel fails and returns cudaError(4): cudaErrorLaunchFailure.
Common causes include dereferencing an invalid pointer and accessing out of bounds
memory.
Notes: 
a)	Is it possible that not all leaf nodes are assigned a parent?
An if (node == NULL) return; should resolve that question.
Make sure to initialize all leafnodes with a null parent.

Update: Intuition was correct. It appears that not all leaf nodes are assigned
a parent during tree construction. There must be a bug or a mistake in my algorithm.
I need to re-check radix tree creation.

New update: I was partially wrong. Apparently leaf nodes DO have a parent. But not
all INTERNAL nodes do (removing the checkpoint at the begining does not cause pro-
blems but doing so inside the recursive loop does). Apparently using that guy's
function, which includes simple error-checking did the trick.

Note: I will also stick with atomics since they seem to be working and it is what 
Karras recommends.
*/
__global__ void assignPrimitives(Primitive *internalNodes, Primitive *leafNodes, int numberOfPrimitives, int *nodeCounter)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numberOfPrimitives)
		return;
	Primitive *node = (leafNodes + idx); //start at each leaf
	node = node->parent;
	//if (node == NULL) return;
	int currentIndex = node - internalNodes;
	int res = atomicAdd(nodeCounter + currentIndex, 1);
	while (1)
	{
		//attempting to use atomics instead of propagating only right childs
		//Update: using atomics did not solve the bug. Will only attempt reverting to
		//previous hack after resolving the issue.
		/*if (res == 0) {
			return;
		}*/
		//if this is a left child and a right child does indeed exist
		//this is used because only right childs should access their parent
		//to avoid read write conflicts
		//Karras proposes that this is done using atomic add
		//if errors persist then I will change this to recommended version
		/*if (node == node->parent->left && node->parent->right != NULL)
		{
			return;
		}*/

		//after this point only right childs should continue
		//or left childs only when no right child exists
		//node = node->parent;
		//if either of the children is NULL then the parent will have the same
		//primitive as its only child
		Primitive *lChild = node->left != NULL ? node->left : node->right;
		Primitive *rChild = node->right != NULL ? node->right : node->left;
		//centroid is mid point between centroids of children
		//this is only true for sphere-modeled particles
		//needs to be updated if I use other primitives
		node->centroid.x = (lChild->centroid.x + rChild->centroid.x) / 2;
		node->centroid.y = (lChild->centroid.y + rChild->centroid.y) / 2;
		node->centroid.z = (lChild->centroid.z + rChild->centroid.z) / 2;
		//__fsqrt_rd is a CUDA builtin function which computes the root square of a
		//number rounding down
		float r1 = lChild->radius + 
			__fsqrt_rd( (lChild->centroid.x - node->centroid.x) * (lChild->centroid.x - node->centroid.x) +
			(lChild->centroid.y - node->centroid.z) * (lChild->centroid.y - node->centroid.y) +
			(lChild->centroid.y - node->centroid.z) * (lChild->centroid.z - node->centroid.z) );
		float r2 = rChild->radius + 
			__fsqrt_rd( (rChild->centroid.x - node->centroid.x) * (rChild->centroid.x - node->centroid.x) +
			(rChild->centroid.y - node->centroid.z) * (rChild->centroid.y - node->centroid.y) +
			(rChild->centroid.y - node->centroid.z) * (rChild->centroid.z - node->centroid.z) );

		node->radius = MAX(r1, r2);

		//after assigning primitives we also need to compute rightmost leafs for each subtree
		//if I am to use this configuration (i.e. assigning the rightmost leaf of each child)
		//then I will have to make sure that leafnodes report their rightmost leaf as themselves
		//Done. What about NULLs, i.e. when no children exist
		//if either of the children are NULL then that subtree is also NULL
		//will need to check that when I check for subtrees
		node->rightmost = node->right != NULL ? node->right->rightmost : 0;
		node->leftmost = node->left != NULL ? node->left->rightmost : 0;
		
		if (node == internalNodes)
			return; //return after handling root
		node = node->parent;
		//if (node == NULL) return;
		currentIndex = node - internalNodes;
		res = atomicAdd(nodeCounter + currentIndex, 1);
	}

}

cudaError_t generateHierarchy(Primitive *internalNodes,
	Primitive* leafNodes,
	unsigned int* sortedMortonCodes,
	const int& numberOfPrimitives,
	const int& numberOfThreads)
{
	//these are to be allocated on the GPU
	cudaError_t cudaStatus;

	//launch for numberOfPrimitives - 1
	//total number of internal nodes for a bvh with N leaves is N-1
	constructInternalNodes << <(numberOfPrimitives - 1 + numberOfThreads - 1) / numberOfThreads, numberOfThreads >> >(internalNodes, leafNodes, sortedMortonCodes, numberOfPrimitives);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		goto Error;

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
		goto Error;
	
	int *nodeCounter;
	cudaStatus = cudaMalloc(&nodeCounter, sizeof(int) * numberOfPrimitives);
	cudaStatus = cudaMemset(nodeCounter, 0, sizeof(int) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess) 
		goto Error;
	//assign primitives to internal nodes
	//launch kernel for each leaf node
	//each thread works each way recursively to the top
	assignPrimitives << <(numberOfPrimitives + numberOfThreads - 1) / numberOfThreads, numberOfThreads >> >(internalNodes, leafNodes, numberOfPrimitives, nodeCounter);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		goto Error;

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		goto Error;

Error:
	//exit(1);
	if (nodeCounter)
		cudaFree(nodeCounter);
	// Node 0 is the root.
	return cudaSuccess;
}