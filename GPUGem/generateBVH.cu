#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Primitives.h"
#include <stdio.h>

cudaError_t cudaFail(cudaError_t cudaStatus, char *funcName);
//custom __popc implementation because __popc requires 
//compute capability 5+ (sm 20)
__device__ inline int bitCount(int a)
{
	int numBits = 0;
	while (a != 0) {
		if (a & 0x1)  numBits++;
		a = a >> 1;
	}
	return numBits;
}

__device__ inline int MIN(int x, int y)
{
	return x < y ? x : y;
}

__device__ inline int MAX(int x, int y)
{
	return x > y ? x : y;
}

//determine range using Terro Karras' algorithm presented in
//Maximizing Parallelism in the Construction of BVHs, Octrees and k-d Trees
__device__ int2 determineRange(Particle *internalNodes, Particle *leafnodes, unsigned int* sortedMortonCodes, int numObjects, int idx)
{
	int d;
	int d1 = bitCount(sortedMortonCodes[idx] & sortedMortonCodes[idx + 1]);
	int d2 = bitCount(sortedMortonCodes[idx] & sortedMortonCodes[idx - 1]);
	d = d1 > d2 ? 1 : -1;
	
	
	int dmin = bitCount(sortedMortonCodes[idx] & sortedMortonCodes[idx - d]);
	int lmax = 128;
	while (bitCount(sortedMortonCodes[idx] & sortedMortonCodes[idx + lmax * d]) > dmin) lmax << 2;
	
	int l = 0;
	int t = lmax >> 1;
	while (t > 0) //at last iteration 1 >> 1 = 0 (integers)
	{
		if (bitCount(sortedMortonCodes[idx] & sortedMortonCodes[idx + lmax * d]) > dmin)
			l += t;
		t >> 1;
	}
	int j = idx + l*d; //found range's other end (I think)
	//this is the point where we need to check for duplicate keys
	//we use findSplit function as a guide
	//corresponding to (first, last)
	//is this all that's necessary to deal with duplicate keys?
	if (sortedMortonCodes[idx] == sortedMortonCodes[j])
	{
		int2 range;
		range.x = d > 0 ? idx : (idx + j) / 2; //if d positive then start at idx
		range.y = d < 0 ? idx : (idx + j) / 2; //if d negative then stop at idx
	}

	//from now on we search for the split position
	int dnode = bitCount(sortedMortonCodes[idx] & sortedMortonCodes[j]);
	int s = 0;
	t = lmax >> 1;
	while (t > 0) //at last iteration 1 >> 2 = 0 (integers)
	{
		if (bitCount(sortedMortonCodes[idx] & sortedMortonCodes[idx + (s+t) * d]) > dnode)
			s += t;
		t >> 1;
	}
	int gamma = idx + s*d + MIN(d, 0);
	int2 range;
	range.x = j;
	range.y = gamma;
	return range;
	//(internalNodes + idx)->left = MIN(idx, j) == gamma ? (leafnodes + gamma) : (internalNodes + gamma);
	//(internalNodes + idx)->right = MAX(idx, j) == gamma + 1 ? (leafnodes + gamma + 1) : (internalNodes + gamma + 1);

}

int findSplit(unsigned int* sortedMortonCodes, int first, int last)
{
	// Identical Morton codes => split the range in the middle.

	unsigned int firstCode = sortedMortonCodes[first];
	unsigned int lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

	//int commonPrefix = __clz(firstCode ^ lastCode);
	int commonPrefix = 31 - floor(log2(firstCode ^ lastCode));

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.

	int split = first; // initial guess
	int step = last - first;

	do
	{
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last)
		{
			unsigned int splitCode = sortedMortonCodes[newSplit];
			//int splitPrefix = __clz(firstCode ^ splitCode);
			int splitPrefix = 31 - floor(log2(firstCode ^ splitCode));
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}

__global__ void constructInternalNodes(Particle* internalNodes, Particle* leafNodes, unsigned int* sortedMortonCodes, int numObjects)
{
	//root is located at internalNodes[0]
	//The indices of each particular node's children are assigned
	//according to its respective split position
	//left child is L[split] if it is a leaf or I[split] otherwise
	//similarly right child is L[split+1] if it is a leaf or I[split+1] otherwise
	//This way the index of every internal node coincides with either its first
	//or its last key
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int2 range = determineRange(internalNodes, leafNodes, sortedMortonCodes, numObjects, idx);
	int j = range.x;
	int gamma = range.y;

	//Update by Andreas:
	//No need to explicitly find split
	//determineRange returns indices of left and right childs (sort of)

	// Determine where to split the range.
	//int split = findSplit(sortedMortonCodes, first, last);

	// Select childA.

	/*Particle* childA;
	if (split == first)
		childA = &leafNodes[split];
	else
		childA = &internalNodes[split];

	// Select childB.

	Particle* childB;
	if (split + 1 == last)
		childB = &leafNodes[split + 1];
	else
		childB = &internalNodes[split + 1];

	// Record parent-child relationships.

	internalNodes[idx].left = childA;
	internalNodes[idx].right = childB;
	childA->parent = &internalNodes[idx];
	childB->parent = &internalNodes[idx];*/

	
	Particle *leftChild = MIN(idx, j) == gamma ? (leafNodes + gamma) : (internalNodes + gamma);
	(internalNodes + idx)->left = leftChild;
	leftChild->parent = (internalNodes + idx);
	
	Particle *rightChild = MAX(idx, j) == gamma + 1 ? (leafNodes + gamma + 1) : (internalNodes + gamma + 1);
	(internalNodes + idx)->right = rightChild;
	rightChild->parent = (internalNodes + idx);
	internalNodes[idx].isLeaf = true;
}

__global__ void assignPrimitives(Particle *internalNodes, Particle *leafNodes, int numObjects)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numObjects)
	{
		Particle *node = (leafNodes + idx);
		while (node != internalNodes)
		{
			//if this is a left child and a right child does indeed exist
			if (node == node->parent->left && node->parent->right != NULL)
			{
				break;
			}
			node = node->parent;
			//if either of the children is NULL then the parent will have the same
			//primitive its only child
			Particle *lChild = node->left != NULL ? node->left : node->right;
			Particle *rChild = node->right != NULL ? node->right : node->left;
			//centroid is mid point between centroids of children
			//this is only true for sphere-modeled particles
			//needs to be updated if I use other primitives
			node->centroid.x = (lChild->centroid.x + rChild->centroid.x) / 2;
			node->centroid.y = (lChild->centroid.y + rChild->centroid.y) / 2;
			node->centroid.z = (lChild->centroid.z + rChild->centroid.z) / 2;
			//__fsqrt_rd is a CUDA builtin function which computes the root square of a
			//number rounding down
			float r1 = lChild->radius + __fsqrt_rd((lChild->centroid.x - node->centroid.x)*(lChild->centroid.x - node->centroid.x) +
				(lChild->centroid.y - node->centroid.z)*(lChild->centroid.y - node->centroid.y) +
				(lChild->centroid.y - node->centroid.z)*(lChild->centroid.z - node->centroid.z));
			float r2 = rChild->radius + __fsqrt_rd((rChild->centroid.x - node->centroid.x)*(rChild->centroid.x - node->centroid.x) +
				(rChild->centroid.y - node->centroid.z)*(rChild->centroid.y - node->centroid.y) +
				(rChild->centroid.y - node->centroid.z)*(rChild->centroid.z - node->centroid.z));

			node->radius = MAX(r1, r2);

			//after assigning primitives we also need to compute rightmost leafs for each subtree
			//if I am to use this configuration (i.e. assigning the rightmost leaf of each child)
			//then I will have to make sure that leafnodes report their rightmost leaf as themselves
			//Done. What about NULLs, i.e. when no children exist
			//if either of the children are NULL then that subtree is also NULL
			//will need to check that when I check for subtrees
			node->rightmost = node->right != NULL ? node->right->rightmost : NULL;
			node->leftmost = node->left != NULL ? node->left->rightmost : NULL;
		}
	}

}

cudaError_t generateHierarchy(Particle *internalNodes,
	Particle* leafNodes,
	unsigned int* sortedMortonCodes,
	int           numObjects)
{
	//these are to be allocated on the GPU
	//LeafNode* leafNodes = new LeafNode[numObjects];
	//InternalNode* internalNodes = new InternalNode[numObjects - 1];
	cudaError_t cudaStatus;

	/*cudaStatus = cudaMalloc((void**)&internalNodes, numObjects * sizeof(Particle));
	if (cudaStatus != cudaSuccess) {
		cudaFree(internalNodes);
		return cudaFail(cudaStatus, "cudaMalloc_internalNodes");
	}*/
	// Construct leaf nodes.
	// Note: This step can be avoided by storing
	// the tree in a slightly different way.
	//for (int idx = 0; idx < numObjects; idx++) // in parallel
	//leafNodes[idx].id = sortedObjectIDs[idx];
	int numOfThreads = 512;

	//launch for numobjects - 1
	//total number of internal nodes for a bvh with N leaves is N-1
	constructInternalNodes << <(numObjects - 1 + numOfThreads - 1) / numOfThreads, numOfThreads >> >(internalNodes, leafNodes, sortedMortonCodes, numObjects);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cudaFree(internalNodes);
		return cudaFail(cudaStatus, "constructInternalNodes_cudaGetLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cudaFree(internalNodes);
		return cudaFail(cudaStatus, "constructInternalNodes_cudaDeviceSynchronize");
	}

	//assign primitives to internal nodes
	//launch kernel for each leaf node
	//each thread works each way recursively to the top
	assignPrimitives << <(numObjects + numOfThreads - 1) / numOfThreads, numOfThreads >> >(internalNodes, leafNodes, numObjects);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cudaFree(internalNodes);
		return cudaFail(cudaStatus, "assignPrimitives_cudaGetLastError");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cudaFree(internalNodes);
		return cudaFail(cudaStatus, "assignPrimitives_cudaDeviceSynchronize");
	}

	// Construct internal nodes.

	/*for (int idx = 0; idx < numObjects - 1; idx++) // in parallel
	{
		// Find out which range of objects the node corresponds to.
		// (This is where the magic happens!)
	}*/

	// Node 0 is the root.
	return cudaSuccess;
}