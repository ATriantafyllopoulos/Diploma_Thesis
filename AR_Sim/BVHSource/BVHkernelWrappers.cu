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

cudaError_t findAABB(float4 *positions, float3 *d_out, int numberOfPrimitives);
__global__ void generateMortonCodes(float4 *positions, unsigned int *mortonCodes, int *indices, const int numberOfPrimitives,
	float3 *minPos, float3 *maxPos);

template <typename BoundingVolume>
__global__
void staticCollideBVH(float4 *positions,
float4 *vel,
TreeNode<BoundingVolume> *treeNodes,
TreeNode<BoundingVolume> *treeLeaves,
uint    numParticles,
SimParams params);

template <typename BoundingVolume>
__global__ void kernelConstructLeafNodes(int len, TreeNode<BoundingVolume> *treeLeaves,
	int *sorted_geometry_indices, float4 *positions, float particleRadius);

template <typename BoundingVolume>
__global__ void kernelConstructInternalNodes(int len, TreeNode<BoundingVolume> *treeNodes, TreeNode<BoundingVolume> *treeLeaves, int *nodeCounter);

template <typename BoundingVolume>
__global__ void kernelConstructRadixTree(int len,
	TreeNode<BoundingVolume> *radixTreeNodes,
	TreeNode<BoundingVolume> *radixTreeLeaves,
	unsigned int *sortedMortoncodes);

template <typename BoundingVolume>
__global__
void collideBVH(float4 *color,
float4 *vel,
TreeNode<BoundingVolume> *treeNodes,    // input: sorted particle indices
TreeNode<BoundingVolume> *treeLeaves,
uint    numParticles,
SimParams params);

template <typename BoundingVolume>
cudaError_t constructRadixTree(TreeNode<BoundingVolume> **cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> **cudaDeviceTreeLeaves,
	unsigned int *sortedMortoncodes,
	int numberOfPrimitives,
	int numberOfThreads)
{
	int internalNodes = numberOfPrimitives - 1;

	// Configure GPU running parameters
	dim3 blockDim(numberOfThreads, 1);
	dim3 gridDim((internalNodes + numberOfThreads - 1) / numberOfThreads, 1);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Launch the radix tree construction kernel
	kernelConstructRadixTree<BoundingVolume> << <gridDim, blockDim >> >(internalNodes,
		*cudaDeviceTreeNodes, *cudaDeviceTreeLeaves, sortedMortoncodes);

	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	return cudaSuccess;
}

template <typename BoundingVolume>
cudaError_t constructBVHTree(TreeNode<BoundingVolume> **cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> **cudaDeviceTreeLeaves,
	float *positions,
	float particleRadius,
	int *sorted_geometry_indices,
	unsigned int *sortedMortonCodes,
	int numberOfPrimitives,
	int numberOfThreads) {

	
	// nodeCounter makes sure that only 1 thread gets to work on a node
	// in BVH construction kernel
	int *nodeCounter;

	checkCudaErrors(cudaMalloc(&nodeCounter, sizeof(int) * (numberOfPrimitives - 1)));
	checkCudaErrors(cudaMemset(nodeCounter, 0, sizeof(int) * (numberOfPrimitives - 1)));

	// Configure GPU running parameters
	dim3 blockDim(numberOfThreads, 1);
	dim3 gridDim((numberOfPrimitives + numberOfThreads - 1) / numberOfThreads, 1);
	TreeNode<BoundingVolume> *intern = new TreeNode<BoundingVolume>[numberOfPrimitives - 1];
	TreeNode<BoundingVolume> *leaves = new TreeNode<BoundingVolume>[numberOfPrimitives];
	checkCudaErrors(cudaMemcpy(intern, *cudaDeviceTreeNodes, (numberOfPrimitives - 1) * sizeof(TreeNode<BoundingVolume>), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(leaves, *cudaDeviceTreeLeaves, (numberOfPrimitives)* sizeof(TreeNode<BoundingVolume>), cudaMemcpyDeviceToHost));


	kernelConstructLeafNodes << <gridDim, blockDim >> >(numberOfPrimitives, *cudaDeviceTreeLeaves,
		sorted_geometry_indices, (float4 *)positions, particleRadius);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		for (int index = 0; index < numberOfPrimitives; index++)
		{
			TreeNode<BoundingVolume> node = leaves[index];
			int iterations = 0;
			bool problem = false;
			//std::cout << "Testing leaf node: " << index << std::endl;
			//std::cout << "Node's parent: " << node.parentIndex << std::endl;
			/*while (node != intern && !problem)
			{
			std::cout << "Node's parent: " << node->parentIndex << std::endl;
			node = intern + node->parentIndex;
			if (++iterations == numberOfPrimitives)problem = true;
			}*/
			if (problem)
				std::cout << "Problem @ node: " << index << std::endl;
		}
	}
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	kernelConstructInternalNodes<BoundingVolume> << <gridDim, blockDim >> >(numberOfPrimitives, *cudaDeviceTreeNodes,
		*cudaDeviceTreeLeaves, nodeCounter);

	

	delete intern;
	delete leaves;
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(nodeCounter));
	
	return cudaSuccess;
	
}

template <typename BoundingVolume>
cudaError_t createCUDAarrays(float4 *positions,
	TreeNode<BoundingVolume> **cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> **cudaDeviceTreeLeaves,
	unsigned int **mortonCodes,
	int **indices,
	unsigned int **sortedMortonCodes,
	int **sortedIndices,
	int numberOfPrimitives,
	int numberOfThreads)
{
	//malloc arrays
	checkCudaErrors(cudaMalloc((void**)cudaDeviceTreeNodes, sizeof(TreeNode<BoundingVolume>) * (numberOfPrimitives - 1)));
	checkCudaErrors(cudaMalloc((void**)cudaDeviceTreeLeaves, sizeof(TreeNode<BoundingVolume>) * numberOfPrimitives));
	checkCudaErrors(cudaMalloc((void**)mortonCodes, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMalloc((void**)indices, sizeof(int) * numberOfPrimitives));
	checkCudaErrors(cudaMalloc((void**)sortedMortonCodes, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMalloc((void**)sortedIndices, sizeof(int) * numberOfPrimitives));
	
	checkCudaErrors(cudaMemset(*cudaDeviceTreeNodes, 0, sizeof(TreeNode<BoundingVolume>) * (numberOfPrimitives - 1)));
	checkCudaErrors(cudaMemset(*cudaDeviceTreeLeaves, ~0, sizeof(TreeNode<BoundingVolume>) * numberOfPrimitives));
	checkCudaErrors(cudaMemset(*mortonCodes, 0, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMemset(*indices, 0, sizeof(int) * numberOfPrimitives));
	checkCudaErrors(cudaMemset(*sortedMortonCodes, 0, sizeof(unsigned int) * numberOfPrimitives));
	checkCudaErrors(cudaMemset(*sortedIndices, 0, sizeof(int) * numberOfPrimitives));

	return cudaSuccess;
}

template <typename BoundingVolume>
cudaError_t collisionDetectionAndHandling(float4 *color, 
	float *vel,
	TreeNode<BoundingVolume> *cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> *cudaDeviceTreeLeaves,
	int numParticles,
	int numberOfThreads,
	SimParams params)
{
	dim3 blockDim(numberOfThreads, 1);
	dim3 gridDim((numParticles + numberOfThreads - 1) / numberOfThreads, 1);

	collideBVH<BoundingVolume> << <gridDim, blockDim >> >(color,
		(float4 *)vel,
		cudaDeviceTreeNodes,    // input: sorted particle indices
		cudaDeviceTreeLeaves,
		numParticles, 
		params);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	return cudaSuccess;
}

template <typename BoundingVolume>
cudaError_t staticCollisionDetection(float *positions,
	float *vel,
	TreeNode<BoundingVolume> *treeNodes,
	TreeNode<BoundingVolume> *treeLeaves,
	int numParticles,
	int numberOfThreads,
	SimParams params)
{
	dim3 blockDim(numberOfThreads, 1);
	dim3 gridDim((numParticles + numberOfThreads - 1) / numberOfThreads, 1);

	staticCollideBVH<BoundingVolume> << <gridDim, blockDim >> >((float4 *)positions,
		(float4 *)vel,
		treeNodes,
		treeLeaves,
		numParticles,
		params);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	return cudaSuccess;
}


template cudaError_t createCUDAarrays(float4 *positions,
	TreeNode<AABB> **cudaDeviceTreeNodes,
	TreeNode<AABB> **cudaDeviceTreeLeaves,
	unsigned int **mortonCodes,
	int **indices,
	unsigned int **sortedMortonCodes,
	int **sortedIndices,
	int numberOfPrimitives,
	int numberOfThreads);

template cudaError_t constructBVHTree(TreeNode<AABB> **cudaDeviceTreeNodes,
	TreeNode<AABB> **cudaDeviceTreeLeaves,
	float *positions,
	float particleRadius,
	int *sorted_geometry_indices,
	unsigned int *sortedMortonCodes,
	int numberOfPrimitives,
	int numberOfThreads);

template cudaError_t constructRadixTree(TreeNode<AABB> **cudaDeviceTreeNodes,
	TreeNode<AABB> **cudaDeviceTreeLeaves,
	unsigned int *sortedMortoncodes,
	int numberOfPrimitives,
	int numberOfThreads);

template cudaError_t collisionDetectionAndHandling(float4 *color,
	float *vel,
	TreeNode<AABB> *cudaDeviceTreeNodes,
	TreeNode<AABB> *cudaDeviceTreeLeaves,
	int numParticles,
	int numberOfThreads,
	SimParams params);

template cudaError_t staticCollisionDetection(float *positions,
	float *vel,
	TreeNode<AABB> *treeNodes,
	TreeNode<AABB> *treeLeaves,
	int numParticles,
	int numberOfThreads,
	SimParams params);