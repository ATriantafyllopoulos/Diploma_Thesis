#include "auxiliaryKernels.cuh"
/*
routine is called before BVH is created
leaf node primitives are yet unsorted
radius and mass are currently hard-coded -> 1 [OPEN]
Make radius parametric. Design interface to input parameters. [OPEN]
*/
__global__ void constructLeafNodesKernel(Primitive* leafNodes,
	float3 *positions,
	float3 *linearMomenta,
	const int numberOfPrimitives)
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
	leafNodes[index].linearMomentum = linearMomenta[index];
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
	__syncthreads();

	leafNodes[idx].linearMomentum.x = (sMomenta)[threadIdx.x];
	leafNodes[idx + 512].linearMomentum.y = (sMomenta)[threadIdx.x + 512];
	leafNodes[idx + 1024].linearMomentum.z = (sMomenta)[threadIdx.x + 1024];*/
	//leafNodes[index].angularMomentum = angularMomentums[index];
	//leafNodes[index].quaternion = quaternions[index];
}

cudaError_t constructLeafNodes(Primitive* leafNodes,
	float3 *positions,
	float3 *linearMomenta,
	const int &numberOfPrimitives,
	const int &numberOfThreads)
{
	const int numberOfBlocks = (numberOfPrimitives + numberOfThreads - 1) / numberOfThreads;
	constructLeafNodesKernel << < numberOfBlocks, numberOfThreads >> >(leafNodes, positions, linearMomenta, numberOfPrimitives);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return cudaSuccess;
	cudaStatus = cudaDeviceSynchronize();
	return cudaStatus;
}