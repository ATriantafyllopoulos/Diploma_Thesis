// Parallel BVH Interface
//#define M_PI 3.14159265359
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <math.h>
#include "helper_math.h"
#include "particleSystem.cuh"
#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif
#ifndef BVHCREATION_H
#define BVHCREATION_H

//Note: using only c++ generic types after encountering problems caused by memory alignment
//TODO: figure a way to integrate CUDA builtin types to increase efficiency
typedef struct Sphere
{
	float4 augmentedCM;
	//float x, y, z, r;

}Sphere;

typedef struct MY_ALIGN(16) AABB
{
	float3 min;
	float3 max;
}AABB;

template<typename T>
struct MY_ALIGN(16) TreeNode{

	int min;
	int max;
	int leaf;

	TreeNode *left;
	TreeNode *right;
	TreeNode *parent;
	float4 cm;
	//float cmX, cmY, cmZ;
	float radius;
	int index;
	//Bounding volume is currently hardcoded
	//TODO: find a way to switch at runtime
	//use templates?
	T boundingVolume;
};

template <typename BoundingVolume>
cudaError_t createCUDAarrays(float4 *positions,
	TreeNode<BoundingVolume> **cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> **cudaDeviceTreeLeaves,
	unsigned int **mortonCodes,
	int **indices,
	unsigned int **sortedMortonCodes,
	int **sortedIndices,
	int numberOfPrimitives,
	int numberOfThreads);

template <typename BoundingVolume>
cudaError_t constructBVHTree(TreeNode<BoundingVolume> **cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> **cudaDeviceTreeLeaves,
	float *positions,
	float particleRadius,
	int *sorted_geometry_indices,
	unsigned int *sortedMortonCodes,
	int numberOfPrimitives,
	int numberOfThreads);

template <typename BoundingVolume>
cudaError_t constructRadixTree(TreeNode<BoundingVolume> **cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> **cudaDeviceTreeLeaves,
	unsigned int *sortedMortoncodes,
	int numberOfPrimitives,
	int numberOfThreads);

template <typename BoundingVolume>
cudaError_t collisionDetectionAndHandling(float4 *color,
	float *vel,
	TreeNode<BoundingVolume> *cudaDeviceTreeNodes,
	TreeNode<BoundingVolume> *cudaDeviceTreeLeaves,
	int numParticles,
	int numberOfThreads,
	SimParams params);

template <typename BoundingVolume>
cudaError_t staticCollisionDetection(float *positions,
	float *vel,
	TreeNode<BoundingVolume> *treeNodes,
	TreeNode<BoundingVolume> *treeLeaves,
	int numParticles,
	int numberOfThreads,
	SimParams params);

cudaError_t createMortonCodes(float4 *positions,
	unsigned int **mortonCodes,
	int **indices,
	unsigned int **sortedMortonCodes,
	int **sortedIndices,
	int numberOfPrimitives,
	int numberOfThreads);


void wrapperConstructRadixTreeSoA(
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	unsigned int *sortedMortonCodes,
	int numThreads,
	int numParticles);

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
	);

void wrapperConstructInternalNodesSoA(
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int numThreads,
	int numParticles);

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
	int numThreads);

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
	int numberOfThreads);

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
	SimParams params); //simulation parameters
#endif
