/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
//#include <GL/freeglut.h>
#endif

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
//#include <cub/test/test_util.h>

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "particleSystem.cuh"
#include <time.h>

__global__
void calcHashD(uint   *gridParticleHash,  // output
uint   *gridParticleIndex, // output
float4 *pos,               // input: positions
uint    numParticles);
__global__
void reorderDataAndFindCellStartD(int *rbIndices, //index of the rigid body each particle belongs to
uint   *cellStart,        // output: cell start index
uint   *cellEnd,          // output: cell end index
float4 *sortedPos,        // output: sorted positions
float4 *sortedVel,        // output: sorted velocities
uint   *gridParticleHash, // input: sorted grid hashes
uint   *gridParticleIndex,// input: sorted particle indices
float4 *oldPos,           // input: sorted position array
float4 *oldVel,           // input: sorted velocity array
uint    numParticles);
__global__
void collideD(float4 *pForce, //total force applied to rigid body
int *rbIndices, //index of the rigid body each particle belongs to
float4 *relativePos, //particle's relative position
float4 *pTorque,  //rigid body angular momentum
float4 *color,
float4 *newVel,               // output: new velocity
float4 *oldPos,               // input: sorted positions
float4 *oldVel,               // input: sorted velocities
uint   *gridParticleIndex,    // input: sorted particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles);
__global__
void staticCollideD(
float4 *dCol,
float4 *rbForces, //total force applied to rigid body
int *rbIndices, //index of the rigid body each particle belongs to
float4 *relativePos, //particle's relative position
float4 *rbTorque,  //rigid body angular momentum
float *r_radii, //radii of all scene particles
float4 *newVel,               // output: new velocity
float4 *oldPos,               // input: sorted positions
float4 *oldVel,               // input: sorted velocities
float4 *staticPos,
uint   *gridParticleIndex,    // input: sorted particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles);
void cudaInit(int argc, char **argv)
{
	int devID;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0)
	{
		printf("No CUDA Capable devices found, exiting...\n");
		exit(EXIT_SUCCESS);
	}
}

void cudaGLInit(int argc, char **argv)
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaGLDevice(argc, (const char **)argv);
}

void allocateArray(void **devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}

void threadSync()
{
	checkCudaErrors(cudaDeviceSynchronize());
}

void copyArrayToDevice(void *device, const void *host, int offset, int size)
{
	checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
		cudaGraphicsMapFlagsNone));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
	void *ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
		*cuda_vbo_resource));
	return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void copyArrayFromDevice(void *host, const void *device,
struct cudaGraphicsResource **cuda_vbo_resource, int size)
{
	if (cuda_vbo_resource)
	{
		device = mapGLBufferObject(cuda_vbo_resource);
	}

	checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

	if (cuda_vbo_resource)
	{
		unmapGLBufferObject(*cuda_vbo_resource);
	}
}

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}


void calcHash(uint  *gridParticleHash,
	uint  *gridParticleIndex,
	float *pos,
	int    numParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// execute the kernel
	calcHashD << < numBlocks, numThreads >> >(gridParticleHash,
		gridParticleIndex,
		(float4 *)pos,
		numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed: calcHashD");
}

void reorderDataAndFindCellStart(int *rbIndices, //index of the rigid body each particle belongs to
	uint  *cellStart,
	uint  *cellEnd,
	float *sortedPos,
	float *sortedVel,
	uint  *gridParticleHash,
	uint  *gridParticleIndex,
	float *oldPos,
	float *oldVel,
	uint   numParticles,
	uint   numCells)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
#endif

	uint smemSize = sizeof(uint)*(numThreads + 1);
	reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> >(rbIndices, //index of the rigid body each particle belongs to
		//reorderDataAndFindCellStartD << < numBlocks, numThreads>> >(
		cellStart,
		cellEnd,
		(float4 *)sortedPos,
		(float4 *)sortedVel,
		gridParticleHash,
		gridParticleIndex,
		(float4 *)oldPos,
		(float4 *)oldVel,
		numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
	checkCudaErrors(cudaUnbindTexture(oldPosTex));
	checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
}

void collide(float4 *pForce, //total force applied to rigid body - per particle
	int *rbIndices, //index of the rigid body each particle belongs to
	float4 *relativePos, //particle's relative position
	float4 *pTorque,  //rigid body angular momentum - per particle
	float *color,
	float *newVel,
	float *sortedPos,
	float *sortedVel,
	uint  *gridParticleIndex,
	uint  *cellStart,
	uint  *cellEnd,
	uint   numParticles,
	uint   numCells)
{
#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

	// thread per particle
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// execute the kernel
	collideD << < numBlocks, numThreads >> >(pForce, //total force applied to rigid body
		rbIndices, //index of the rigid body each particle belongs to
		relativePos, //particle's relative position
		pTorque,  //rigid body angular momentum
		(float4 *)color,
		(float4 *)newVel,
		(float4 *)sortedPos,
		(float4 *)sortedVel,
		gridParticleIndex,
		cellStart,
		cellEnd,
		numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");

#if USE_TEX
	checkCudaErrors(cudaUnbindTexture(oldPosTex));
	checkCudaErrors(cudaUnbindTexture(oldVelTex));
	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
}


void sortParticles(uint **dGridParticleHash, uint **dGridParticleIndex, uint numParticles)
{
#define PROFILE_SORT
#ifdef PROFILE_SORT
	static float RadixSortTime = 0;
	static float totalMemTime = 0;
	static int iterations = 0;
#endif
	/*thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
	thrust::device_ptr<uint>(dGridParticleHash + numParticles),
	thrust::device_ptr<uint>(dGridParticleIndex));*/
#ifdef PROFILE_SORT
	clock_t start = clock();
#endif

	cub::DoubleBuffer<uint> sortKeys; //keys to sort by 
	cub::DoubleBuffer<uint> sortValues; //also sort corresponding particles by key

	//intermediate memory to initialize geometry indices
	uint *sortedGridParticleHash;

	//intermediate variable used to deallocate memory used to store the sorted Morton codes
	//if it is not used, we soon run out of memory
	uint *sortedGridParticleIndex;
	cudaError_t cudaStatus;

	cub::CachingDeviceAllocator  g_allocator(true);
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;



	cudaStatus = cudaMalloc((void**)&sortedGridParticleHash, sizeof(uint) * numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void**)&sortedGridParticleIndex, sizeof(uint) * numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;

	//presumambly, there is no need to allocate space for the current buffers
	sortKeys.d_buffers[0] = *dGridParticleHash;
	sortValues.d_buffers[0] = *dGridParticleIndex;

	sortKeys.d_buffers[1] = sortedGridParticleHash;
	sortValues.d_buffers[1] = sortedGridParticleIndex;

#ifdef PROFILE_SORT
	clock_t end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

	// Allocate temporary storage
#ifdef PROFILE_SORT
	start = clock();
#endif
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// Run sort
	//Note: why do I need to sort the particles themselves?
	//The code I found does nothing of the kind.
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;

#ifdef PROFILE_SORT
	end = clock();
	RadixSortTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_SORT
	start = clock();
#endif

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		goto Error;
	*dGridParticleHash = sortKeys.Current();
	sortedGridParticleHash = sortKeys.Alternate();

	*dGridParticleIndex = sortValues.Current();
	sortedGridParticleIndex = sortValues.Alternate();
Error:
	if (sortedGridParticleHash)
		cudaFree(sortedGridParticleHash);
	if (d_temp_storage)
		cudaFree(d_temp_storage);
	if (sortedGridParticleIndex)
		cudaFree(sortedGridParticleIndex);

#ifdef PROFILE_SORT
	end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_SORT
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last " << iterations << " iterations..." << std::endl;
		std::cout << "Average time spent on memory operations: " << totalMemTime / (float)iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on radix sort: " << RadixSortTime / (float)iterations << " (ms)" << std::endl;
		std::cout << std::endl;
	}
#endif
}

void sortParticlesPreallocated(
	uint **dGridParticleHash,
	uint **dGridParticleIndex,
	uint **sortedGridParticleHash,
	uint **sortedGridParticleIndex,
	uint numParticles)
{
//#define PROFILE_SORT
#ifdef PROFILE_SORT
	static float RadixSortTime = 0;
	static float totalMemTime = 0;
	static int iterations = 0;
#endif
	/*thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
	thrust::device_ptr<uint>(dGridParticleHash + numParticles),
	thrust::device_ptr<uint>(dGridParticleIndex));*/
#ifdef PROFILE_SORT
	clock_t start = clock();
#endif

	cub::DoubleBuffer<uint> sortKeys; //keys to sort by 
	cub::DoubleBuffer<uint> sortValues; //also sort corresponding particles by key

	////intermediate memory to initialize geometry indices
	//uint *sortedGridParticleHash;

	////intermediate variable used to deallocate memory used to store the sorted Morton codes
	////if it is not used, we soon run out of memory
	//uint *sortedGridParticleIndex;
	cudaError_t cudaStatus;

	cub::CachingDeviceAllocator  g_allocator(true);
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;



	/*cudaStatus = cudaMalloc((void**)&sortedGridParticleHash, sizeof(uint) * numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void**)&sortedGridParticleIndex, sizeof(uint) * numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;*/

	//presumambly, there is no need to allocate space for the current buffers
	sortKeys.d_buffers[0] = *dGridParticleHash;
	sortValues.d_buffers[0] = *dGridParticleIndex;

	sortKeys.d_buffers[1] = *sortedGridParticleHash;
	sortValues.d_buffers[1] = *sortedGridParticleIndex;

#ifdef PROFILE_SORT
	clock_t end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

	// Allocate temporary storage
#ifdef PROFILE_SORT
	start = clock();
#endif
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// Run sort
	//Note: why do I need to sort the particles themselves?
	//The code I found does nothing of the kind.
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numParticles);
	if (cudaStatus != cudaSuccess)
		goto Error;

#ifdef PROFILE_SORT
	end = clock();
	RadixSortTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_SORT
	start = clock();
#endif

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		goto Error;
	*dGridParticleHash = sortKeys.Current();
	*sortedGridParticleHash = sortKeys.Alternate();

	*dGridParticleIndex = sortValues.Current();
	*sortedGridParticleIndex = sortValues.Alternate();
Error:
	if (d_temp_storage)
		cudaFree(d_temp_storage);
	/*if (sortedGridParticleHash)
		cudaFree(sortedGridParticleHash);
	if (sortedGridParticleIndex)
		cudaFree(sortedGridParticleIndex);*/

#ifdef PROFILE_SORT
	end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_SORT
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last " << iterations << " iterations..." << std::endl;
		std::cout << "Average time spent on memory operations: " << totalMemTime / (float)iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on radix sort: " << RadixSortTime / (float)iterations << " (ms)" << std::endl;
		std::cout << std::endl;
	}
#endif
}

void staticCollide(float4 *dCol,
		float4 *pForces, //total force applied to rigid body
	int *rbIndices, //index of the rigid body each particle belongs to
	float4 *relativePos, //particle's relative position
	float4 *pTorque,  //rigid body angular momentum
	float *r_radii, //radii of all scene particles
	float *newVel,
	float *sortedPos,
	float *sortedVel,
	float *staticSortedPos,
	uint  *gridParticleIndex,
	uint  *cellStart,
	uint  *cellEnd,
	uint   numParticles,
	uint   numCells)
{
#if USE_TEX
	checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

	// thread per particle
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 512, numBlocks, numThreads);

	// execute the kernel
	staticCollideD << < numBlocks, numThreads >> >(
		dCol,
		pForces, //total force applied to rigid body
		rbIndices, //index of the rigid body each particle belongs to
		relativePos, //particle's relative position
		pTorque,  //rigid body angular momentum
		r_radii, //radii of all scene particles
		(float4 *)newVel,
		(float4 *)sortedPos,
		(float4 *)sortedVel,
		(float4 *)staticSortedPos,
		gridParticleIndex,
		cellStart,
		cellEnd,
		numParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed: staticCollideD");

#if USE_TEX
	checkCudaErrors(cudaUnbindTexture(oldPosTex));
	checkCudaErrors(cudaUnbindTexture(oldVelTex));
	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
}

