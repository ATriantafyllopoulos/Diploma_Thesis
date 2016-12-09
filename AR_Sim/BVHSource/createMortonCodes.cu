#define CUB_STDERR
//define _CubLog to avoid encountering error: "undefined reference"
#if !defined(_CubLog)
#if (CUB_PTX_ARCH == 0)
#define _CubLog(format, ...) printf(format,__VA_ARGS__);
#elif (CUB_PTX_ARCH >= 200)
#define _CubLog(format, ...) printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, __VA_ARGS__);
#endif
#endif
#define inf 0x7f800000
//cub headers
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

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
#include <time.h>
cudaError_t findAABB(float4 *positions, float3 *d_out, int numberOfPrimitives);
__global__ void generateMortonCodes(float4 *positions, unsigned int *mortonCodes, int *indices, const int numberOfPrimitives,
	float4 *minPos, float4 *maxPos);

__global__ void generateMortonCodes(float4 *positions, unsigned int *mortonCodes, int *indices, const int numberOfPrimitives,
	float4 minPos, float4 maxPos);

struct CustomMin4
{
	template <typename T>
	__device__ __forceinline__
	T operator()(const T &a, const T &b) const {
		T res;
		res.x =  (b.x < a.x) ? b.x : a.x;
		res.y =  (b.y < a.y) ? b.y : a.y;
		res.z =  (b.z < a.z) ? b.z : a.z;
		res.w =  (b.w < a.w) ? b.w : a.w;
		return res;
	}
};

struct CustomMax4
{
	template <typename T>
	__device__ __forceinline__
	T operator()(const T &a, const T &b) const {
		T res;
		res.x =  (b.x > a.x) ? b.x : a.x;
		res.y =  (b.y > a.y) ? b.y : a.y;
		res.z =  (b.z > a.z) ? b.z : a.z;
		res.w =  (b.w > a.w) ? b.w : a.w;
		return res;
	}
};

cudaError_t findAABBCub(
		float4 *d_in,
		float4 &cpuMin,
		float4 &cpuMax,
		float4 *gpuMin,
		float4 *gpuMax,
		int numberOfPrimitives)
{
	CustomMin4    min_op;
	float4 init = make_float4(inf, inf, inf, inf);
	float4 *d_out;
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float4) * numberOfPrimitives));
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, numberOfPrimitives, min_op, init));
	// Allocate temporary storage
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	// Run reduction
	checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, numberOfPrimitives, min_op, init));
	// d_out <-- [0]
	checkCudaErrors(cudaMemcpy(&cpuMin, &d_out[0], sizeof(float4), cudaMemcpyDeviceToHost));
	if (gpuMin)
		checkCudaErrors(cudaMemcpy(gpuMin, &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));

	CustomMax4    max_op;
	init = make_float4(-inf, -inf, -inf, -inf);
	// Determine temporary device storage requirements

	checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, numberOfPrimitives, max_op, init));
	// Allocate temporary storage
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	// Run reduction
	checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, numberOfPrimitives, max_op, init));
	// d_out <-- [0]
	checkCudaErrors(cudaMemcpy(&cpuMax, &d_out[0], sizeof(float4), cudaMemcpyDeviceToHost));
	if (gpuMax)
		checkCudaErrors(cudaMemcpy(gpuMax, &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));


	checkCudaErrors(cudaFree(d_out));
	return cudaSuccess;
}

cudaError_t createMortonCodes(float4 *positions,
	unsigned int **mortonCodes,
	int **indices,
	unsigned int **sortedMortonCodes,
	int **sortedIndices,
	int numberOfPrimitives,
	int numberOfThreads)
{
#define PROFILE_MORTON
#ifdef PROFILE_MORTON
	static float totalAABBtime = 0;
	static float totalRadixSortTime = 0;
	static float totalGenMortonCodesTime = 0;
	static float totalMemTime = 0;
	static int iterations = 0;
#endif
	//float3 *temp;
	//find bounding box
	//use it to normalize positions and create Morton codes
	//Morton codes require vertex input to be in the unit cube
	//checkCudaErrors(cudaMalloc((void**)&temp, sizeof(float) * 3 * numberOfPrimitives));
	float4 cpuMin, cpuMax, *gpuMin, *gpuMax;

#ifdef PROFILE_MORTON
	clock_t start = clock();
#endif
	checkCudaErrors(cudaMalloc((void**)&gpuMin, sizeof(float4)));
	checkCudaErrors(cudaMalloc((void**)&gpuMax, sizeof(float4)));
#ifdef PROFILE_MORTON
	clock_t end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_MORTON
	start = clock();
#endif
	checkCudaErrors(findAABBCub(positions, cpuMin, cpuMax, gpuMin, gpuMax, numberOfPrimitives));
	//checkCudaErrors(findAABB(positions, temp, numberOfPrimitives));
#ifdef PROFILE_MORTON
	end = clock();
	totalAABBtime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

	dim3 blockDim(numberOfThreads, 1);
	dim3 gridDim((numberOfPrimitives + numberOfThreads - 1) / numberOfThreads, 1);
	if (gridDim.x < 1) gridDim.x = 1;
#ifdef PROFILE_MORTON
	start = clock();
#endif
	generateMortonCodes << <gridDim, blockDim >> >(positions, *mortonCodes, *indices, numberOfPrimitives,
		gpuMin, gpuMax);
#ifdef PROFILE_MORTON
	end = clock();
	totalGenMortonCodesTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaPeekAtLastError());
	//checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaFree(temp));
#ifdef PROFILE_MORTON
	start = clock();
#endif
	checkCudaErrors(cudaFree(gpuMin));
	checkCudaErrors(cudaFree(gpuMax));
#ifdef PROFILE_MORTON
	end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_MORTON
	start = clock();
#endif

	cub::CachingDeviceAllocator  g_allocator(true);
	cub::DoubleBuffer<unsigned int> sortKeys; //keys to sort by - Morton codes
	cub::DoubleBuffer<int> sortValues; //also sort corresponding particles by key
	//presumambly, there is no need to allocate space for the current buffers
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;

	// Allocate temporary storage
	sortKeys.d_buffers[0] = *mortonCodes;
	sortValues.d_buffers[0] = *indices;
	sortKeys.d_buffers[1] = *sortedMortonCodes;
	sortValues.d_buffers[1] = *sortedIndices;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numberOfPrimitives));


	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));


	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numberOfPrimitives));

	
#ifdef PROFILE_MORTON
	end = clock();
	totalRadixSortTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif
	//checkCudaErrors(cudaDeviceSynchronize());

#ifdef PROFILE_MORTON
	start = clock();
#endif
	*sortedIndices = sortValues.Current();
	*sortedMortonCodes = sortKeys.Current();

	*indices = sortValues.Alternate();
	*mortonCodes = sortKeys.Alternate();
#ifdef PROFILE_MORTON
	end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif



#ifdef PROFILE_MORTON
	start = clock();
#endif
	if (d_temp_storage)
		cudaFree(d_temp_storage);
#ifdef PROFILE_MORTON
	end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_MORTON
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last " << iterations << " iterations..." << std::endl;
		std::cout << "Average time spent on memory operations: " << totalMemTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on finding AABB: " << totalAABBtime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on creating Morton codes: " << totalGenMortonCodesTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on binary radix sort: " << totalRadixSortTime / iterations << " (ms)" << std::endl;
		std::cout << std::endl;
	}
#endif

	//checkCudaErrors(cudaPeekAtLastError());
	//checkCudaErrors(cudaDeviceSynchronize());
	return cudaSuccess;
}



void createMortonCodesPreallocated(
	float4 *positions,
	unsigned int **mortonCodes,
	int **indices,
	unsigned int **sortedMortonCodes,
	int **sortedIndices,
	float4 minPos,
	float4 maxPos,
	int elements,
	int numberOfThreads)
{
//#define PROFILE_MORTON
#ifdef PROFILE_MORTON
	static float totalRadixSortTime = 0;
	static float totalGenMortonCodesTime = 0;
	static float totalMemTime = 0;
	static int iterations = 0;
#endif

	dim3 blockDim(numberOfThreads, 1);
	dim3 gridDim((elements + numberOfThreads - 1) / numberOfThreads, 1);
	if (gridDim.x < 1) gridDim.x = 1;
#ifdef PROFILE_MORTON
	clock_t start = clock();
#endif
	generateMortonCodes << <gridDim, blockDim >> >(positions, *mortonCodes, *indices, elements,
		minPos, maxPos);
#ifdef PROFILE_MORTON
	clock_t end = clock();
	totalGenMortonCodesTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef PROFILE_MORTON
	start = clock();
#endif

	cub::CachingDeviceAllocator  g_allocator(true);
	cub::DoubleBuffer<unsigned int> sortKeys; //keys to sort by - Morton codes
	cub::DoubleBuffer<int> sortValues; //also sort corresponding particles by key
	//presumambly, there is no need to allocate space for the current buffers
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;

	// Allocate temporary storage
	sortKeys.d_buffers[0] = *mortonCodes;
	sortValues.d_buffers[0] = *indices;
	sortKeys.d_buffers[1] = *sortedMortonCodes;
	sortValues.d_buffers[1] = *sortedIndices;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, elements));


	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));


	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, elements));


#ifdef PROFILE_MORTON
	end = clock();
	totalRadixSortTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif
	//checkCudaErrors(cudaDeviceSynchronize());

#ifdef PROFILE_MORTON
	start = clock();
#endif
	*sortedIndices = sortValues.Current();
	*sortedMortonCodes = sortKeys.Current();

	*indices = sortValues.Alternate();
	*mortonCodes = sortKeys.Alternate();

	if (d_temp_storage)
		cudaFree(d_temp_storage);
#ifdef PROFILE_MORTON
	end = clock();
	totalMemTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
#endif

#ifdef PROFILE_MORTON
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last " << iterations << " iterations..." << std::endl;
		std::cout << "Average time spent on memory operations: " << totalMemTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on creating Morton codes: " << totalGenMortonCodesTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on binary radix sort: " << totalRadixSortTime / iterations << " (ms)" << std::endl;
		std::cout << std::endl;
	}
#endif
}
