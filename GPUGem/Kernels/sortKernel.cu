#include "auxiliaryKernels.cuh"

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

cudaError_t sortMortonCodes(Primitive* leafNodes,
	unsigned int *mortonCodes,
	Primitive **sortedLeafNodes,
	unsigned int **sortedMortonCodes,
	const int &numberOfPrimitives)
{
	cub::DoubleBuffer<unsigned int> sortKeys; //keys to sort by - Morton codes
	cub::DoubleBuffer<Primitive> sortValues; //also sort corresponding particles by key
	//presumambly, there is no need to allocate space for the current buffers
	sortKeys.d_buffers[0] = mortonCodes;
	sortValues.d_buffers[0] = leafNodes;

	cub::CachingDeviceAllocator  g_allocator(true);
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;

	cudaError_t cudaStatus;
	cudaStatus = g_allocator.DeviceAllocate((void**)&sortKeys.d_buffers[1], sizeof(unsigned int) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = g_allocator.DeviceAllocate((void**)&sortValues.d_buffers[1], sizeof(Primitive) * numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// Allocate temporary storage
	
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// Run sort
	//Note: why do I need to sort the particles themselves?
	//The code I found does nothing of the kind.
	cudaStatus = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys, sortValues, numberOfPrimitives);
	if (cudaStatus != cudaSuccess)
		goto Error;
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		goto Error;
	*sortedLeafNodes = sortValues.Current();
	*sortedMortonCodes = sortKeys.Current();
Error:
	if (d_temp_storage)
		cudaFree(d_temp_storage);
	if (sortKeys.d_buffers[1])
		g_allocator.DeviceFree(sortKeys.d_buffers[1]);
	if (sortValues.d_buffers[1])
		g_allocator.DeviceFree(sortValues.d_buffers[1]);
	
	return cudaStatus;
}