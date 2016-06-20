

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <time.h>


#if __DEVICE_EMULATION__
#define DEBUG_SYNC __syncthreads();
#else
#define DEBUG_SYNC
#endif

#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)	__mul24(x,y)	
#else
#define int_mult(x,y)	x*y
#endif

#define inf 0x7f800000 

__device__ void warp_reduce_max(volatile float3* smem)
{
	/*if (blockSize >= 1024)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 512].x > smem[threadIdx.x].x ?
	smem[threadIdx.x + 512].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 512].y > smem[threadIdx.x].y ?
	smem[threadIdx.x + 512].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 512].z > smem[threadIdx.x].z ?
	smem[threadIdx.x + 512].z : smem[threadIdx.x].z;
	}
	if (blockSize >= 512)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 256].x > smem[threadIdx.x].x ?
	smem[threadIdx.x + 256].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 256].y > smem[threadIdx.x].y ?
	smem[threadIdx.x + 256].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 256].z > smem[threadIdx.x].z ?
	smem[threadIdx.x + 256].z : smem[threadIdx.x].z;
	}

	if (blockSize >= 256)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 128].x > smem[threadIdx.x].x ?
	smem[threadIdx.x + 128].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 128].y > smem[threadIdx.x].y ?
	smem[threadIdx.x + 128].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 128].z > smem[threadIdx.x].z ?
	smem[threadIdx.x + 128].z : smem[threadIdx.x].z;
	}

	if (blockSize >= 128)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 64].x > smem[threadIdx.x].x ?
	smem[threadIdx.x + 64].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 64].y > smem[threadIdx.x].y ?
	smem[threadIdx.x + 64].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 64].z > smem[threadIdx.x].z ?
	smem[threadIdx.x + 64].z : smem[threadIdx.x].z;
	}
	__syncthreads();*/
	smem[threadIdx.x].x = smem[threadIdx.x + 32].x > smem[threadIdx.x].x ?
		smem[threadIdx.x + 32].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 32].y > smem[threadIdx.x].y ?
		smem[threadIdx.x + 32].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 32].z > smem[threadIdx.x].z ?
		smem[threadIdx.x + 32].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 16].x > smem[threadIdx.x].x ?
		smem[threadIdx.x + 16].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 16].y > smem[threadIdx.x].y ?
		smem[threadIdx.x + 16].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 16].z > smem[threadIdx.x].z ?
		smem[threadIdx.x + 16].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 8].x > smem[threadIdx.x].x ?
		smem[threadIdx.x + 8].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 8].y > smem[threadIdx.x].y ?
		smem[threadIdx.x + 8].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 8].z > smem[threadIdx.x].z ?
		smem[threadIdx.x + 8].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 4].x > smem[threadIdx.x].x ?
		smem[threadIdx.x + 4].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 4].y > smem[threadIdx.x].y ?
		smem[threadIdx.x + 4].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 4].z > smem[threadIdx.x].z ?
		smem[threadIdx.x + 4].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 2].x > smem[threadIdx.x].x ?
		smem[threadIdx.x + 2].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 2].y > smem[threadIdx.x].y ?
		smem[threadIdx.x + 2].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 2].z > smem[threadIdx.x].z ?
		smem[threadIdx.x + 2].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 1].x > smem[threadIdx.x].x ?
		smem[threadIdx.x + 1].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 1].y > smem[threadIdx.x].y ?
		smem[threadIdx.x + 1].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 1].z > smem[threadIdx.x].z ?
		smem[threadIdx.x + 1].z : smem[threadIdx.x].z;



}

__device__ void warp_reduce_min(volatile float3* smem)
{
	/*if (blockSize >= 1024)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 512].x < smem[threadIdx.x].x ?
	smem[threadIdx.x + 512].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 512].y < smem[threadIdx.x].y ?
	smem[threadIdx.x + 512].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 512].z < smem[threadIdx.x].z ?
	smem[threadIdx.x + 512].z : smem[threadIdx.x].z;
	}
	if (blockSize >= 512)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 256].x < smem[threadIdx.x].x ?
	smem[threadIdx.x + 256].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 256].y < smem[threadIdx.x].y ?
	smem[threadIdx.x + 256].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 256].z < smem[threadIdx.x].z ?
	smem[threadIdx.x + 256].z : smem[threadIdx.x].z;
	}

	if (blockSize >= 256)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 128].x < smem[threadIdx.x].x ?
	smem[threadIdx.x + 128].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 128].y < smem[threadIdx.x].y ?
	smem[threadIdx.x + 128].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 128].z < smem[threadIdx.x].z ?
	smem[threadIdx.x + 128].z : smem[threadIdx.x].z;
	}

	if (blockSize >= 128)
	{
	smem[threadIdx.x].x = smem[threadIdx.x + 64].x < smem[threadIdx.x].x ?
	smem[threadIdx.x + 64].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 64].y < smem[threadIdx.x].y ?
	smem[threadIdx.x + 64].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 64].z < smem[threadIdx.x].z ?
	smem[threadIdx.x + 64].z : smem[threadIdx.x].z;
	}
	__syncthreads();*/
	smem[threadIdx.x].x = smem[threadIdx.x + 32].x < smem[threadIdx.x].x ?
		smem[threadIdx.x + 32].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 32].y < smem[threadIdx.x].y ?
		smem[threadIdx.x + 32].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 32].z < smem[threadIdx.x].z ?
		smem[threadIdx.x + 32].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 16].x < smem[threadIdx.x].x ?
		smem[threadIdx.x + 16].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 16].y < smem[threadIdx.x].y ?
		smem[threadIdx.x + 16].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 16].z < smem[threadIdx.x].z ?
		smem[threadIdx.x + 16].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 8].x < smem[threadIdx.x].x ?
		smem[threadIdx.x + 8].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 8].y < smem[threadIdx.x].y ?
		smem[threadIdx.x + 8].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 8].z < smem[threadIdx.x].z ?
		smem[threadIdx.x + 8].z : smem[threadIdx.x].z;


	smem[threadIdx.x].x = smem[threadIdx.x + 4].x < smem[threadIdx.x].x ?
		smem[threadIdx.x + 4].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 4].y < smem[threadIdx.x].y ?
		smem[threadIdx.x + 4].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 4].z < smem[threadIdx.x].z ?
		smem[threadIdx.x + 4].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 2].x < smem[threadIdx.x].x ?
		smem[threadIdx.x + 2].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 2].y < smem[threadIdx.x].y ?
		smem[threadIdx.x + 2].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 2].z < smem[threadIdx.x].z ?
		smem[threadIdx.x + 2].z : smem[threadIdx.x].z;

	smem[threadIdx.x].x = smem[threadIdx.x + 1].x < smem[threadIdx.x].x ?
		smem[threadIdx.x + 1].x : smem[threadIdx.x].x;
	smem[threadIdx.x].y = smem[threadIdx.x + 1].y < smem[threadIdx.x].y ?
		smem[threadIdx.x + 1].y : smem[threadIdx.x].y;
	smem[threadIdx.x].z = smem[threadIdx.x + 1].z < smem[threadIdx.x].z ?
		smem[threadIdx.x + 1].z : smem[threadIdx.x].z;

}

template<int threads>
__global__ void find_min_max_dynamic(float3* in, float3* out, int n, int start_adr, int num_blocks)
{

	__shared__ float3 smem_min[threads];
	__shared__ float3 smem_max[threads];

	int tid = threadIdx.x + start_adr;

	float3 max = make_float3(-inf, -inf, -inf);
	float3 min = make_float3(inf, inf, inf);
	float3 val;


	// tail part
	int mult = 0;
	for (int i = 1; mult + tid < n; i++)
	{
		val = in[tid + mult];

		min.x = val.x < min.x ? val.x : min.x;
		min.y = val.y < min.y ? val.y : min.y;
		min.z = val.z < min.z ? val.z : min.z;

		max.x = val.x > max.x ? val.x : max.x;
		max.y = val.y > max.y ? val.y : max.y;
		max.z = val.z > max.z ? val.z : max.z;

		mult = int_mult(i, threads);
	}

	// previously reduced MIN part
	mult = 0;
	int i;
	for (i = 1; mult + threadIdx.x < num_blocks; i++)
	{
		val = out[threadIdx.x + mult];

		min.x = val.x < min.x ? val.x : min.x;
		min.y = val.y < min.y ? val.y : min.y;
		min.z = val.z < min.z ? val.z : min.z;

		mult = int_mult(i, threads);
	}

	// MAX part
	for (; mult + threadIdx.x < num_blocks * 2; i++)
	{
		val = out[threadIdx.x + mult];

		max.x = val.x > max.x ? val.x : max.x;
		max.y = val.y > max.y ? val.y : max.y;
		max.z = val.z > max.z ? val.z : max.z;

		mult = int_mult(i, threads);
	}


	if (threads == 32)
	{
		smem_min[threadIdx.x + 32] = make_float3(0.0f, 0.0f, 0.0f);
		smem_max[threadIdx.x + 32] = make_float3(0.0f, 0.0f, 0.0f);

	}

	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;

	__syncthreads();

	if (threads >= 1024)
	{
		if (threadIdx.x < 512)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 512].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 512].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 512].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 512].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 512].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 512].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 512].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 512].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 512].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 512].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 512].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 512].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}
	if (threads >= 512)
	{
		if (threadIdx.x < 256)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 256].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 256].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 256].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 256].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 256].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 256].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 256].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 256].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 256].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 256].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 256].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 256].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}

	if (threads >= 256)
	{
		if (threadIdx.x < 128)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 128].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 128].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 128].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 128].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 128].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 128].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 128].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 128].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 128].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 128].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 128].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 128].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}

	if (threads >= 128)
	{
		if (threadIdx.x < 64)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 64].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 64].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 64].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 64].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 64].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 64].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 64].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 64].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 64].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 64].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 64].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 64].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if (threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x];
	}


}

template<int blockSize, int threads>
__global__ void find_min_max(float3* in, float3* out)
{
	__shared__ float3 smem_min[threads];
	__shared__ float3 smem_max[threads];

	int tid = threadIdx.x + blockIdx.x * blockSize;

	float3 max = make_float3(-inf, -inf, -inf);
	float3 min = make_float3(inf, inf, inf);
	float3 val;

	const int iters = blockSize / threads;

#pragma unroll  
	for (int i = 0; i < iters; i++)
	{

		val = in[tid + i*threads];

		min.x = val.x < min.x ? val.x : min.x;
		min.y = val.y < min.y ? val.y : min.y;
		min.z = val.z < min.z ? val.z : min.z;

		max.x = val.x > max.x ? val.x : max.x;
		max.y = val.y > max.y ? val.y : max.y;
		max.z = val.z > max.z ? val.z : max.z;

	}


	if (threads == 32)
	{
		smem_min[threadIdx.x + 32] = make_float3(0.0f, 0.0f, 0.0f);
		smem_max[threadIdx.x + 32] = make_float3(0.0f, 0.0f, 0.0f);

	}

	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;
	__syncthreads();

	if (threads >= 1024)
	{
		if (threadIdx.x < 512)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 512].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 512].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 512].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 512].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 512].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 512].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 512].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 512].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 512].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 512].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 512].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 512].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}
	if (threads >= 512)
	{
		if (threadIdx.x < 256)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 256].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 256].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 256].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 256].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 256].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 256].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 256].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 256].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 256].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 256].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 256].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 256].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}

	if (threads >= 256)
	{
		if (threadIdx.x < 128)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 128].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 128].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 128].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 128].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 128].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 128].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 128].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 128].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 128].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 128].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 128].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 128].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}

	if (threads >= 128)
	{
		if (threadIdx.x < 64)
		{
			smem_min[threadIdx.x].x = smem_min[threadIdx.x + 64].x < smem_min[threadIdx.x].x ?
				smem_min[threadIdx.x + 64].x : smem_min[threadIdx.x].x;
			smem_min[threadIdx.x].y = smem_min[threadIdx.x + 64].y < smem_min[threadIdx.x].y ?
				smem_min[threadIdx.x + 64].y : smem_min[threadIdx.x].y;
			smem_min[threadIdx.x].z = smem_min[threadIdx.x + 64].z < smem_min[threadIdx.x].z ?
				smem_min[threadIdx.x + 64].z : smem_min[threadIdx.x].z;

			smem_max[threadIdx.x].x = smem_max[threadIdx.x + 64].x > smem_max[threadIdx.x].x ?
				smem_max[threadIdx.x + 64].x : smem_max[threadIdx.x].x;
			smem_max[threadIdx.x].y = smem_max[threadIdx.x + 64].y > smem_max[threadIdx.x].y ?
				smem_max[threadIdx.x + 64].y : smem_max[threadIdx.x].y;
			smem_max[threadIdx.x].z = smem_max[threadIdx.x + 64].z > smem_max[threadIdx.x].z ?
				smem_max[threadIdx.x + 64].z : smem_max[threadIdx.x].z;
		}
		__syncthreads();
	}
	__syncthreads();

	if (threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if (threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x];
	}

}

int findBlockSize(const int numberOfPrimitives)
{
	const float pretty_big_number = 24.0f*1024.0f*1024.0f;

	float ratio = float(numberOfPrimitives) / pretty_big_number;


	if (ratio > 0.8f)
		return 5;
	else if (ratio > 0.6f)
		return 4;
	else if (ratio > 0.4f)
		return 3;
	else if (ratio > 0.2f)
		return 2;
	
	return 1;

}
/*
This function is still buggy. For some reason using 1024 threads increases overhead and also results in mistakes.
Number of threads should be in range [64-256].
I will not use template metaprogramming because it will severely increase compilation time. I will predetermine number of threads.
This will be one of the few kernel functions that do not have a numberOfThreads argument.
*/
cudaError_t findAABB(float3 *positions, float3 *minPos, float3 *maxPos, int numberOfPrimitives)
{
	cudaError_t cudaStatus;
	int numberOfThreads = 128;
	int numberOfBlocks = (numberOfPrimitives + numberOfThreads - 1) / numberOfThreads;
	
	find_min_max << < numberOfBlocks, numberOfThreads >> >(d_in, d_out);
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)return cudaStatus;
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)return cudaStatus;

	find_min_max_dynamic << < 1, numberOfThreads >> >(positions, d_out, num_els, start_adr, num_blocks);
	
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)return cudaStatus;
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)return cudaStatus;

	return cudaStatus;
}