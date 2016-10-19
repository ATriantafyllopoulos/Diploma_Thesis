#include "BVHAuxiliary.cuh"

#define inf 0x7f800000 
#define CODE_OFFSET (1<<21)
#define CODE_LENGTH (21)
//#define threads 128
__device__ void warp_reduce_max(volatile float3* smem)
{
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

template <int threads>
__global__ void find_min_max_dynamic(float4* in, float3* out, int n, int start_adr, int num_blocks)
{

    __shared__ float3 smem_min[threads];
    __shared__ float3 smem_max[threads];
//    extern __shared__ float array[];
//    float3 *smem_min = (float3 *)array;
//    float3 *smem_max = (float3 *)&smem_min[threads];
	int tid = threadIdx.x + start_adr;
    if (tid > n) return;

	float3 max = make_float3(-inf, -inf, -inf);
	float3 min = make_float3(inf, inf, inf);
	float3 val;


	// tail part
	int mult = 0;
	for (int i = 1; mult + tid < n; i++)
	{
		val = make_float3(in[tid + i*threads]);

		min.x = val.x < min.x ? val.x : min.x;
		min.y = val.y < min.y ? val.y : min.y;
		min.z = val.z < min.z ? val.z : min.z;

		max.x = val.x > max.x ? val.x : max.x;
		max.y = val.y > max.y ? val.y : max.y;
		max.z = val.z > max.z ? val.z : max.z;

		mult = i * threads;
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

		mult = i * threads;
	}

	// MAX part
    for (; mult + threadIdx.x < num_blocks * 2; i++)
	{
		val = out[threadIdx.x + mult];

		max.x = val.x > max.x ? val.x : max.x;
		max.y = val.y > max.y ? val.y : max.y;
		max.z = val.z > max.z ? val.z : max.z;

		mult = i * threads;
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

template <int blockSize, int threads>
__global__ void find_min_max(float4* in, float3* out)
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

		val = make_float3(in[tid + i*threads]);

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

const int blockSize1 = 4096 / 2;
void findBlockSize(int* whichSize, int* num_el)
{

	const float pretty_big_number = 24.0f*1024.0f*1024.0f;

	float ratio = float((*num_el)) / pretty_big_number;


	if (ratio > 0.8f)
		(*whichSize) = 5;
	else if (ratio > 0.6f)
		(*whichSize) = 4;
	else if (ratio > 0.4f)
		(*whichSize) = 3;
	else if (ratio > 0.2f)
		(*whichSize) = 2;
	else
		(*whichSize) = 1;


}

template <int threads>
void choose_numberOfThreads(float4* d_in, float3* d_out, int numberOfPrimitives)
{
	int whichSize = -1;
	//num_els *= 3;
	findBlockSize(&whichSize, &numberOfPrimitives);
	//num_els /= 3;
	//whichSize = 5;

	int block_size = powf(2, whichSize - 1)*blockSize1;
	int num_blocks = numberOfPrimitives / block_size;
	if (num_blocks == 0)num_blocks = 2;
	int tail = numberOfPrimitives - num_blocks*block_size;
	int start_adr = numberOfPrimitives - tail;

	std::cout << "Finding BB of " << numberOfPrimitives << " particles." << std::endl;
	std::cout << "Size to use: " << whichSize << std::endl;
	std::cout << "Address offset: " << start_adr  << std::endl;
	std::cout << "Tail size: " << tail  << std::endl;
	std::cout << "Number of blocks: " << num_blocks  << std::endl;
	std::cout << "Block size: " << block_size  << std::endl;
	if (whichSize == 1)
		find_min_max<blockSize1, threads> << < num_blocks, threads >> >(d_in, d_out);
	else if (whichSize == 2)
		find_min_max<blockSize1 * 2, threads> << < num_blocks, threads >> >(d_in, d_out);
	else if (whichSize == 3)
		find_min_max<blockSize1 * 4, threads> << < num_blocks, threads >> >(d_in, d_out);
	else if (whichSize == 4)
		find_min_max<blockSize1 * 8, threads> << < num_blocks, threads >> >(d_in, d_out);
	else
		find_min_max<blockSize1 * 16, threads> << < num_blocks, threads >> >(d_in, d_out);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    find_min_max_dynamic<threads> << < 1, threads>> >(d_in, d_out, numberOfPrimitives, start_adr, num_blocks);

    float3 minPos, maxPos;
    cudaMemcpy(&minPos, &d_out[0], sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxPos, &d_out[1], sizeof(float3), cudaMemcpyDeviceToHost);

    std::cout << "Bounding box: " << std::endl;
    std::cout << "Min: (" << minPos.x << ", " << minPos.y << ", " << minPos.z << ")" << std::endl;
    std::cout << "Max: (" << maxPos.x << ", " << maxPos.y << ", " << maxPos.z << ")" << std::endl;

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

cudaError_t findAABB(float4 *positions, float3 *d_out, int numberOfPrimitives)
{
	cudaError_t cudaStatus;
	fprintf(stderr, "Old findAABB function is deprecated! Please remove from code.");
    choose_numberOfThreads<128>(positions, d_out, numberOfPrimitives);
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)return cudaStatus;
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)return cudaStatus;

	return cudaStatus;
}


template <typename BoundingVolume>
__global__ void kernelConstructRadixTree(int len,
	TreeNode<BoundingVolume> *radixTreeNodes, TreeNode<BoundingVolume> *radixTreeLeaves, unsigned int *sortedMortonCodes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= len) return;

	// Run radix tree construction algorithm
	// Determine direction of the range (+1 or -1)
	int dPrev = longestCommonPrefix(i, i - 1, len + 1, sortedMortonCodes);
	int dNext = longestCommonPrefix(i, i + 1, len + 1, sortedMortonCodes);
	int d = dNext - dPrev > 0 ? 1 : -1;

	// Compute upper bound for the length of the range
	int sigMin = longestCommonPrefix(i, i - d, len + 1, sortedMortonCodes);
	int lmax = 2;

	while (longestCommonPrefix(i, i + lmax * d, len + 1, sortedMortonCodes) > sigMin) {
		lmax *= 2;
	}

	// Find the other end using binary search
	int l = 0;
	int t = lmax / 2;
	for (t = lmax / 2; t >= 1; t /= 2)
	{
		if (longestCommonPrefix(i, i + (l + t) * d, len + 1, sortedMortonCodes) > sigMin)
			l += t;
		if (t == 1)	break;
	}
	
	int j = i + l * d;

	// Find the split position using binary search
	int sigNode = longestCommonPrefix(i, j, len + 1, sortedMortonCodes);

	int s = 0;
	double div;
	double t2;
	for (div = 2; t >= 1; div *= 2)
	{
		t2 = __int2double_rn(l) / div;
		t = __double2int_ru(t2);
		int temp = longestCommonPrefix(i, i + (s + t) * d, len + 1, sortedMortonCodes);
		if (temp > sigNode) 
		{
			s = s + t;
		}
		if (t == 1)	break;
	}

	int gamma = i + s * d + intMin(d, 0);

	// Output child pointers
	TreeNode<BoundingVolume> *current = radixTreeNodes + i;
	current->leaf = false;
	//current->gamma = gamma;
	//(gamma < i && gamma >j) || (gamma > i && gamma <j)
	if (intMin(i, j) == gamma) {
		current->left = radixTreeLeaves + gamma;
		(radixTreeLeaves + gamma)->parent = current;
		//(radixTreeLeaves + gamma)->parentIndex = i;
		//(radixTreeLeaves + gamma)->edited = 23;
	}
	else {
		current->left = radixTreeNodes + gamma;
		(radixTreeNodes + gamma)->parent = current;
		//(radixTreeNodes + gamma)->parentIndex = i;
		//(radixTreeNodes + gamma)->edited = 23;
	}

	if (intMax(i, j) == gamma + 1) {
		current->right = radixTreeLeaves + gamma + 1;
		(radixTreeLeaves + gamma + 1)->parent = current;
		//(radixTreeLeaves + gamma + 1)->parentIndex = i;
		//(radixTreeLeaves + gamma + 1)->edited = 23;
	}
	else {
		current->right = radixTreeNodes + gamma + 1;
		(radixTreeNodes + gamma + 1)->parent = current;
		//(radixTreeNodes + gamma + 1)->parentIndex = i;
		//(radixTreeNodes + gamma + 1)->edited = 23;
	}
	//if (current == radixTreeNodes)
		//return;
	current->min = intMin(i, j);
	current->max = intMax(i, j);
}

/**
* BVH Construction kernel
* Algorithm described in karras2012 paper (bottom-up approach).
*/
template <typename BoundingVolume>
__global__ void kernelConstructLeafNodes(int len, TreeNode<BoundingVolume> *treeLeaves,
	int *sorted_geometry_indices, float4 *positions,float particleRadius)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//random comment
	if (index >= len) return;

	TreeNode<BoundingVolume> *leaf = treeLeaves + index;
	leaf->leaf = true;
	// Handle leaf first
	int geometry_index = sorted_geometry_indices[index];
	//float4 cm = FETCH(positions, geometry_index);
	leaf->cm = FETCH(positions, geometry_index);
	leaf->radius = particleRadius;
	leaf->index = geometry_index;
	leaf->min = index;
	leaf->max = index;
	initBound(&(leaf->boundingVolume), leaf->cm, particleRadius);
}

/**
* BVH Construction kernel
* Algorithm described in karras2012 paper (bottom-up approach).
*/
template <typename BoundingVolume>
__global__ void kernelConstructInternalNodes(int len, TreeNode<BoundingVolume> *treeNodes, TreeNode<BoundingVolume> *treeLeaves, int *nodeCounter)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index >= len) return;

	TreeNode<BoundingVolume> *leaf = treeLeaves + index;
	TreeNode<BoundingVolume> *current = leaf->parent;
	int currentIndex = current - treeNodes;
	int res = atomicAdd(nodeCounter + currentIndex, 1);

	// Go up and handle internal nodes
	while (1) {
		if (res == 0) {
			break;
		}

		mergeBounds(&(current->boundingVolume), current->left->boundingVolume, current->right->boundingVolume);
		current->cm = (current->left->cm + current->right->cm) / 2;
		current->radius = MAX(length(current->cm - current->left->cm) + current->left->radius, length(current->cm - current->right->cm) + current->right->radius);
		// If current is root, return
		if (current == treeNodes) {
			return;
		}
		current = current->parent;
		currentIndex = current - treeNodes;
		res = atomicAdd(nodeCounter + currentIndex, 1);
	}
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z)
{
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__global__ void generateMortonCodes(float4 *positions, unsigned int *mortonCodes, int *indices, int numberOfPrimitives,
	float4 *minPos, float4 *maxPos)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfPrimitives)
		return;

	float min_x = minPos->x;
	float min_y = minPos->y;
	float min_z = minPos->z;
	float max_x = maxPos->x;
	float max_y = maxPos->y;
	float max_z = maxPos->z;

	float3 p = make_float3(positions[index].x, positions[index].y, positions[index].z);
	
	float x = (p.x - min_x) / (max_x - min_x);
	float y = (p.y - min_y) / (max_y - min_y);
	float z = (p.z - min_z) / (max_z - min_z);
	mortonCodes[index] = morton3D(x, y, z);//*/
	indices[index] = index;
}

template <typename BoundingVolume>
__global__
void collideBVH(float4 *color,
float4 *vel,
TreeNode<BoundingVolume> *treeNodes,
TreeNode<BoundingVolume> *treeLeaves,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	
	float3 force = make_float3(0.0f);
	TreeNode<BoundingVolume> *queryLeaf = treeLeaves + index;

	float3 queryVel = make_float3(FETCH(vel, queryLeaf->index));
	TreeNode<BoundingVolume>* stack[64]; //AT: Is 64 the correct size to use?
	TreeNode<BoundingVolume>** stackPtr = stack;
	//when stack is empty thread will return
	*stackPtr++ = NULL; // push NULL at beginning
	int numCollisions = 0;
	// Traverse nodes starting from the root.
	float4 queryPos = queryLeaf->cm;
	float queryRad = queryLeaf->radius;
	TreeNode<BoundingVolume>* node = treeNodes;
	do
	{
		// Check each child node for overlap.
		TreeNode<BoundingVolume>* childL = node->left;
		TreeNode<BoundingVolume>* childR = node->right;
		/*float distL = length(queryPos - childL->cm);
		float distR = length(queryPos - childR->cm);
		bool overlapL = distL <= queryRad + childL->radius;
		bool overlapR = distR <= queryRad + childR->radius;*/
		bool overlapL = checkOverlap(queryLeaf->boundingVolume, childL->boundingVolume);
		bool overlapR = checkOverlap(queryLeaf->boundingVolume, childR->boundingVolume);
		/*if ((node->min > index) || queryLeaf->index == childL->index)
			overlapL = false;
		if ((node->max < index) || queryLeaf->index == childR->index)
			overlapR = false;*/
		/*if ((!childL->leaf && childL->max <= queryLeaf->index) || (childL->index == queryLeaf->index && childL->leaf))
			overlapL = false;
		if ((!childR->leaf && childR->max <= queryLeaf->index) || (childR->index == queryLeaf->index && childR->leaf))
			overlapR = false;*/
		if (childL->index == queryLeaf->index && childL->leaf)
			overlapL = false;
		if (childR->index == queryLeaf->index && childR->leaf)
			overlapR = false;

		// Query overlaps a leaf node => report collision.
		if (overlapL && childL->leaf)
		{
			numCollisions++;
			color[queryLeaf->index] = make_float4(1, 0, 0, 0);
			//color[childL->index] = make_float4(1, 0, 0, 0);
			force += collideSpheresBVH(queryPos, childL->cm,
				queryVel, make_float3(FETCH(vel, childL->index)),
				queryRad, childL->radius,
				params);
			//vel[queryLeaf->index] += make_float4(force, 0);
			//vel[childL->index] += make_float4(force, 0);
		}
		

		if (overlapR && childR->leaf)
		{
			numCollisions++;
			color[queryLeaf->index] = make_float4(1, 0, 0, 0);
			//color[childR->index] = make_float4(1, 0, 0, 0);
			force += collideSpheresBVH(queryPos, childR->cm,
				queryVel, make_float3(FETCH(vel, childR->index)),
				queryRad, childR->radius,
				params);
			//vel[queryLeaf->index] += make_float4(force, 0);
			//vel[childR->index] += make_float4(force, 0);
		}

		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && !childL->leaf);
		bool traverseR = (overlapR && !childR->leaf);

		if (!traverseL && !traverseR)
			node = *--stackPtr; // pop
		else
		{
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR)
				*stackPtr++ = childR; // push
		}
		/*if (overlapL || overlapR)
		{
			traverseL = true;
			traverseR = true;
		}*/
	} while (node != NULL);
	if (!numCollisions)
		color[queryLeaf->index] = make_float4(0, 0, 1, 0);
	// collide with cursor sphere
	float4 colPos = make_float4(params.colliderPos.x, params.colliderPos.y, params.colliderPos.z, 1);
	if (length(queryPos - colPos) <= queryRad + params.colliderRadius)
		force += collideSpheresBVH(queryPos,
		colPos,
		queryVel,
		make_float3(0.0f, 0.0f, 0.0f),
		queryRad,
		params.colliderRadius,
		params);
	vel[queryLeaf->index] = make_float4(queryVel + force, 0);
	// write new velocity back to original unsorted location
	//uint originalIndex = queryLeaf->index;
	//vel[originalIndex] = make_float4(queryVel + force, 0.0f);
}

template <typename BoundingVolume>
__global__
void staticCollideBVH(float4 *positions,
float4 *vel,
TreeNode<BoundingVolume> *treeNodes,
TreeNode<BoundingVolume> *treeLeaves,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	float3 force = make_float3(0.0f);
	//rand
	TreeNode<BoundingVolume>* stack[64]; //AT: Is 64 the correct size to use?
	TreeNode<BoundingVolume>** stackPtr = stack;
	//when stack is empty thread will return
	*stackPtr++ = NULL; // push NULL at beginning
	// Traverse nodes starting from the root.
	float3 queryVel = make_float3(vel[index]);
	float4 queryPos = positions[index];
	float queryRad = params.particleRadius;
	TreeNode<BoundingVolume>* node = treeNodes;
	do
	{
		// Check each child node for overlap.
		TreeNode<BoundingVolume>* childL = node->left;
		TreeNode<BoundingVolume>* childR = node->right;
		bool overlapL = checkOverlap(queryPos, childL->boundingVolume, queryRad);
		bool overlapR = checkOverlap(queryPos, childR->boundingVolume, queryRad);
		/*float distL = length(queryPos - childL->cm);
		float distR = length(queryPos - childR->cm);
		bool overlapL = distL <= queryRad + childL->radius;
		bool overlapR = distR <= queryRad + childR->radius;*/

		/*if (childL->index == queryLeaf->index && childL->leaf)
			overlapL = false;
		if (childR->index == queryLeaf->index && childR->leaf)
			overlapR = false;*/

		// Query overlaps a leaf node => report collision.
		if (overlapL && childL->leaf)
		{
			force += collideSpheresBVH(queryPos, childL->cm,
				queryVel, make_float3(FETCH(vel, childL->index)),
				queryRad, childL->radius,
				params);
		}


		if (overlapR && childR->leaf)
		{
			force += collideSpheresBVH(queryPos, childR->cm,
				queryVel, make_float3(FETCH(vel, childR->index)),
				queryRad, childR->radius,
				params);
		}

		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && !childL->leaf);
		bool traverseR = (overlapR && !childR->leaf);

		if (!traverseL && !traverseR)
			node = *--stackPtr; // pop
		else
		{
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR)
				*stackPtr++ = childR; // push
		}
		/*if (overlapL || overlapR)
		{
		traverseL = true;
		traverseR = true;
		}*/
	} while (node != NULL);

	vel[index] = make_float4(queryVel + force, 0);
	// write new velocity back to original unsorted location
	//uint originalIndex = queryLeaf->index;
	//vel[originalIndex] = make_float4(queryVel + force, 0.0f);
}

template __global__
void staticCollideBVH(float4 *positions,
float4 *vel,
TreeNode<AABB> *treeNodes,
TreeNode<AABB> *treeLeaves,
uint    numParticles,
SimParams params);

template __global__ void kernelConstructLeafNodes(int len, TreeNode<AABB> *treeLeaves,
	int *sorted_geometry_indices, float4 *positions, float particleRadius);

template __global__ 
void kernelConstructInternalNodes(int len, TreeNode<AABB> *treeNodes, TreeNode<AABB> *treeLeaves, int *nodeCounter);

template __global__ void kernelConstructRadixTree(int len,
	TreeNode<AABB> *radixTreeNodes,
	TreeNode<AABB> *radixTreeLeaves,
	unsigned int *sortedMortoncodes);

template __global__
void collideBVH(float4 *color,
float4 *vel,
TreeNode<AABB> *treeNodes,
TreeNode<AABB> *treeLeaves,
uint    numParticles,
SimParams params);
