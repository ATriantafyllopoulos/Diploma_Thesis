#include "auxiliaryKernels.cuh"

//Expands a 10-bit integer into 30 bits
//by inserting 2 zeros after each bit.
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
	x = MIN(MAX(x * 1024.0f, 0.0f), 1023.0f);
	y = MIN(MAX(y * 1024.0f, 0.0f), 1023.0f);
	z = MIN(MAX(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__global__ void generateMortonCodes(float3 *positions, unsigned int *mortonCodes, int numberOfPrimitives)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfPrimitives)
		return;

	mortonCodes[index] = morton3D(positions[index].x, positions[index].y, positions[index].z);
}