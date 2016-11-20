#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Platform.h"
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "BVHAuxiliary.cuh"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
//#define M_PI 3.14159265359

__global__ void RGBDloader(	unsigned short *image,
	float *VAOdata,
	float *staticPos,
	glm::mat4 cameraTransformation,
	float *staticRadii,
	float particleRadius,
	int imageWidth,
	int imageHeight)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int index = y*imageWidth + x;
	if (index > (imageWidth - 1) * (imageHeight - 1))
		return;
	int currentPos = 5 * index;

	//kinect sensor parameters (currently hardcoded)
	//found somewhere on stack overflow - work well for now
	//should be calibrated properly for specific sensor
	//	float focalX = 579.83;
	//	float focalY = 586.73;
	//	float cx = 321.55;
	//	float cy = 235.01;
	float focalX = 528;
	float focalY = 528;
	float cx = 319.5;
	float cy = 239.5;

	
	float posZ = image[y*imageWidth + x];// / 1000.f;
	posZ /= 5000.f;
	//posZ /= 5000.f;
	staticRadii[index] = particleRadius;// * posZ;

	if (posZ < 0.01)
		posZ = -1;
//	else
//		posZ += 3; //add custom offset

	float posX = posZ * (x - cx) / focalX;
	float posY = posZ * (y - cy) / focalY;

	cameraTransformation = transpose((cameraTransformation));
	//pcameraTransformation[3][0] = -cameraTransformation[3][0];
	//cameraTransformation[3][1] = -cameraTransformation[3][1];
	//cameraTransformation[3][2] = -cameraTransformation[3][2];
	glm::vec4 position(posX, posY, posZ, 0.f);
	glm::vec4 transformedPosition = cameraTransformation * position;

	// hardcoded accelerometer data
	// -0.478957 8.010560 -4.166928
	glm::vec3 g(0.478957, -8.010560, 4.166928);
	g = glm::normalize(g);
	glm::mat4 gravityR(1.f, 0.f, 0.f, 0.f,
		-g.x, -g.y, -g.z, 0.f,
		0.f, g.z, -g.y, 0.f,
		0.f, 0.f, 0.f, 1.f);

	transformedPosition = gravityR * transformedPosition;

	posX = transformedPosition.x;
	posY = -transformedPosition.y;
	posZ = transformedPosition.z;
	posZ = -posZ;
//	float posX = posZ * (x - cx) / focalX;
//	float posY = posZ * (y - cy) / focalY;
//	//posZ = posZ;
//	cameraTransformation = (inverse(cameraTransformation));
//
//	glm::vec4 position(posX, posY, posZ, 1.f);
//	glm::vec4 transformedPosition = cameraTransformation * position;
//
//	posX = transformedPosition.x;
//	posY = -transformedPosition.y;
//	posZ = -transformedPosition.z;


	VAOdata[currentPos] = posX;
	VAOdata[currentPos + 1] = posY;
	VAOdata[currentPos + 2] = posZ;
	VAOdata[currentPos + 3] = (float)x / (float)imageWidth;
	VAOdata[currentPos + 4] = (float)y / (float)imageHeight; //[0, 0] is top-left at image coordinates but bottom-left at OpenGL
	staticPos[4 * index] = posX;
	staticPos[4 * index + 1] = posY;
	staticPos[4 * index + 2] = posZ; 
	staticPos[4 * index + 3] = 0.0f;
	
}

/*
* Compute normal for each vertex using 8-neighborhood.
* Loading positions can be done using shared memory to increase efficiency.
* I also perform the same calculations over and over again.
* This code has a lot of redundancy but it is just a simple test and will only be executed at startup anyway.
* Real application will get normals directly from kinect sensor (hopefully).
*/
__global__ void computeNormals(float4 *staticPos, float4 *staticNorm, int imageWidth, int imageHeight)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int index = y*imageWidth + x;
	if (x > (imageWidth - 2) || x == 0 || y == 0 || y > (imageHeight - 2))
		return;
	float4 current = staticPos[index];
	float4 neighborhood[8];
	neighborhood[0] = staticPos[(y - 1) * imageWidth + x - 1]; //top left
	neighborhood[1] = staticPos[(y - 1) * imageWidth + x]; //top mid
	neighborhood[2] = staticPos[(y - 1) * imageWidth + x + 1]; //top right
	neighborhood[3] = staticPos[(y)* imageWidth + x + 1]; //mid right
	neighborhood[4] = staticPos[(y + 1) * imageWidth + x + 1]; //bottom right
	neighborhood[5] = staticPos[(y + 1) * imageWidth + x]; //bottom mid
	neighborhood[6] = staticPos[(y + 1) * imageWidth + x - 1]; //bottom left
	neighborhood[7] = staticPos[(y)* imageWidth + x - 1]; //mid left

	//float3 normal = make_float3(0, 0, 0);
	//float meanAngleCoefficient = 0;
	//for (int neighbor = 0; neighbor < 8; neighbor++)
	//{
	//	if (neighborhood[neighbor].z > 0)
	//		neighborhood[neighbor] = current;
	//	float3 vector1 = make_float3(neighborhood[neighbor] - current);
	//	int otherNeighbor = (neighbor + 9) / 8;
	//	if (neighborhood[otherNeighbor].z > 0)
	//		neighborhood[otherNeighbor] = current;
	//	float3 vector2 = make_float3(neighborhood[otherNeighbor] - current);
	//	//float angleCoefficient = acosf(dot(vector1, vector2) / length(vector1) / length(vector2));
	//	//if (angleCoefficient != angleCoefficient)
	//		//angleCoefficient = 1.f;
	//	float angleCoefficient = 1.f;

	//	normal += normalize(cross(vector2, vector1)) * angleCoefficient;
	//	meanAngleCoefficient += angleCoefficient;

	//}
	//if (abs(meanAngleCoefficient) < 0.0000001)
	//	meanAngleCoefficient = 1.f;
	//normal /= meanAngleCoefficient;
	//normal = normalize(normal);
	/*float dzdx = (staticPos[y * imageWidth + (x + 1)].z - staticPos[y * imageWidth + (x - 1)].z) / 2.f;
	float dzdy = (staticPos[(y + 1) * imageWidth + x].z - staticPos[(y - 1) * imageWidth + x].z) / 2.f;
	float3 normal = make_float3(-dzdx, -dzdy, 1.0);
	normal = normalize(normal);*/
	const float3 dxv = make_float3(neighborhood[3] - neighborhood[7]);
	const float3 dyv = make_float3(neighborhood[5] - neighborhood[1]);
	float3 normal = normalize(cross(dyv, dxv));
	staticNorm[index] = make_float4(normal, 0);
	staticNorm[index] = make_float4(0, 1, 0, 0);
}

cudaError_t loadRangeImage(
	unsigned short *image,
	float **VAOdata,
	float **staticPos,
	float **staticNorm,
	glm::mat4 &cameraTransformation,
	float *staticRadii,
	float particleRadius,
	int imageWidth,
	int imageHeight)
{
	cudaError_t cudaStatus;
	// calculate grid size
	dim3 block(16, 32, 1);
	//dim3 block(16, 16, 1);
	dim3 grid(imageWidth / block.x, imageHeight / block.y, 1);
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
		return cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) 
		return cudaStatus;

	RGBDloader << < grid, block >> >(image,
		*VAOdata,
		*staticPos,
		cameraTransformation,
		staticRadii,
		particleRadius,
		imageWidth,
		imageHeight);
	
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) 
		return cudaStatus;
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
		return cudaStatus;

	computeNormals << < grid, block >> >((float4 *)*staticPos, (float4 *)*staticNorm, imageWidth, imageHeight);
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
		return cudaStatus;
	return cudaStatus;
}


__global__ void simpleRGBDloader(
	unsigned short *image,
	float *positions,
	int imageWidth,
	int imageHeight)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int index = y*imageWidth + x;
	if (index > (imageWidth - 1) * (imageHeight - 1))
		return;

	//kinect sensor parameters (currently hardcoded)
	//found somewhere on stack overflow - work well for now
	//should be calibrated properly for specific sensor
//	float focalX = 579.83;
//	float focalY = 586.73;
//	float cx = 321.55;
//	float cy = 235.01;
	float focalX = 528;
	float focalY = 528;
	float cx = 319.5;
	float cy = 239.5;
	float posZ = image[y*imageWidth + x] / 1000.f;
	//float posZ = 100.f / 1000.f;
	if (posZ < 0.01)
		posZ = -1;
	else
		posZ += 3; //add custom offset



	float posX = posZ * (x - cx) / focalX;
	float posY = -posZ * (y - cy) / focalY;



	posZ = -posZ;

	positions[4 * index] = posX;
	positions[4 * index + 1] = posY;
	positions[4 * index + 2] = posZ;
	positions[4 * index + 3] = 1.0f;

}

cudaError_t simpleLoadRangeImage(
	unsigned short *image,
	float *positions,
	int imageWidth,
	int imageHeight)
{
	cudaError_t cudaStatus;
	// calculate grid size
	dim3 block(16, 32, 1);
	//dim3 block(16, 16, 1);
	dim3 grid(imageWidth / block.x, imageHeight / block.y, 1);

	simpleRGBDloader << < grid, block >> >(
		image,
		positions,
		imageWidth,
		imageHeight);

	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
		return cudaStatus;

	return cudaStatus;
}

__global__ void simpleInitializerKernel(
	float *positions,
	int elements)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= elements)
		return;


	positions[4 * index] = 4.f;
	positions[4 * index + 1] = 3.f;
	positions[4 * index + 2] = 2.f;
	positions[4 * index + 3] = 1.0f;

}

cudaError_t simpleInitializer(
		float *positions,
		int elements)
{

	cudaError_t cudaStatus;
		int numThreads = 64;
		int numBlocks = (elements + numThreads - 1) / numThreads;
		std::cout << "Number of blocks: " << numBlocks << std::endl;
		curandStateTest_t *state = new curandStateTest_t;
		simpleInitializerKernel << < numBlocks, numThreads >> >(
				positions,
				elements);
		delete state;
		if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
			return cudaStatus;
		if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
			return cudaStatus;

		return cudaStatus;
}
