#include <GL/glew.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"*/
#include <stdio.h>

cudaError_t animateWithCuda(struct cudaGraphicsResource* testingVBO_CUDA, size_t *num_bytes, double offset);

__global__ void animateKernel(float3* positions, double offset)
{
	int i = threadIdx.y * blockDim.x + threadIdx.x;
	int j = threadIdx.y;
	positions[i].x += offset;
	positions[i].y = (float)j;
	positions[i].z = -10.0;
}

// Helper function for using CUDA to animate particles.
cudaError_t animateWithCuda(struct cudaGraphicsResource* testingVBO_CUDA, size_t *num_bytes, double offset)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	/*cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}*/
	float3* positions;

	cudaStatus = cudaGraphicsMapResources(1, &testingVBO_CUDA, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGraphicsMapResources returned error code %d before launching animateKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&positions, num_bytes, testingVBO_CUDA);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGraphicsResourceGetMappedPointer returned error code %d before launching animateKernel!\n", cudaStatus);
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	animateKernel << <2, 512 >> >(positions, offset);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "initializeKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching animateKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGraphicsUnmapResources(1, &testingVBO_CUDA, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGraphicsUnmapResources returned error code %d after launching animateKernel!\n", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}
