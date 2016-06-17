#include "PhysicsEngineCUDA.h"
__global__ void initializeKernel(float3* positions, float3* linearMomenta);

PhysicsEngineCUDA::PhysicsEngineCUDA()
{
	offset = 0.1; //remove in future versions - only used for animation
}


PhysicsEngineCUDA::~PhysicsEngineCUDA()
{
	cudaFree(linearMomenta);
}

cudaError_t PhysicsEngineCUDA::registerResources(GLuint &GLvao, int number, size_t sz)
{
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		return error("registerResources");
	cudaStatus = cudaGLSetGLDevice(0);
	if (cudaStatus != cudaSuccess)
		return error("registerResources");
	cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaVAO, GLvao, cudaGraphicsMapFlagsWriteDiscard);
	if (cudaStatus != cudaSuccess)
		return error("registerResources");
	numOfParticles = number;
	numBytes = sz;
	return cudaStatus;
}

cudaError_t PhysicsEngineCUDA::unregisterResources()
{
	cudaStatus = cudaGraphicsUnregisterResource(cudaVAO);
	if (cudaStatus != cudaSuccess)
		return error("unregisterResources");
	return cudaStatus;
}

cudaError_t PhysicsEngineCUDA::initialize()
{
	float3* positions;

	cudaStatus = cudaGraphicsMapResources(1, &cudaVAO, 0);
	if (cudaStatus != cudaSuccess)
		return error("initialize_cudaGraphicsMapResources");

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&positions, &numBytes, cudaVAO);
	if (cudaStatus != cudaSuccess)
		return error("initialize_cudaGraphicsResourceGetMappedPointer");

	// Launch a kernel on the GPU with one thread for each element.
	dummyInitialization(positions, linearMomenta, numOfParticles);
	if (cudaStatus != cudaSuccess)
		return error("initialize_dummyInitialization");
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		return error("initialize_cudaDeviceSynchronize");

	cudaStatus = cudaGraphicsUnmapResources(1, &cudaVAO, 0);
	if (cudaStatus != cudaSuccess)
		return error("initialize_cudaGraphicsUnmapResources");

	return cudaStatus;
}

/*
TODO:
a) Investigate the comment that morton3D works for points inside the unit cube [0, 1].
b) Stress test collision detection after adding primitive creation. [barely acceptable for 2 >> 18 particles]
c) Implement tree traversal.
d) Add collisions properly.
*/
cudaError_t PhysicsEngineCUDA::collisionDetection()
{
	float3* positions;

	cudaStatus = cudaGraphicsMapResources(1, &cudaVAO, 0);
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_cudaGraphicsMapResources");

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&positions, &numBytes, cudaVAO);
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_cudaGraphicsResourceGetMappedPointer");

	cudaStatus = detectCollisions(positions, linearMomenta, numOfParticles);
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_detectCollisions");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_cudaDeviceSynchronize");

	cudaStatus = cudaGraphicsUnmapResources(1, &cudaVAO, 0);
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_cudaGraphicsUnmapResources");

	return cudaStatus;
}

cudaError_t PhysicsEngineCUDA::update()
{
	return cudaStatus;
}

cudaError_t PhysicsEngineCUDA::animate()
{
	float3* positions;

	cudaStatus = cudaGraphicsMapResources(1, &cudaVAO, 0);
	if (cudaStatus != cudaSuccess)
		return error("animate_cudaGraphicsMapResources");

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&positions, &numBytes, cudaVAO);
	if (cudaStatus != cudaSuccess)
		return error("animate_cudaGraphicsResourceGetMappedPointer");

	// Launch a kernel on the GPU with one thread for each element.
	cudaStatus = dummyAnimation(positions, offset, numOfParticles);
	if (cudaStatus != cudaSuccess)
		return error("initialize_dummyAnimation");
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return error("animate_cudaGetLastError");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		return error("animate_cudaDeviceSynchronize");

	cudaStatus = cudaGraphicsUnmapResources(1, &cudaVAO, 0);
	if (cudaStatus != cudaSuccess)
		return error("animate_cudaGraphicsUnmapResources");

	offset = -offset;
	return cudaStatus;
}

cudaError_t PhysicsEngineCUDA::error(char *func)
{
	std::cout << "CUDA engine failed!" << std::endl;
	std::cout << "callback function: " << func << std::endl;
	std::cout << "Error type: " << cudaStatus << std::endl;
	std::cout << "Enter random character to continue..." << std::endl;
	int x;
	std::cin >> x;
	return cudaStatus;
}