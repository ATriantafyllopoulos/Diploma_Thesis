#include "PhysicsEngineCUDA.h"


PhysicsEngineCUDA::PhysicsEngineCUDA()
{
	offset = 1; //remove in future versions
}


PhysicsEngineCUDA::~PhysicsEngineCUDA()
{
}

cudaError_t PhysicsEngineCUDA::registerResources(GLuint &GLvao, int number, size_t sz)
{
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
	dummyInitialization(positions);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return error("initialize_cudaGetLastError");

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

cudaError_t PhysicsEngineCUDA::collisionDetection()
{
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
	dummyAnimation(positions, offset);

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
	system("pause"); //for now pause system when an error occurs (only for debug purposes)
	return cudaStatus;
}