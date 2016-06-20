#include "PhysicsEngineCUDA.h"
PhysicsEngineCUDA::PhysicsEngineCUDA(int numThreads)
{
	numberOfThreads = numThreads;
	offset = 0.1; //remove in future versions - only used for animation
	timeStep = 0.0001;
}


PhysicsEngineCUDA::~PhysicsEngineCUDA()
{
}

cudaError_t PhysicsEngineCUDA::registerResources(GLuint &GLvao, int number, size_t sz)
{
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		return error("registerResources_cudaSetDevice");
	cudaStatus = cudaGLSetGLDevice(0);
	if (cudaStatus != cudaSuccess)
		return error("registerResources_cudaGLSetGLDevice");
	cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaVAO, GLvao, cudaGraphicsMapFlagsWriteDiscard);
	if (cudaStatus != cudaSuccess)
		return error("registerResources_cudaGraphicsGLRegisterBuffer");
	numberOfParticles = number;
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
	initialization(positions, &linearMomenta, numberOfParticles, numberOfThreads);
	if (cudaStatus != cudaSuccess)
		return error("initialize_launchKernel");
	
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
	float3* positions;

	cudaStatus = cudaGraphicsMapResources(1, &cudaVAO, 0);
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_cudaGraphicsMapResources");

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&positions, &numBytes, cudaVAO);
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_cudaGraphicsResourceGetMappedPointer");

	cudaStatus = detectCollisions(positions, &linearMomenta, timeStep, numberOfParticles, numberOfThreads);
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
	cudaStatus = animation(positions, offset, numberOfParticles, numberOfThreads);
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
	/*std::cout << "Enter random character to continue..." << std::endl;
	int x;
	std::cin >> x;*/
	cleanUp();
	return cudaStatus;
}

cudaError_t PhysicsEngineCUDA::cleanUp()
{
	cudaError_t temp;
	temp = cudaSuccess;
	if (linearMomenta)
		temp = cleanup((void**)&linearMomenta);
	return temp;
}