#include "PhysicsEngineCUDA.h"


PhysicsEngineCUDA::PhysicsEngineCUDA()
{
	offset = 0.1; //remove in future versions - only used for animation
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
	dummyInitialization(positions, numOfParticles);
	
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

	/*float d = 0.1; //grid size
	float3 s; //smallest grid coordinates
	s.x = -10.f;
	s.y = -10.f;
	s.z = -10.f;

	cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(float4) * 10, 10, 10);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_make_cudaExtent");

	cudaPitchedPtr grid; //3D grid texture
	cudaStatus = cudaMalloc3D(&grid, volumeSizeBytes);
	if (cudaStatus != cudaSuccess)
		return error("collisionDetection_cudaMalloc3D");

	// Launch a kernel on the GPU with one thread for each element.
	dummyMeshCreation(positions, grid, s, d, numOfParticles);
	*/
	CollisionList *collisions;
	collisions = detectCollisions(positions, numOfParticles);

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
	dummyAnimation(positions, offset, numOfParticles);

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