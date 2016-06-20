#pragma once
#include <GL/glew.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <iostream>
#include "Primitives.h"
//list of kernel declarations
//these kernels must be actually implemented in a separate .cu file
//in these project they are implemented in physicsKernel.cu
//use wrapper functions instead of kernels because kernels can only be invoked in .cu files
cudaError_t initialization(float3* positions, float3** linearMomenta, const int &numberOfPrimitives, const int &numberOfThreads);
cudaError_t animation(float3* positions, const double &offset, const int &numberOfPrimitives, const int &numberOfThreads);
cudaError_t detectCollisions(float3 *positions, float3 **linearMomenta, const float &timeStep, const int &numberOfPrimitives, const int &numberOfThreads);
cudaError_t cleanup(void** pt);
/**
CUDA functionality wrapper class
*/
class PhysicsEngineCUDA
{
public:
	//input number of threads
	//default is minimum observed maximum available threads - 512 for my laptop
	PhysicsEngineCUDA(int numThreads = 512);
	~PhysicsEngineCUDA();

	//register shared memory
	cudaError_t registerResources(GLuint &GLvao, int number, size_t sz);

	//unregister shared memory
	cudaError_t unregisterResources();

	//initialize particle state vectors
	cudaError_t initialize();

	//detect and handle particle-particle and particle-object collisions
	cudaError_t collisionDetection();

	//update particle state vectors
	cudaError_t update();

	//animate particles (only used for testing purposes)
	cudaError_t animate();

	//function responsible for cleaning up device memory allocated by this class (must be called explicitly)
	cudaError_t cleanUp();
private:
	struct cudaGraphicsResource *cudaVAO; //CUDA-OpenGL shared memory
	
	float timeStep;
	float3 *linearMomenta; //use a double pointer so that the device pointer can remain valid for all calls
	size_t numBytes; //number of bytes in VAO
	int numberOfThreads; //number of threads to use, input by user, default is GPU dependent
	int numberOfParticles; //total number of particles, input by user
	double offset; //animation offset (only used for testing purposes)
	cudaError_t cudaStatus;

	cudaError_t error(char *func);
};

