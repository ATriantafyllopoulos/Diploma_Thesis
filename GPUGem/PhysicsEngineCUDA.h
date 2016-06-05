#pragma once
#include <GL/glew.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <iostream>

//list of kernel declarations
//these kernels must be actually implemented in a separate .cu file
//in these project they are implemented in physicsKernel.cu
//use dummy functions instead of kernels because kernels can only be invoked in .cu files
void dummyInitialization(float3* positions, const int &numberOfParticles);
void dummyAnimation(float3* positions, const double &offset, const int &numberOfParticles);

/**
CUDA functionality wrapper class
*/
class PhysicsEngineCUDA
{
public:
	PhysicsEngineCUDA();
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
private:
	struct cudaGraphicsResource* cudaVAO; //CUDA-OpenGL shared memory
	size_t numBytes; //number of bytes in VAO
	int numOfParticles; //total number of particles
	double offset; //animation offset (only used for testing purposes)
	cudaError_t cudaStatus;

	cudaError_t error(char *func);
};

