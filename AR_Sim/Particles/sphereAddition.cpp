#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
#include "BVHcreation.h"

#include <iostream>
#include <fstream>
#include <sstream>

template <typename T1, typename T2>
void allocateMemory(T1 **oldArray, uint size, T2 *initValue)
{
	T1 *newArray;
	cudaMalloc((void**)&newArray, sizeof(T1) * size); //placeholder for new array
	cudaMemcpy(newArray, initValue, sizeof(T1) * size, cudaMemcpyHostToDevice); //copy new value from host
	if (*oldArray)
		cudaFree(*oldArray); //free old array
	*oldArray = newArray; //pointer to newly allocated space
}

template <typename T1, typename T2>
void reAllocateMemory(T1 **oldArray, uint size, T2 *initValue, uint newElements, uint oldElements, cudaMemcpyKind TransferType = cudaMemcpyHostToDevice)
{
	T1 *newArray;
	cudaMalloc((void**)&newArray, sizeof(T1) * size); //placeholder for new array
	cudaMemcpy(newArray, *oldArray, sizeof(T1) * oldElements, cudaMemcpyDeviceToDevice); //copy old array to new array
	cudaMemcpy(&newArray[oldElements], initValue, sizeof(T1) * newElements, TransferType); //copy new value from host or device

	if (*oldArray)
		cudaFree(*oldArray); //free old array
	*oldArray = newArray; //pointer to newly allocated space
}

inline float myFRand()
{
	return rand() / (float)RAND_MAX;
}

void ParticleSystem::addSphere(int start, glm::vec3 pos, glm::vec3 vel, int r, float spacing)
{
	if (!m_numParticles)
	{
		_initialize(1024);
	}
	objectsThrown++;
	uint index = start;

	for (int z = -r; z <= r; z++)
	{
		for (int y = -r; y <= r; y++)
		{
			for (int x = -r; x <= r; x++)
			{
				float dx = x*spacing;
				float dy = y*spacing;
				float dz = z*spacing;
				float l = sqrtf(dx*dx + dy*dy + dz*dz);
				float jitter = m_params.particleRadius*0.01f;

				if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
				{
					m_hPos[index * 4] = pos.x + dx + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 1] = pos.y + dy + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 2] = pos.z + dz + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 3] = 0.f;

					m_hVel[index * 4] = vel.x;
					m_hVel[index * 4 + 1] = vel.y;
					m_hVel[index * 4 + 2] = vel.z;
					m_hVel[index * 4 + 3] = 0.f;
					index++;
				}
			}
		}
	}

	std::cout << "Initialized " << index << " out of " << m_numParticles << " particles." << std::endl;
	//setArray(POSITION, m_hPos, start, index);
	//setArray(VELOCITY, m_hVel, start, index);
	unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
	unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
	size_t memSize = sizeof(float) * 4 * m_numParticles;

	glGenVertexArrays(1, &m_virtualVAO);
	glBindVertexArray(m_virtualVAO);

	glGenBuffers(1, &m_posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glGenBuffers(1, &m_colorVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glBindVertexArray(0);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	checkCudaErrors(cudaMemcpy(dPos, m_hPos, sizeof(float) * 4 * m_numParticles, cudaMemcpyHostToDevice)); //new particles position array


	int *indices = new int[m_numParticles];
	for (int i = 0; i < m_numParticles; i++)
		indices[i] = -1;
	allocateMemory(&rbIndices, m_numParticles, indices);
	delete indices;

	float *newValue = new float[4 * m_numParticles];
	memset(newValue, 0, 4 * m_numParticles * sizeof(float));
	allocateMemory(&pForce, 4 * m_numParticles, newValue);
	//allocateMemory(&pPositions, 4 * m_numParticles, newValue);
	allocateMemory(&pTorque, 4 * m_numParticles, newValue);
	delete newValue;

	if (isRigidBody)
		delete isRigidBody;
	isRigidBody = new bool[objectsThrown];
	bool newObject = false;
	memcpy(&isRigidBody[(objectsThrown - 1)], &newObject, sizeof(bool));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	allocateMemory(&m_dVel, 4 * m_numParticles, m_hVel); //new particles velocity array

	particlesPerObjectThrown = new int[objectsThrown];
	int newParticles = m_numParticles - start;
	memcpy(particlesPerObjectThrown, &newParticles, sizeof(int));

	//float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	cudaMalloc((void**)&relativePos, sizeof(float) * 4 * m_numParticles);
	mapRelativePositionIndependentParticlesWrapper(
		(float4 *)dPos, //particle positions
		(float4 *)relativePos, //relative particle positions
		rbIndices, //rigid body indices
		m_numParticles,
		numThreads);
	unmapGLBufferObject(m_cuda_posvbo_resource);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	initializeVirtualSoA(); //initialize SoA variables for virtual particles
	reallocGridAuxiliaries();
}

void ParticleSystem::addNewSphere(int particles, glm::vec3 pos, glm::vec3 vel, int r, float spacing)
{
	if (!m_numParticles)
	{
		_initialize(1024);
		std::cout << "Got here safely." << std::endl;
		addSphere(0, pos, vel, r, spacing);
		return;
	}
	objectsThrown++;
	//std::cout << "Number of particles before new addition: " << m_numParticles << std::endl;
	std::cout << "Adding new sphere consisting of: " << particles << " particles" << std::endl;
	//reallocate host memory to fit new data
	float *oldPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	float *h_newPos = new float[(m_numParticles + particles) * 4]; //add #(particles) new particles to our system
	float *h_newVel = new float[(m_numParticles + particles) * 4];
	cudaMemcpy(h_newPos, oldPos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost); //copy old positions array to newly allocated space
	cudaMemcpy(h_newVel, m_dVel, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost); //copy old velocities array to newly allocated space
	free(m_hPos); //free old positions array
	m_hPos = h_newPos; //change pointer to new positions array
	free(m_hVel); //free old velocities array
	m_hVel = h_newVel; //change pointer to new velocities array
	unmapGLBufferObject(m_cuda_posvbo_resource);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	int start = m_numParticles;
	for (int z = -r; z <= r; z++)
	{
		for (int y = -r; y <= r; y++)
		{
			for (int x = -r; x <= r; x++)
			{
				float dx = x*spacing;
				float dy = y*spacing;
				float dz = z*spacing;
				float l = sqrtf(dx*dx + dy*dy + dz*dz);
				float jitter = m_params.particleRadius*0.01f;

				if ((l <= m_params.particleRadius*2.0f*r) && (m_numParticles - start) < particles)
				{
					int index = m_numParticles;
					m_hPos[index * 4] = pos.x + dx + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 1] = pos.y + dy + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 2] = pos.z + dz + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 3] = 0.f;

					/*std::cout << "Position @: " << m_numParticles << " is: (" << m_hPos[index * 4] << ", " <<
					m_hPos[index * 4 + 1] << ", " << m_hPos[index * 4 + 2] << ", " << m_hPos[index * 4 + 3] <<
					")" << std::endl;*/

					m_hVel[index * 4] = vel.x;
					m_hVel[index * 4 + 1] = vel.y;
					m_hVel[index * 4 + 2] = vel.z;
					m_hVel[index * 4 + 3] = 0.f;
					m_numParticles++;
					/*std::cout << "Velocity @: " << m_numParticles << " is: (" << m_hVel[index * 4] << ", " <<
					m_hVel[index * 4 + 1] << ", " << m_hVel[index * 4 + 2] << ", " << m_hVel[index * 4 + 3] <<
					")" << std::endl;*/
				}
			}
		}
	}
	std::cout << "Number of particles after newest addition: " << m_numParticles << std::endl;
	//reallocate client (GPU) memory to fit new data

	unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
	unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
	size_t memSize = sizeof(float) * 4 * m_numParticles;

	glGenVertexArrays(1, &m_virtualVAO);
	glBindVertexArray(m_virtualVAO);

	glGenBuffers(1, &m_posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glGenBuffers(1, &m_colorVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glBindVertexArray(0);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
    checkCudaErrors(cudaMemcpy(dPos, m_hPos, sizeof(float) * 4 * m_numParticles, cudaMemcpyHostToDevice)); //new particles position array
    //unmapGLBufferObject(m_cuda_posvbo_resource);

	reAllocateMemory(&m_dVel, 4 * m_numParticles, &m_hVel[start * 4], (m_numParticles - start) * 4, start * 4); //new particles velocity array
	int *indices = new int[(m_numParticles - start)];
	for (int i = 0; i < (m_numParticles - start); i++)
		indices[i] = -1; //independent virtual particles are indicated by a negative index
	reAllocateMemory(&rbIndices, m_numParticles, indices, particles, start); //new rigid body index array
	delete indices;
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *newValue = new float[4 * particles]; //all zeros (I hope)
	memset(newValue, 0, 4 * particles * sizeof(float));
	reAllocateMemory(&pForce, 4 * m_numParticles, newValue, 4 * (m_numParticles - start), 4 * start);
	//reAllocateMemory(&pPositions, 4 * m_numParticles, newValue, 4 * (m_numParticles - start), 4 * start);
	reAllocateMemory(&pTorque, 4 * m_numParticles, newValue, 4 * (m_numParticles - start), 4 * start);
	delete newValue;

	//int newParticles = 0;
	//reAllocateMemory(&particlesPerObjectThrown, objectsThrown, &newParticles, 1, (objectsThrown - 1));
	int *newparticlesPerObjectThrown = new int[objectsThrown];
	memcpy(newparticlesPerObjectThrown, particlesPerObjectThrown, sizeof(int) * (objectsThrown - 1));
	if (particlesPerObjectThrown)
		delete particlesPerObjectThrown;
	particlesPerObjectThrown = newparticlesPerObjectThrown;
	int newParticles = m_numParticles - start;
	memcpy(&particlesPerObjectThrown[(objectsThrown - 1)], &newParticles, sizeof(int));

	bool *newIsRigidBody = new bool[objectsThrown];
	memcpy(newIsRigidBody, isRigidBody, sizeof(bool) * (objectsThrown - 1));
	if (isRigidBody)
		delete isRigidBody;
	isRigidBody = newIsRigidBody;
	bool newObject = false;
	memcpy(&isRigidBody[(objectsThrown - 1)], &newObject, sizeof(bool));

    reAllocateMemory(&relativePos, 4 * m_numParticles, &m_hPos[start * 4], (m_numParticles - start) * 4, start * 4); //create new relative-actual particle position array
//	mapActualPositionRigidBodyParticlesWrapper(
//			(float4 *)dPos, //particle positions
//			(float4 *)relativePos, //relative particle positions
//			(float4 *)rbPositions, //rigid body center of mass
//			rbIndices, //rigid body indices
//			m_numParticles,
//			numThreads);
    mapRelativePositionIndependentParticlesWrapper(
        (float4 *)dPos, //particle positions
        (float4 *)relativePos, //relative particle positions
        rbIndices, //rigid body indices
        m_numParticles,
        numThreads);
    unmapGLBufferObject(m_cuda_posvbo_resource);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	initializeVirtualSoA(); //initialize SoA variables for virtual particles
	reallocGridAuxiliaries();

}

void ParticleSystem::addRigidSphere(int particles, glm::vec3 pos, glm::vec3 vel, float r, float spacing)
{
	if (!m_numParticles)
	{
		int index = 0;
		for (int z = -r; z <= r; z++)
			{
				for (int y = -r; y <= r; y++)
				{
					for (int x = -r; x <= r; x++)
					{
						float dx = x*spacing;
						float dy = y*spacing;
						float dz = z*spacing;
						float l = sqrtf(dx*dx + dy*dy + dz*dz);
						float jitter = m_params.particleRadius*0.01f;

						if ((l <= m_params.particleRadius*2.0f*r) && index < particles)
						{
							index++; //finding out how many particles fit in the new rigid body sphere
						}
					}
				}
			}
		std::cout << "Initializing rigid sphere with " << index << " particles" << std::endl;
		_initialize(index);
		initRigidSphere(index, pos, vel, r, spacing);
		return;
	}
	objectsThrown++;
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//reallocate host memory to fit new data
	float *h_newPos = new float[(m_numParticles + particles) * 4]; //add #(particles) new particles to our system
	float *h_newVel = new float[(m_numParticles + particles) * 4];
	
	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	//relativePositions[index] = positions[index] for rbIndices[index] == -1
//	mapRelativePositionRigidBodyParticlesWrapper(
//			(float4 *)dPos, //particle positions
//			(float4 *)relativePos, //relative particle positions
//			rbIndices, //rigid body indices
//			m_numParticles,
//			numThreads);
	//relativePositions[index] = positions[index] for rbIndices[index] != -1
//	mapRelativePositionRigidBodyParticlesWrapper(
//			(float4 *)dPos, //particle positions
//			(float4 *)relativePos, //relative particle positions
//			rbIndices, //rigid body indices
//			m_numParticles,
//			numThreads);
	cudaMemcpy(h_newPos, relativePos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost); //copy old positions array to newly allocated space
	unmapGLBufferObject(m_cuda_posvbo_resource);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(h_newVel, m_dVel, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost); //copy old velocities array to newly allocated space
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	free(m_hPos); //free old positions array
	m_hPos = h_newPos; //change pointer to new positions array
	free(m_hVel); //free old velocities array
	m_hVel = h_newVel; //change pointer to new velocities array
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	int start = m_numParticles;

	glm::mat3 inertiaTensor;
	for (int z = -r; z <= r; z++)
	{
		for (int y = -r; y <= r; y++)
		{
			for (int x = -r; x <= r; x++)
			{
				float dx = x*spacing;
				float dy = y*spacing;
				float dz = z*spacing;
				float l = sqrtf(dx*dx + dy*dy + dz*dz);
				float jitter = m_params.particleRadius*0.01f;

				if ((l <= m_params.particleRadius*2.0f*r) && (m_numParticles - start) < particles)
				{
					int index = m_numParticles;
					m_hPos[index * 4] = dx + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 1] = dy + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 2] = dz + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 3] = 0.f;
					
					inertiaTensor[0][0] += m_hPos[index * 4 + 1] * m_hPos[index * 4 + 1] + m_hPos[index * 4 + 2] * m_hPos[index * 4 + 2]; //y*y + z*z
					inertiaTensor[0][1] -= m_hPos[index * 4 + 1] * m_hPos[index * 4]; //x*y
					inertiaTensor[0][2] -= m_hPos[index * 4 + 2] * m_hPos[index * 4]; //x*z

					inertiaTensor[1][0] -= m_hPos[index * 4 + 1] * m_hPos[index * 4]; //x*y
					inertiaTensor[1][1] += m_hPos[index * 4] * m_hPos[index * 4] + m_hPos[index * 4 + 2] * m_hPos[index * 4 + 2]; //x*x + z*z
					inertiaTensor[1][2] -= m_hPos[index * 4 + 2] * m_hPos[index * 4 + 1]; //y*z

					inertiaTensor[2][0] -= m_hPos[index * 4 + 2] * m_hPos[index * 4]; //x*z
					inertiaTensor[2][1] -= m_hPos[index * 4 + 2] * m_hPos[index * 4 + 1]; //y*z
					inertiaTensor[2][2] += m_hPos[index * 4 + 1] * m_hPos[index * 4 + 1] + m_hPos[index * 4] * m_hPos[index * 4]; //x*x + y*y

//					m_hVel[index * 4] = vel.x;
//					m_hVel[index * 4 + 1] = vel.y;
//					m_hVel[index * 4 + 2] = vel.z;
//					m_hVel[index * 4 + 3] = 0.f;
					m_hVel[index * 4] = 0.f;
					m_hVel[index * 4 + 1] = 0.f;
					m_hVel[index * 4 + 2] = 0.f;
					m_hVel[index * 4 + 3] = 0.f;
					m_numParticles++;
				}
			}
		}
	}
	//float inverseInertia[9];
	//invert(inertiaTensor, inverseInertia);
	inertiaTensor = inverse(inertiaTensor);

	std::cout << "Number of particles after newest addition: " << m_numParticles << std::endl;
	//reallocate client (GPU) memory to fit new data
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
	unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	glGenVertexArrays(1, &m_virtualVAO);
	glBindVertexArray(m_virtualVAO);

	glGenBuffers(1, &m_posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glGenBuffers(1, &m_colorVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glBindVertexArray(0);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//re-allocate memory to fit new data

	reAllocateMemory(&relativePos, 4 * m_numParticles, m_hPos, 4 * m_numParticles, 0); //create new relative-actual particle position array

	reAllocateMemory(&m_dVel, 4 * m_numParticles, m_hVel, 4 * (m_numParticles - start), 4 * start); //new particle velocity array


	numRigidBodies++; //increase number of rigid bodies
	std::cout << "Number of rigid bodies after newest addition: " << numRigidBodies << std::endl;

//	float4 *testVel = new float4[(numRigidBodies - 1)];
//	cudaMemcpy(testVel, rbPositions, sizeof(float) * 4 *(numRigidBodies - 1), cudaMemcpyDeviceToHost);
//	std::cout << "Printing old positions..." << std::endl;
//	for(int i = 0; i < (numRigidBodies - 1); i++)
//		std::cout << "Rigid body #" << i+1 << ": (" << testVel[i].x << " " << testVel[i].y << " " << testVel[i].z << ")" << std::endl;
//	delete testVel;
	float4 newValue = make_float4(pos.x, pos.y, pos.z, 0);
	reAllocateMemory(&rbPositions, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body center of mass array

//	testVel = new float4[(numRigidBodies)];
//	cudaMemcpy(testVel, rbPositions, sizeof(float) * 4 *(numRigidBodies), cudaMemcpyDeviceToHost);
//	std::cout << "Printing new positions0.3*..." << std::endl;
//	for(int i = 0; i < (numRigidBodies); i++)
//		std::cout << "Rigid body #" << i+1 << ": (" << testVel[i].x << " " << testVel[i].y << " " << testVel[i].z << ")" << std::endl;
//	delete testVel;
	newValue = make_float4(vel.x, vel.y, vel.z, 0);
	reAllocateMemory(&rbVelocities, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1));//new rigid body velocity array

	newValue = make_float4(0, 0, 0, 0);
	reAllocateMemory(&rbAngularVelocity, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body angular velocity array
	glm::vec3 newAngularAcceleration(0, 0, 0);
	//reAllocateMemory(&rbAngularAcceleration, numRigidBodies, &newAngularAcceleration, 1, numRigidBodies - 1); //new rigid body angular velocity array
	reAllocateMemory(&rbInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
	reAllocateMemory(&rbCurrentInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array

	glm::quat newQuatValue(1, 0, 0, 0);
	reAllocateMemory(&rbQuaternion, numRigidBodies, &newQuatValue, 1, (numRigidBodies - 1)); //new rigid body quaternion array


	newValue = make_float4(0, 0, 0, 0);
	//reAllocateMemory(&rbForces, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array

	//ISSUE: rigid bodies have the same mass as particles
	float newMass = 1.f / 15.f;//(float)(m_numParticles - start); //all rigid bodies have a mass of 1
	reAllocateMemory(&rbMass, numRigidBodies, &newMass, 1, (numRigidBodies - 1)); //new rigid body mass array
	newValue = make_float4(0, 0, 0, 0);
	//reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(vel.x , vel.y, vel.z, 0) / newMass;
	//reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(0, 0, 0, 0);
	//reAllocateMemory(&rbTorque, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body torque array - possibly not needed

	float newRadius = m_params.particleRadius*2.0f*r;
	//reAllocateMemory(&rbRadii, numRigidBodies, &newRadius, 1, (numRigidBodies - 1)); //new rigid body radius array



	int *newparticlesPerObjectThrown = new int[objectsThrown];
	memcpy(newparticlesPerObjectThrown, particlesPerObjectThrown, sizeof(int) * (objectsThrown - 1));
	if (particlesPerObjectThrown)
		delete particlesPerObjectThrown;
	particlesPerObjectThrown = newparticlesPerObjectThrown;
	int newParticles = m_numParticles - start;
	memcpy(&particlesPerObjectThrown[(objectsThrown - 1)], &newParticles, sizeof(int));

	bool *newIsRigidBody = new bool[objectsThrown];
	memcpy(newIsRigidBody, isRigidBody, sizeof(bool) * (objectsThrown - 1));
	if (isRigidBody)
		delete isRigidBody;
	isRigidBody = newIsRigidBody;
	bool newObject = true;
	memcpy(&isRigidBody[(objectsThrown - 1)], &newObject, sizeof(bool));

	int *indices = new int[(m_numParticles - start)];
	for (int i = 0; i < (m_numParticles - start); i++)
		indices[i] = numRigidBodies - 1; //new rigid body index

	reAllocateMemory(&rbIndices, m_numParticles, indices, (m_numParticles - start), start); //new rigid body index array
	delete indices;
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *newParticleValue = new float[4 * particles]; //all zeros (I hope)
	memset(newParticleValue, 0, 4 * particles * sizeof(float));
	reAllocateMemory(&pForce, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
	//reAllocateMemory(&pPositions, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
	reAllocateMemory(&pTorque, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
	delete newParticleValue;

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);

	mapActualPositionRigidBodyParticlesWrapper(
			(float4 *)dPos, //particle positions
			(float4 *)relativePos, //relative particle positions
			(float4 *)rbPositions, //rigid body center of mass
			rbIndices, //rigid body indices
			m_numParticles,
			numThreads);

	mapActualPositionIndependentParticlesWrapper(
			(float4 *)dPos, //particle positions
			(float4 *)relativePos, //relative particle positions
			rbIndices, //rigid body indices
			m_numParticles,
			numThreads);
	unmapGLBufferObject(m_cuda_posvbo_resource);



	reallocGridAuxiliaries();

	//number of virtual particles has changed! re-initialize SoA
	initializeVirtualSoA(); //initialize SoA variables for virtual particles

	//updateRigidBodies(0.01); //update simulation so new values are copied
	//testing variables
//	float *relTest = new float[4 * m_numParticles];
//	checkCudaErrors(cudaMemcpy(relTest, relativePos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost));
//	std::cout << "Relative positions of particles belonging to latest rigid body" << std::endl;
//	for(int i = m_numParticles - particles; i < m_numParticles; i++)
//	{
//		std::cout << "Particle #" << i + 1 << " relative position (" << relTest[4*i] << " " <<
//				relTest[4*i+1] << " " << relTest[4*i+2] << " " << relTest[4*i+3] << ")" << std::endl;
//	}
//	delete relTest;
//
//
//	float *cmTest = new float[4 * numRigidBodies];
//	checkCudaErrors(cudaMemcpy(cmTest, rbPositions, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
//	for(int i = 0; i < numRigidBodies; i++)
//	{
//		std::cout << "Rigid body #" << i + 1 << " center of mass: (" << cmTest[4*i] << " " <<
//				cmTest[4*i+1] << " " << cmTest[4*i+2] << " " << cmTest[4*i+3] << ")" << std::endl;
//	}
//	delete cmTest;


}

void ParticleSystem::initRigidSphere(int particles, glm::vec3 pos, glm::vec3 vel, int r, float spacing)
{
	objectsThrown++;
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int start = 0;
	int index = 0;
	glm::mat3 inertiaTensor;
	for (int z = -r; z <= r; z++)
	{
		for (int y = -r; y <= r; y++)
		{
			for (int x = -r; x <= r; x++)
			{
				float dx = x*spacing;
				float dy = y*spacing;
				float dz = z*spacing;
				float l = sqrtf(dx*dx + dy*dy + dz*dz);
				float jitter = m_params.particleRadius*0.01f;

				if ((l <= m_params.particleRadius*2.0f*r) && index < m_numParticles)
				{
					m_hPos[index * 4] = dx + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 1] = dy + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 2] = dz + (myFRand()*2.0f - 1.0f)*jitter;
					m_hPos[index * 4 + 3] = 0.f;

					inertiaTensor[0][0] += m_hPos[index * 4 + 1] * m_hPos[index * 4 + 1] + m_hPos[index * 4 + 2] * m_hPos[index * 4 + 2]; //y*y + z*z
					inertiaTensor[0][1] -= m_hPos[index * 4 + 1] * m_hPos[index * 4]; //x*y
					inertiaTensor[0][2] -= m_hPos[index * 4 + 2] * m_hPos[index * 4]; //x*z

					inertiaTensor[1][0] -= m_hPos[index * 4 + 1] * m_hPos[index * 4]; //x*y
					inertiaTensor[1][1] += m_hPos[index * 4] * m_hPos[index * 4] + m_hPos[index * 4 + 2] * m_hPos[index * 4 + 2]; //x*x + z*z
					inertiaTensor[1][2] -= m_hPos[index * 4 + 2] * m_hPos[index * 4 + 1]; //y*z

					inertiaTensor[2][0] -= m_hPos[index * 4 + 2] * m_hPos[index * 4]; //x*z
					inertiaTensor[2][1] -= m_hPos[index * 4 + 2] * m_hPos[index * 4 + 1]; //y*z
					inertiaTensor[2][2] += m_hPos[index * 4 + 1] * m_hPos[index * 4 + 1] + m_hPos[index * 4] * m_hPos[index * 4]; //x*x + y*y

					m_hVel[index * 4] = vel.x;
					m_hVel[index * 4 + 1] = vel.y;
					m_hVel[index * 4 + 2] = vel.z;
					m_hVel[index * 4 + 3] = 0.f;
					index++;
				}
			}
		}
	}
	inertiaTensor = inverse(inertiaTensor);

	std::cout << "Number of particles after newest addition: " << index << std::endl;
	//reallocate client (GPU) memory to fit new data
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
	unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	glGenVertexArrays(1, &m_virtualVAO);
	glBindVertexArray(m_virtualVAO);

	glGenBuffers(1, &m_posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glGenBuffers(1, &m_colorVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	glBindVertexArray(0);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//re-allocate memory to fit new data

	allocateMemory(&relativePos, 4 * m_numParticles, m_hPos); //create new relative-actual particle position array

	allocateMemory(&m_dVel, 4 * m_numParticles, m_hVel); //new particle velocity array
	numRigidBodies++; //increase number of rigid bodies
	std::cout << "Number of rigid bodies after newest addition: " << numRigidBodies << std::endl;

	float4 newValue = make_float4(pos.x, pos.y, pos.z, 0);
	reAllocateMemory(&rbPositions, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body center of mass array

	newValue = make_float4(vel.x, vel.y, vel.z, 0);
	reAllocateMemory(&rbVelocities, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1));//new rigid body velocity array

	newValue = make_float4(0, 0, 0, 0);
	reAllocateMemory(&rbAngularVelocity, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body angular velocity array
	glm::vec3 newAngularAcceleration(0, 0, 0);
	//reAllocateMemory(&rbAngularAcceleration, numRigidBodies, &newAngularAcceleration, 1, numRigidBodies - 1); //new rigid body angular velocity array
	reAllocateMemory(&rbInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
	reAllocateMemory(&rbCurrentInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array

	glm::quat newQuatValue(1, 0, 0, 0);
	reAllocateMemory(&rbQuaternion, numRigidBodies, &newQuatValue, 1, (numRigidBodies - 1)); //new rigid body quaternion array
	newValue = make_float4(0, 0, 0, 0);
	//reAllocateMemory(&rbForces, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array

	//ISSUE: rigid bodies have the same mass as particles
	float newMass = 1.f / 15.f;//(float)(m_numParticles); //all rigid bodies have a mass of 1
	reAllocateMemory(&rbMass, numRigidBodies, &newMass, 1, (numRigidBodies - 1)); //new rigid body mass array

	newValue = make_float4(0, 0, 0, 0);
	//reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(vel.x , vel.y, vel.z, 0) / newMass;
	//reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(0, 0, 0, 0);

	//reAllocateMemory(&rbTorque, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body torque array - possibly not needed

	float newRadius = m_params.particleRadius*2.0f*r;
	//reAllocateMemory(&rbRadii, numRigidBodies, &newRadius, 1, (numRigidBodies - 1)); //new rigid body radius array



	int *newparticlesPerObjectThrown = new int[objectsThrown];
	memcpy(newparticlesPerObjectThrown, particlesPerObjectThrown, sizeof(int) * (objectsThrown - 1));
	if (particlesPerObjectThrown)
		delete particlesPerObjectThrown;
	particlesPerObjectThrown = newparticlesPerObjectThrown;
	int newParticles = m_numParticles - start;
	memcpy(&particlesPerObjectThrown[(objectsThrown - 1)], &newParticles, sizeof(int));

	bool *newIsRigidBody = new bool[objectsThrown];
	memcpy(newIsRigidBody, isRigidBody, sizeof(bool) * (objectsThrown - 1));
	if (isRigidBody)
		delete isRigidBody;
	isRigidBody = newIsRigidBody;
	bool newObject = true;
	memcpy(&isRigidBody[(objectsThrown - 1)], &newObject, sizeof(bool));

	int *indices = new int[(m_numParticles - start)];
	for (int i = 0; i < (m_numParticles - start); i++)
		indices[i] = numRigidBodies - 1; //new rigid body index

	reAllocateMemory(&rbIndices, m_numParticles, indices, (m_numParticles - start), start); //new rigid body index array
	delete indices;
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *newParticleValue = new float[4 * particles]; //all zeros (I hope)
	memset(newParticleValue, 0, 4 * particles * sizeof(float));
	reAllocateMemory(&pForce, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
	//reAllocateMemory(&pPositions, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
	reAllocateMemory(&pTorque, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
	delete newParticleValue;

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);

	mapActualPositionRigidBodyParticlesWrapper(
			(float4 *)dPos, //particle positions
			(float4 *)relativePos, //relative particle positions
			(float4 *)rbPositions, //rigid body center of mass
			rbIndices, //rigid body indices
			m_numParticles,
			numThreads);

	mapActualPositionIndependentParticlesWrapper(
			(float4 *)dPos, //particle positions
			(float4 *)relativePos, //relative particle positions
			rbIndices, //rigid body indices
			m_numParticles,
			numThreads);
	unmapGLBufferObject(m_cuda_posvbo_resource);


	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	reallocGridAuxiliaries();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//number of virtual particles has changed! re-initialize SoA
	initializeVirtualSoA(); //initialize SoA variables for virtual particles

}

