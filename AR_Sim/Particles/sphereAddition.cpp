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

/*
 * Auxiliary function to reallocate device memory after a new addition.
 * All input pointers must be DEVICE pointers.
 * m_numParticles is increased before calling this function
 */
void ParticleSystem::addRigidBody(
		int previousParticleCount,
		int particlesAdded,
		float *newRelativePos, //new relative position - 4 * particlesAdded
		float *newParticleVelocity, //new particle velocity - 4 * particlesAdded
		glm::mat3 *newInverseInertia, //new inverse inertia tensor - 1
		float *newRigidBodyCM, //new rigid body center of mass - 4
		float *newRigidBodyVelocity, //new rigid body velocity - 4
		float *newRigidBodyAngularVelocity, //new rigid body angular velocity - 4
		glm::vec3 *newRigidBodyAngularAcceleration, //1
		glm::quat *newRigidBodyQuaternion, //new rigid body quaternion - 1
		float *newRigidBodyForce, //new rigid body force - 4
		float *newRigidBodyMass, //1
		float *newRigidBodyAngularMomentum, //4
		float *newRigidBodyLinearMomentum, //4
		float *newRigidBodyTorque, //4
		float *newRigidBodyRadius, //1
		float *newParticleForce, //4 * particlesAdded
		float *newParticleTorque, //4 * particlesAdded
		float *newParticlePosition, //4 * particlesAdded
		int *newCountARCollions, //particlesAdded
		int *newParticleIndex, //particlesAdded
		bool isRigidBodyLocal)
{


	objectsThrown++; //increase number of objects thrown
	numRigidBodies++; //increase number of rigid bodies
	std::cout << "Number of rigid bodies after newest addition: " << numRigidBodies << std::endl;


	unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
	unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	//create new VAO
	glGenVertexArrays(1, &m_virtualVAO);
	glBindVertexArray(m_virtualVAO);

	//create and initialize new position VBO
	glGenBuffers(1, &m_posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

	//create and initialize new color VBO
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

	//new per-particle values
	reAllocateMemory(&relativePos, 4 * m_numParticles, newRelativePos, 4 * particlesAdded, 4 * previousParticleCount, cudaMemcpyDeviceToDevice);
	reAllocateMemory(&m_dVel, 4 * m_numParticles, newParticleVelocity, 4 * particlesAdded, 4 * previousParticleCount, cudaMemcpyDeviceToDevice);
	reAllocateMemory(&pForce, 4 * m_numParticles, newParticleForce, 4 * particlesAdded, 4 * previousParticleCount, cudaMemcpyDeviceToDevice);
	reAllocateMemory(&pPositions, 4 * m_numParticles, newParticlePosition, 4 * particlesAdded, 4 * previousParticleCount, cudaMemcpyDeviceToDevice);
	reAllocateMemory(&pTorque, 4 * m_numParticles, newParticleTorque, 4 * particlesAdded, 4 * previousParticleCount, cudaMemcpyDeviceToDevice);
	reAllocateMemory(&rbIndices, m_numParticles, newParticleIndex, particlesAdded, previousParticleCount, cudaMemcpyDeviceToDevice); //new rigid body index array
	reAllocateMemory(&pCountARCollions, m_numParticles, newCountARCollions, particlesAdded, previousParticleCount, cudaMemcpyDeviceToDevice);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//new per rigid body values
	reAllocateMemory(&rbInertia, numRigidBodies, newInverseInertia, 1, (numRigidBodies-1), cudaMemcpyDeviceToDevice);
	reAllocateMemory(&rbCurrentInertia, numRigidBodies, newInverseInertia, 1, (numRigidBodies - 1), cudaMemcpyDeviceToDevice);
	reAllocateMemory(&rbAngularAcceleration, numRigidBodies, newRigidBodyAngularAcceleration, 1, numRigidBodies - 1, cudaMemcpyDeviceToDevice); //new rigid body angular velocity array
	reAllocateMemory(&rbQuaternion, numRigidBodies, newRigidBodyQuaternion, 1, (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body quaternion array
	reAllocateMemory(&rbRadii, numRigidBodies, newRigidBodyRadius, 1, (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body radius array
	reAllocateMemory(&rbMass, numRigidBodies, newRigidBodyMass, 1, (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body mass array

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	reAllocateMemory(&rbPositions, 4 * numRigidBodies, newRigidBodyCM, 4, 4 * (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body center of mass array
	reAllocateMemory(&rbVelocities, 4 * numRigidBodies, newRigidBodyVelocity, 4, 4 * (numRigidBodies - 1), cudaMemcpyDeviceToDevice);//new rigid body velocity array
	reAllocateMemory(&rbAngularVelocity, 4 * numRigidBodies, newRigidBodyAngularVelocity, 4, 4 * (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body angular velocity array
	reAllocateMemory(&rbForces, 4 * numRigidBodies, newRigidBodyForce, 4, 4 * (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body force array
	reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, newRigidBodyAngularMomentum, 4, 4 * (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body force array
	reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, newRigidBodyLinearMomentum, 4, 4 * (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body force array
	reAllocateMemory(&rbTorque, 4 * numRigidBodies, newRigidBodyTorque, 4, 4 * (numRigidBodies - 1), cudaMemcpyDeviceToDevice); //new rigid body torque array - possibly not needed

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//new CPU variables
	int *newparticlesPerObjectThrown = new int[objectsThrown];
	memcpy(newparticlesPerObjectThrown, particlesPerObjectThrown, sizeof(int) * (objectsThrown - 1));
	if (particlesPerObjectThrown)
		delete particlesPerObjectThrown;
	particlesPerObjectThrown = newparticlesPerObjectThrown;

	int newParticles = particlesAdded;
	memcpy(&particlesPerObjectThrown[(objectsThrown - 1)], &newParticles, sizeof(int));

	bool *newIsRigidBody = new bool[objectsThrown];
	memcpy(newIsRigidBody, isRigidBody, sizeof(bool) * (objectsThrown - 1));


	if (isRigidBody)
		delete isRigidBody;
	isRigidBody = newIsRigidBody;
	memcpy(&isRigidBody[(objectsThrown - 1)], &isRigidBodyLocal, sizeof(bool));

	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);

	mapActualPositionRigidBodyParticlesWrapper(
			(float4 *)dPos, //particle positions
			(float4 *)relativePos, //relative particle positions
			(float4 *)rbPositions, //rigid body center of mass
			rbIndices, //rigid body indices
			m_numParticles,
			numThreads);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

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
	//number of virtual particles has changed! re-initialize SoA
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	initializeVirtualSoA(); //initialize SoA variables for virtual particles

	// free collision detection buffers
	checkCudaErrors(cudaFree(collidingRigidBodyIndex));
	checkCudaErrors(cudaFree(collidingParticleIndex));
	checkCudaErrors(cudaFree(contactDistance));

	// allocate collision detection buffers
	checkCudaErrors(cudaMalloc((void**)&collidingRigidBodyIndex, sizeof(int) * m_numParticles));
	checkCudaErrors(cudaMalloc((void**)&collidingParticleIndex, sizeof(int) * m_numParticles));
	checkCudaErrors(cudaMalloc((void**)&contactDistance, sizeof(float) * m_numParticles));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//free all unnecessary device allocations
	if(newRelativePos)cudaFree(newRelativePos);
	if(newParticleVelocity)cudaFree(newParticleVelocity);
	if(newInverseInertia)cudaFree(newInverseInertia);
	if(newRigidBodyCM)cudaFree(newRigidBodyCM);
	if(newRigidBodyVelocity)cudaFree(newRigidBodyVelocity);
	if(newRigidBodyAngularVelocity)cudaFree(newRigidBodyAngularVelocity);
	if(newRigidBodyAngularAcceleration)cudaFree(newRigidBodyAngularAcceleration);
	if(newRigidBodyQuaternion)cudaFree(newRigidBodyQuaternion);
	if(newRigidBodyForce)cudaFree(newRigidBodyForce);
	if(newRigidBodyMass)cudaFree(newRigidBodyMass);
	if(newRigidBodyAngularMomentum)cudaFree(newRigidBodyAngularMomentum);
	if(newRigidBodyLinearMomentum)cudaFree(newRigidBodyLinearMomentum);
	if(newRigidBodyTorque)cudaFree(newRigidBodyTorque);
	if(newRigidBodyRadius)cudaFree(newRigidBodyRadius);
	if(newParticleForce)cudaFree(newParticleForce);
	if(newParticleTorque)cudaFree(newParticleTorque);
	if(newParticlePosition)cudaFree(newParticlePosition);
	if(newParticleIndex)cudaFree(newParticleIndex);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::initBunny(glm::vec3 pos, glm::vec3 vel)
{
	std::string line;
	std::ifstream myfile ("Data/OBJparticles/bunny/bunny_1_5.txt");
	if (myfile.is_open())
	{
		bool initializedNow = false;
		std::getline (myfile, line);
		std::istringstream in(line);
		int particles;
		in >> particles;
		int start = m_numParticles;
		if (!m_numParticles)
		{
			std::cout << "System has " << m_numParticles << " particles" << std::endl;
			_initialize(particles);
			initializedNow = true;
		}
		std::cout << "Bunny object has " << particles << " particles" << std::endl;
		firstBunnyIndex = numRigidBodies; //numRigidBodies has not yet increased
		bunnyParticles = particles;
//		objectsThrown++;
		//reallocate host memory to fit new data
		float *h_newPos = new float[(m_numParticles + particles) * 4]; //add #(particles) new particles to our system
		float *h_newVel = new float[(m_numParticles + particles) * 4];

		float *dPos;
		if (!initializedNow)
		{

			dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
			mapRelativePositionIndependentParticlesWrapper(
							(float4 *)dPos, //particle positions
							(float4 *)relativePos, //relative particle positions
							rbIndices, //rigid body indices
							start,
							numThreads);
			cudaMemcpy(h_newPos, relativePos, sizeof(float) * 4 * start, cudaMemcpyDeviceToHost); //copy old positions array to newly allocated space
			unmapGLBufferObject(m_cuda_posvbo_resource);

			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			cudaMemcpy(h_newVel, m_dVel, sizeof(float) * 4 * start, cudaMemcpyDeviceToHost); //copy old velocities array to newly allocated space

			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			free(m_hPos); //free old positions array
			m_hPos = h_newPos; //change pointer to new positions array
			free(m_hVel); //free old velocities array
			m_hVel = h_newVel; //change pointer to new velocities array
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}

		glm::mat3 inertiaTensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
		float maxDistance = -100000000;
//		std::cout << "Altering relative positions in range: " << start << "-" << start + particles << std::endl;
		glm::vec3 cm(0, 0, 0);
		for (int i = start; i < start + particles; i++)
		{
			std::getline (myfile, line);
			std::istringstream in(line);
			float x, y, z;
			in >> x >> y >> z;

			m_hPos[4 * i] = x;
			m_hPos[4 * i + 1] = y;
			m_hPos[4 * i + 2] = z;
			m_hPos[4 * i + 3] = 0.f;

			m_hVel[4 * i] = 0.f;
			m_hVel[4 * i + 1] = 0.f;
			m_hVel[4 * i + 2] = 0.f;
			m_hVel[4 * i + 3] = 0.f;

			cm.x += x;
			cm.y += y;
			cm.z += z;


		}

		cm = cm / (float) particles;
		std::cout << "Bunny center of mass: (" << cm.x << ", " << cm.y << ", " << cm.z << ")" << std::endl;
		glm::vec3 test(0, 0, 0);
		for (int i = start; i < start + particles; i++)
		{
			m_hPos[4 * i] -= cm.x;
			m_hPos[4 * i + 1] -= cm.y;
			m_hPos[4 * i + 2] -= cm.z;
			m_hPos[4 * i + 3] = 0.f;

			float x = m_hPos[4 * i];
			float y = m_hPos[4 * i + 1];
			float z = m_hPos[4 * i + 2];

			test.x += x;
			test.y += y;
			test.z += z;
			inertiaTensor[0][0] += y * y + z * z; //y*y + z*z
			inertiaTensor[0][1] -= x * y; //x*y
			inertiaTensor[0][2] -= x * z; //x*z

			inertiaTensor[1][0] -= x * y; //x*y
			inertiaTensor[1][1] += x * x + z * z; //x*x + z*z
			inertiaTensor[1][2] -= y * z; //y*z

			inertiaTensor[2][0] -= x * z; //x*z
			inertiaTensor[2][1] -= y * z; //y*z
			inertiaTensor[2][2] += x * x + y * y; //x*x + y*y

			//find max distance from CM so we can use it as radius
			maxDistance = maxDistance > x ? maxDistance : x;
			maxDistance = maxDistance > y ? maxDistance : y;
			maxDistance = maxDistance > z ? maxDistance : z;
		}
		if (!initializedNow)
			m_numParticles += particles;
		cm = test / (float)particles;
		std::cout << "New center of mass: (" << cm.x << ", " << cm.y << ", " << cm.z << ")" << std::endl;
		std::cout << "Bunny inertia tensor: " << std::endl;
		for (int row = 0; row < 3; row++)
		{
			for (int col = 0; col < 3; col++)
				std::cout << inertiaTensor[row][col] << " ";
			std::cout << std::endl;
		}
		inertiaTensor = glm::inverse(inertiaTensor);
		std::cout << "Bunny inverse inertia tensor: " << std::endl;
		for (int row = 0; row < 3; row++)
		{
			for (int col = 0; col < 3; col++)
				std::cout << inertiaTensor[row][col] << " ";
			std::cout << std::endl;
		}
		cudaMalloc((void**)&bunnyRelativePositions, particles * sizeof(float) * 4);
		cudaMemcpy(bunnyRelativePositions, &m_hPos[4*start], particles * sizeof(float) * 4, cudaMemcpyHostToDevice);

		//new per particle values
		float *newRelativePos;
		checkCudaErrors(cudaMalloc((void**)&newRelativePos, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newRelativePos, &m_hPos[4 * start], 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newParticleVelocity;
		checkCudaErrors(cudaMalloc((void**)&newParticleVelocity, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticleVelocity, &m_hVel[4 * start], 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newPerParticleValues = new float[4 * particles];
		memset(newPerParticleValues, 0, 4 * sizeof(float) * particles);
		float *newParticleForce;
		checkCudaErrors(cudaMalloc((void**)&newParticleForce, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticleForce, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newParticleTorque;
		checkCudaErrors(cudaMalloc((void**)&newParticleTorque, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticleTorque, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newParticlePosition;
		checkCudaErrors(cudaMalloc((void**)&newParticlePosition, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticlePosition, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		int *newIndexArray = new int[particles];
		for(int i = 0; i < particles; i++)
			newIndexArray[i] = numRigidBodies; //numRigidBodies has not yet increased
		int *newParticleIndex;
		checkCudaErrors(cudaMalloc((void**)&newParticleIndex, sizeof(int) * particles));
		checkCudaErrors(cudaMemcpy(newParticleIndex, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

		int *newCountARCollions;
		checkCudaErrors(cudaMalloc((void**)&newCountARCollions, sizeof(int) * particles));
		memset(newIndexArray, 0, sizeof(int) * particles); //reset values to zero
		checkCudaErrors(cudaMemcpy(newCountARCollions, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

		delete newIndexArray;
		delete newPerParticleValues;

		glm::mat3 *newInverseInertia;
		checkCudaErrors(cudaMalloc((void**)&newInverseInertia, sizeof(glm::mat3)));
		checkCudaErrors(cudaMemcpy(newInverseInertia, &inertiaTensor, sizeof(glm::mat3), cudaMemcpyHostToDevice));

		glm::vec3 *newRigidBodyAngularAcceleration;
		glm::vec3 newwdot(0.f, 0.f, 0.f);
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularAcceleration, sizeof(glm::vec3)));
		checkCudaErrors(cudaMemcpy(newRigidBodyAngularAcceleration, &newwdot, sizeof(glm::vec3), cudaMemcpyHostToDevice));

		glm::quat *newRigidBodyQuaternion;
		glm::quat newQ(1.f, 0.f, 0.f, 0.f);
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyQuaternion, sizeof(glm::quat)));
		checkCudaErrors(cudaMemcpy(newRigidBodyQuaternion, &newQ, sizeof(glm::quat), cudaMemcpyHostToDevice));


		float *newRigidBodyCM;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyCM, 4 * sizeof(float)));
		float4 newCM = make_float4(pos.x, pos.y, pos.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
		float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyAngularVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
		float4 newAngVel = make_float4(0, 0, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyAngularVelocity, &newAngVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyForce;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyForce, 4 * sizeof(float)));
		float4 newForce = make_float4(0, 0, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyForce, &newForce, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyAngularMomentum;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularMomentum, 4 * sizeof(float)));
		float4 newL = make_float4(0, 0.f, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyAngularMomentum, &newL, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyLinearMomentum;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyLinearMomentum, 4 * sizeof(float)));
		float4 newP = make_float4(vel.x, vel.y, vel.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyLinearMomentum, &newP, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyTorque;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyTorque, 4 * sizeof(float)));
		float4 newTorque = make_float4(0, 0, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyTorque, &newTorque, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyRadius;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyRadius, sizeof(float)));
		float newRadius = maxDistance;
		checkCudaErrors(cudaMemcpy(newRigidBodyRadius, &newRadius, sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyMass;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyMass, sizeof(float)));
		float newMass = 1.f; //ISSUE: change this to simulate rigid bodies of different mass - after changing it also change inertia
		checkCudaErrors(cudaMemcpy(newRigidBodyMass, &newMass, sizeof(float), cudaMemcpyHostToDevice));

		addRigidBody(start,
				particles,
				newRelativePos, //new relative position - 4 * particlesAdded
				newParticleVelocity, //new particle velocity - 4 * particlesAdded
				newInverseInertia, //new inverse inertia tensor - 1
				newRigidBodyCM, //new rigid body center of mass - 4
				newRigidBodyVelocity, //new rigid body velocity - 4
				newRigidBodyAngularVelocity, //new rigid body angular velocity - 4
				newRigidBodyAngularAcceleration, //1
				newRigidBodyQuaternion, //new rigid body quaternion - 4
				newRigidBodyForce, //new rigid body force - 4
				newRigidBodyMass, //1
				newRigidBodyAngularMomentum, //4
				newRigidBodyLinearMomentum, //4
				newRigidBodyTorque, //4
				newRigidBodyRadius, //1
				newParticleForce, //4 * particlesAdded
				newParticleTorque, //4 * particlesAdded
				newParticlePosition, //4 * particlesAdded
				newCountARCollions, //particlesAdded
				newParticleIndex, //particlesAdded
				true);

		myfile.close();
//		std::cout << "Bunny initial inertia matrix:" << std::endl;
//		for (int row = 0; row < 3; row++)
//		{
//			for (int col = 0; col < 3; col++)
//				std::cout << inertiaTensor[row][col] << " ";
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//		inertiaTensor = inverse(inertiaTensor);
//		std::cout << "Bunny initial inverse inertia matrix:" << std::endl;
//		for (int row = 0; row < 3; row++)
//		{
//			for (int col = 0; col < 3; col++)
//				std::cout << inertiaTensor[row][col] << " ";
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//		std::cout << "Number of particles after newest addition: " << m_numParticles << std::endl;
//		//reallocate client (GPU) memory to fit new data
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//		unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
//		unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
//		unsigned int memSize = sizeof(float) * 4 * m_numParticles;
//
//		glGenVertexArrays(1, &m_virtualVAO);
//		glBindVertexArray(m_virtualVAO);
//
//		glGenBuffers(1, &m_posVbo);
//		glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
//		glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
//		glEnableVertexAttribArray(0);
//		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);
//
//		glGenBuffers(1, &m_colorVBO);
//		glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
//		glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
//		glEnableVertexAttribArray(1);
//		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);
//
//		glBindVertexArray(0);
//		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
//		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		//re-allocate memory to fit new data
//
//		cudaMalloc((void**)&bunnyRelativePositions, particles * sizeof(float) * 4);
//		cudaMemcpy(bunnyRelativePositions, &m_hPos[4*start], particles * sizeof(float) * 4, cudaMemcpyHostToDevice);
//
//		reAllocateMemory(&relativePos, 4 * m_numParticles, m_hPos, 4 * m_numParticles, 0); //create new relative-actual particle position array
//
//		reAllocateMemory(&m_dVel, 4 * m_numParticles, m_hVel, 4 * (m_numParticles - start), 4 * start); //new particle velocity array
//
//
//		firstBunnyIndex = numRigidBodies++; //increase number of rigid bodies
//		std::cout << "Number of rigid bodies after newest addition: " << numRigidBodies << std::endl;
//
//		float4 newValue = make_float4(pos.x, pos.y, pos.z, 0);
//		reAllocateMemory(&rbPositions, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body center of mass array
//
//		newValue = make_float4(vel.x, vel.y, vel.z, 0);
//		reAllocateMemory(&rbVelocities, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1));//new rigid body velocity array
//
//		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbAngularVelocity, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body angular velocity array
//
//		glm::vec3 newAngularAcceleration(0, 0, 0);
//		reAllocateMemory(&rbAngularAcceleration, numRigidBodies, &newAngularAcceleration, 1, numRigidBodies - 1); //new rigid body angular velocity array
//		reAllocateMemory(&rbInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
//		reAllocateMemory(&rbCurrentInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
//
//		glm::quat newQuatValue(1, 0, 0, 0);
//		reAllocateMemory(&rbQuaternion, numRigidBodies, &newQuatValue, 1, (numRigidBodies - 1)); //new rigid body quaternion array
//
//
//		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbForces, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//
//		//ISSUE: rigid bodies have the same mass as particles
//		float newMass = 1.f;// / 15.f;//(float)(m_numParticles - start); //all rigid bodies have a mass of 1
//		reAllocateMemory(&rbMass, numRigidBodies, &newMass, 1, (numRigidBodies - 1)); //new rigid body mass array
//		newValue = make_float4(0.0, 0.0, 0.0, 0);
//		reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//		newValue = make_float4(vel.x , vel.y, vel.z, 0) / newMass;
////		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbTorque, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body torque array - possibly not needed
//
//		float newRadius = m_params.particleRadius*2.0f*10;
//		reAllocateMemory(&rbRadii, numRigidBodies, &newRadius, 1, (numRigidBodies - 1)); //new rigid body radius array
//
//
//
//		int *newparticlesPerObjectThrown = new int[objectsThrown];
//		memcpy(newparticlesPerObjectThrown, particlesPerObjectThrown, sizeof(int) * (objectsThrown - 1));
//		if (particlesPerObjectThrown)
//			delete particlesPerObjectThrown;
//		particlesPerObjectThrown = newparticlesPerObjectThrown;
//		int newParticles = m_numParticles - start;
//		memcpy(&particlesPerObjectThrown[(objectsThrown - 1)], &newParticles, sizeof(int));
//
//		bool *newIsRigidBody = new bool[objectsThrown];
//		memcpy(newIsRigidBody, isRigidBody, sizeof(bool) * (objectsThrown - 1));
//		if (isRigidBody)
//			delete isRigidBody;
//		isRigidBody = newIsRigidBody;
//		bool newObject = true;
//		memcpy(&isRigidBody[(objectsThrown - 1)], &newObject, sizeof(bool));
//
//		int *indices = new int[(m_numParticles - start)];
//		for (int i = 0; i < (m_numParticles - start); i++)
//			indices[i] = numRigidBodies - 1; //new rigid body index
//
//		reAllocateMemory(&rbIndices, m_numParticles, indices, (m_numParticles - start), start); //new rigid body index array
//		delete indices;
////		indices = new int[(m_numParticles)];
////		cudaMemcpy(indices, rbIndices, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost);
////		for (int k = 0; k < m_numParticles; k++)
////		{
////			if(k < start && indices[k] != -1)
////				std::cout << "Wrong independent index @: " << k << std::endl;
////			else if(k > start && indices[k] == -1)
////				std::cout << "Wrong rigid body index @: " << k << std::endl;
////		}
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		float *newParticleValue = new float[4 * particles]; //all zeros (I hope)
//		memset(newParticleValue, 0, 4 * particles * sizeof(float));
//		reAllocateMemory(&pForce, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//		reAllocateMemory(&pPositions, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//		reAllocateMemory(&pTorque, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//		delete newParticleValue;
//
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
//
//		mapActualPositionRigidBodyParticlesWrapper(
//				(float4 *)dPos, //particle positions
//				(float4 *)relativePos, //relative particle positions
//				(float4 *)rbPositions, //rigid body center of mass
//				rbIndices, //rigid body indices
//				m_numParticles,
//				numThreads);
//
//		mapActualPositionIndependentParticlesWrapper(
//				(float4 *)dPos, //particle positions
//				(float4 *)relativePos, //relative particle positions
//				rbIndices, //rigid body indices
//				m_numParticles,
//				numThreads);
//		unmapGLBufferObject(m_cuda_posvbo_resource);
//
//
//
//		reallocGridAuxiliaries();
//
//		//number of virtual particles has changed! re-initialize SoA
//		initializeVirtualSoA(); //initialize SoA variables for virtual particles
//		myfile.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;
}

void ParticleSystem::addBunny(glm::vec3 pos, glm::vec3 vel)
{
	if (firstBunnyIndex == -1)
	{
		initBunny(pos, vel);
		return;
	}

	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	mapRelativePositionIndependentParticlesWrapper(
			(float4 *)dPos, //particle positions
			(float4 *)relativePos, //relative particle positions
			rbIndices, //rigid body indices
			m_numParticles,
			numThreads);
	unmapGLBufferObject(m_cuda_posvbo_resource);

	int start = m_numParticles;
	m_numParticles += bunnyParticles;
	int particles = bunnyParticles;
	std::cout << "Number of particles after newest addition: " << m_numParticles << std::endl;
	//reallocate client (GPU) memory to fit new data
	//new per particle values
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *newRelativePos;
	checkCudaErrors(cudaMalloc((void**)&newRelativePos, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newRelativePos, bunnyRelativePositions, 4 * sizeof(float) * particles, cudaMemcpyDeviceToDevice));

	float *newParticleVelocity;
	checkCudaErrors(cudaMalloc((void**)&newParticleVelocity, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticleVelocity, &m_hVel[4 * start], 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	float *newPerParticleValues = new float[4 * particles];
	memset(newPerParticleValues, 0, 4 * sizeof(float) * particles);
	float *newParticleForce;
	checkCudaErrors(cudaMalloc((void**)&newParticleForce, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticleForce, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	float *newParticleTorque;
	checkCudaErrors(cudaMalloc((void**)&newParticleTorque, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticleTorque, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	float *newParticlePosition;
	checkCudaErrors(cudaMalloc((void**)&newParticlePosition, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticlePosition, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	int *newIndexArray = new int[particles];
	for(int i = 0; i < particles; i++)
		newIndexArray[i] = numRigidBodies; //numRigidBodies has not yet increased
	int *newParticleIndex;
	checkCudaErrors(cudaMalloc((void**)&newParticleIndex, sizeof(int) * particles));
	checkCudaErrors(cudaMemcpy(newParticleIndex, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

	int *newCountARCollions;
	checkCudaErrors(cudaMalloc((void**)&newCountARCollions, sizeof(int) * particles));
	memset(newIndexArray, 0, sizeof(int) * particles); //reset values to zero
	checkCudaErrors(cudaMemcpy(newCountARCollions, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

	delete newIndexArray;
	delete newPerParticleValues;
	glm::mat3 *newInverseInertia;
	checkCudaErrors(cudaMalloc((void**)&newInverseInertia, sizeof(glm::mat3)));
	checkCudaErrors(cudaMemcpy(newInverseInertia, &rbInertia[firstBunnyIndex], sizeof(glm::mat3), cudaMemcpyDeviceToDevice));

	glm::vec3 *newRigidBodyAngularAcceleration;
	glm::vec3 newwdot(0.f, 0.f, 0.f);
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularAcceleration, sizeof(glm::vec3)));
	checkCudaErrors(cudaMemcpy(newRigidBodyAngularAcceleration, &newwdot, sizeof(glm::vec3), cudaMemcpyHostToDevice));

	glm::quat *newRigidBodyQuaternion;
	glm::quat newQ(1.f, 0.f, 0.f, 0.f);
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyQuaternion, sizeof(glm::quat)));
	checkCudaErrors(cudaMemcpy(newRigidBodyQuaternion, &newQ, sizeof(glm::quat), cudaMemcpyHostToDevice));


	float *newRigidBodyCM;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyCM, 4 * sizeof(float)));
	float4 newCM = make_float4(pos.x, pos.y, pos.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
	float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyAngularVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
	float4 newAngVel = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyAngularVelocity, &newAngVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyForce;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyForce, 4 * sizeof(float)));
	float4 newForce = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyForce, &newForce, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyAngularMomentum;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularMomentum, 4 * sizeof(float)));
	float4 newL = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyAngularMomentum, &newL, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyLinearMomentum;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyLinearMomentum, 4 * sizeof(float)));
	float4 newP = make_float4(vel.x, vel.y, vel.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyLinearMomentum, &newP, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyTorque;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyTorque, 4 * sizeof(float)));
	float4 newTorque = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyTorque, &newTorque, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyRadius;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyRadius, sizeof(float)));
	checkCudaErrors(cudaMemcpy(newRigidBodyRadius, &rbRadii[firstBunnyIndex], sizeof(float), cudaMemcpyDeviceToDevice));

	float *newRigidBodyMass;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyMass, sizeof(float)));
	float newMass = 1.f; //ISSUE: change this to simulate rigid bodies of different mass - after changing it also change inertia
	checkCudaErrors(cudaMemcpy(newRigidBodyMass, &newMass, sizeof(float), cudaMemcpyHostToDevice));

	addRigidBody(start,
			particles,
			newRelativePos, //new relative position - 4 * particlesAdded
			newParticleVelocity, //new particle velocity - 4 * particlesAdded
			newInverseInertia, //new inverse inertia tensor - 1
			newRigidBodyCM, //new rigid body center of mass - 4
			newRigidBodyVelocity, //new rigid body velocity - 4
			newRigidBodyAngularVelocity, //new rigid body angular velocity - 4
			newRigidBodyAngularAcceleration, //1
			newRigidBodyQuaternion, //new rigid body quaternion - 4
			newRigidBodyForce, //new rigid body force - 4
			newRigidBodyMass, //1
			newRigidBodyAngularMomentum, //4
			newRigidBodyLinearMomentum, //4
			newRigidBodyTorque, //4
			newRigidBodyRadius, //1
			newParticleForce, //4 * particlesAdded
			newParticleTorque, //4 * particlesAdded
			newParticlePosition, //4 * particlesAdded
			newCountARCollions, //particlesAdded
			newParticleIndex, //particlesAdded
			true);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//	unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
//	unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
//	unsigned int memSize = sizeof(float) * 4 * m_numParticles;
//
//	glGenVertexArrays(1, &m_virtualVAO);
//	glBindVertexArray(m_virtualVAO);
//
//	glGenBuffers(1, &m_posVbo);
//	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
//	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);
//
//	glGenBuffers(1, &m_colorVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
//	glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
//	glEnableVertexAttribArray(1);
//	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);
//
//	glBindVertexArray(0);
//	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
//	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//
//
//
//	objectsThrown++;
//	numRigidBodies++; //increase number of rigid bodies
//	std::cout << "Number of rigid bodies after newest addition: " << numRigidBodies << std::endl;
//
//	//re-allocate memory to fit new data
//	//these reallocations change - we no longer create a new bunny
//	//instead we use the previously loaded values for relative positions
////	reAllocateMemory(&relativePos, 4 * m_numParticles, m_hPos, 4 * m_numParticles, 0); //create new relative-actual particle position array
////	reAllocateMemory(&m_dVel, 4 * m_numParticles, m_hVel, 4 * (m_numParticles - start), 4 * start); //new particle velocity array
////	reAllocateMemory(&rbInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
////	reAllocateMemory(&rbCurrentInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
//
//	float *newArray;
//	checkCudaErrors(cudaMalloc((void**)&newArray, 4 * m_numParticles * sizeof(float)));
//	checkCudaErrors(cudaMemcpy(newArray, relativePos, 4 * start * sizeof(float), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaMemcpy(&newArray[4 * start], bunnyRelativePositions, 4 * bunnyParticles * sizeof(float), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaFree(relativePos));
//	relativePos = newArray;
//
//	checkCudaErrors(cudaMalloc((void**)&newArray, 4 * m_numParticles * sizeof(float)));
//	checkCudaErrors(cudaMemset(newArray, 0, 4 * m_numParticles * sizeof(float)));
//	checkCudaErrors(cudaMemcpy(newArray, m_dVel, 4 * start * sizeof(float), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaFree(m_dVel));
//	m_dVel = newArray;
//
//	glm::mat3 *newInertiaTensor;
//	checkCudaErrors(cudaMalloc((void**)&newInertiaTensor, numRigidBodies * sizeof(glm::mat3)));
//	checkCudaErrors(cudaMemcpy(newInertiaTensor, rbInertia, (numRigidBodies - 1) * sizeof(glm::mat3), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaMemcpy(&newInertiaTensor[numRigidBodies - 1], &rbInertia[firstBunnyIndex], sizeof(glm::mat3), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaFree(rbInertia));
//	rbInertia = newInertiaTensor;
//
//	checkCudaErrors(cudaMalloc((void**)&newInertiaTensor, numRigidBodies * sizeof(glm::mat3)));
//	checkCudaErrors(cudaMemcpy(newInertiaTensor, rbCurrentInertia, (numRigidBodies - 1) * sizeof(glm::mat3), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaMemcpy(&newInertiaTensor[numRigidBodies - 1], &rbInertia[firstBunnyIndex], sizeof(glm::mat3), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaFree(rbCurrentInertia));
//	rbCurrentInertia = newInertiaTensor;
//
//
//
//	float4 newValue = make_float4(pos.x, pos.y, pos.z, 0);
//	reAllocateMemory(&rbPositions, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body center of mass array
//
//	newValue = make_float4(vel.x, vel.y, vel.z, 0);
//	reAllocateMemory(&rbVelocities, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1));//new rigid body velocity array
//
//	newValue = make_float4(0, 0, 0, 0);
//	reAllocateMemory(&rbAngularVelocity, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body angular velocity array
//	glm::vec3 newAngularAcceleration(0, 0, 0);
//	reAllocateMemory(&rbAngularAcceleration, numRigidBodies, &newAngularAcceleration, 1, numRigidBodies - 1); //new rigid body angular velocity array
//	glm::quat newQuatValue(1, 0, 0, 0);
//	reAllocateMemory(&rbQuaternion, numRigidBodies, &newQuatValue, 1, (numRigidBodies - 1)); //new rigid body quaternion array
//
//
//	newValue = make_float4(0, 0, 0, 0);
//	reAllocateMemory(&rbForces, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//
//	//ISSUE: rigid bodies have the same mass as particles
//	float newMass = 1.f;// / 15.f;//(float)(m_numParticles - start); //all rigid bodies have a mass of 1
//	reAllocateMemory(&rbMass, numRigidBodies, &newMass, 1, (numRigidBodies - 1)); //new rigid body mass array
//	newValue = make_float4(0, 0, 0, 0);
//	reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//	newValue = make_float4(vel.x , vel.y, vel.z, 0) / newMass;
//	reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//	newValue = make_float4(0, 0, 0, 0);
//	reAllocateMemory(&rbTorque, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body torque array - possibly not needed
//
//	float newRadius = m_params.particleRadius*2.0f*10;
//	reAllocateMemory(&rbRadii, numRigidBodies, &newRadius, 1, (numRigidBodies - 1)); //new rigid body radius array
//
//
//
//	int *newparticlesPerObjectThrown = new int[objectsThrown];
//	memcpy(newparticlesPerObjectThrown, particlesPerObjectThrown, sizeof(int) * (objectsThrown - 1));
//	if (particlesPerObjectThrown)
//		delete particlesPerObjectThrown;
//	particlesPerObjectThrown = newparticlesPerObjectThrown;
//	int newParticles = m_numParticles - start;
//	memcpy(&particlesPerObjectThrown[(objectsThrown - 1)], &newParticles, sizeof(int));
//
//	bool *newIsRigidBody = new bool[objectsThrown];
//	memcpy(newIsRigidBody, isRigidBody, sizeof(bool) * (objectsThrown - 1));
//	if (isRigidBody)
//		delete isRigidBody;
//	isRigidBody = newIsRigidBody;
//	bool newObject = true;
//	memcpy(&isRigidBody[(objectsThrown - 1)], &newObject, sizeof(bool));
//
//	int *indices = new int[(m_numParticles - start)];
//	for (int i = 0; i < (m_numParticles - start); i++)
//		indices[i] = numRigidBodies - 1; //new rigid body index
//
//	reAllocateMemory(&rbIndices, m_numParticles, indices, (m_numParticles - start), start); //new rigid body index array
//	delete indices;
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	float *newParticleValue = new float[4 * particles]; //all zeros (I hope)
//	memset(newParticleValue, 0, 4 * particles * sizeof(float));
//	reAllocateMemory(&pForce, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//	reAllocateMemory(&pPositions, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//	reAllocateMemory(&pTorque, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//	delete newParticleValue;
//
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
//
//	mapActualPositionRigidBodyParticlesWrapper(
//			(float4 *)dPos, //particle positions
//			(float4 *)relativePos, //relative particle positions
//			(float4 *)rbPositions, //rigid body center of mass
//			rbIndices, //rigid body indices
//			m_numParticles,
//			numThreads);
//
//	mapActualPositionIndependentParticlesWrapper(
//			(float4 *)dPos, //particle positions
//			(float4 *)relativePos, //relative particle positions
//			rbIndices, //rigid body indices
//			m_numParticles,
//			numThreads);
//	unmapGLBufferObject(m_cuda_posvbo_resource);
//
//
//
//	reallocGridAuxiliaries();
//
//	//number of virtual particles has changed! re-initialize SoA
//	initializeVirtualSoA(); //initialize SoA variables for virtual particles
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
	allocateMemory(&pPositions, 4 * m_numParticles, newValue);
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
	reAllocateMemory(&pPositions, 4 * m_numParticles, newValue, 4 * (m_numParticles - start), 4 * start);
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
	reAllocateMemory(&rbAngularAcceleration, numRigidBodies, &newAngularAcceleration, 1, numRigidBodies - 1); //new rigid body angular velocity array
	reAllocateMemory(&rbInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
	reAllocateMemory(&rbCurrentInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array

	glm::quat newQuatValue(1, 0, 0, 0);
	reAllocateMemory(&rbQuaternion, numRigidBodies, &newQuatValue, 1, (numRigidBodies - 1)); //new rigid body quaternion array


	newValue = make_float4(0, 0, 0, 0);
	reAllocateMemory(&rbForces, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array

	//ISSUE: rigid bodies have the same mass as particles
	float newMass = 1.f / 15.f;//(float)(m_numParticles - start); //all rigid bodies have a mass of 1
	reAllocateMemory(&rbMass, numRigidBodies, &newMass, 1, (numRigidBodies - 1)); //new rigid body mass array
	newValue = make_float4(0, 0, 0, 0);
	reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(vel.x , vel.y, vel.z, 0) / newMass;
	reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(0, 0, 0, 0);
	reAllocateMemory(&rbTorque, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body torque array - possibly not needed

	float newRadius = m_params.particleRadius*2.0f*r;
	reAllocateMemory(&rbRadii, numRigidBodies, &newRadius, 1, (numRigidBodies - 1)); //new rigid body radius array



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
	reAllocateMemory(&pPositions, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
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
	reAllocateMemory(&rbAngularAcceleration, numRigidBodies, &newAngularAcceleration, 1, numRigidBodies - 1); //new rigid body angular velocity array
	reAllocateMemory(&rbInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
	reAllocateMemory(&rbCurrentInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array

	glm::quat newQuatValue(1, 0, 0, 0);
	reAllocateMemory(&rbQuaternion, numRigidBodies, &newQuatValue, 1, (numRigidBodies - 1)); //new rigid body quaternion array
	newValue = make_float4(0, 0, 0, 0);
	reAllocateMemory(&rbForces, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array

	//ISSUE: rigid bodies have the same mass as particles
	float newMass = 1.f / 15.f;//(float)(m_numParticles); //all rigid bodies have a mass of 1
	reAllocateMemory(&rbMass, numRigidBodies, &newMass, 1, (numRigidBodies - 1)); //new rigid body mass array

	newValue = make_float4(0, 0, 0, 0);
	reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(vel.x , vel.y, vel.z, 0) / newMass;
	reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
	newValue = make_float4(0, 0, 0, 0);

	reAllocateMemory(&rbTorque, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body torque array - possibly not needed

	float newRadius = m_params.particleRadius*2.0f*r;
	reAllocateMemory(&rbRadii, numRigidBodies, &newRadius, 1, (numRigidBodies - 1)); //new rigid body radius array



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
	reAllocateMemory(&pPositions, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
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

void ParticleSystem::initTeapot(glm::vec3 pos, glm::vec3 vel)
{
	std::string line;
	std::ifstream myfile ("Data/OBJparticles/teapot/teapot_1_5.txt");
	if (myfile.is_open())
	{
		bool initializedNow = false;
		std::getline (myfile, line);
		std::istringstream in(line);
		int particles;
		in >> particles;
		const int start = m_numParticles;
		if (!m_numParticles)
		{
			std::cout << "System has " << m_numParticles << " particles" << std::endl;
			_initialize(particles);
			initializedNow = true;
		}
		std::cout << "Teapot object has " << particles << " particles" << std::endl;
		teapotParticles = particles;
		firstTeapotIndex = numRigidBodies;

		//objectsThrown++;
		//reallocate host memory to fit new data
		float *h_newPos = new float[(start + particles) * 4]; //add #(particles) new particles to our system
		float *h_newVel = new float[(start + particles) * 4];

		float *dPos;
		if (!initializedNow)
		{

			dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
			mapRelativePositionIndependentParticlesWrapper(
							(float4 *)dPos, //particle positions
							(float4 *)relativePos, //relative particle positions
							rbIndices, //rigid body indices
							start,
							numThreads);
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
		}
		float maxDistance = -100000000;
		glm::mat3 inertiaTensor;
		std::cout << "Altering relative positions in range: " << start << "-" << start + particles << std::endl;
		for (int i = start; i < start + particles; i++)
		{
			std::getline (myfile, line);
			std::istringstream in(line);
			float x, y, z;
			in >> x >> y >> z;

			m_hPos[4 * i] = x;
			m_hPos[4 * i + 1] = y;
			m_hPos[4 * i + 2] = z;
			m_hPos[4 * i + 3] = 0.f;

			m_hVel[4 * i] = 0.f;
			m_hVel[4 * i + 1] = 0.f;
			m_hVel[4 * i + 2] = 0.f;
			m_hVel[4 * i + 3] = 0.f;

			inertiaTensor[0][0] += y *y + z * z; //y*y + z*z
			inertiaTensor[0][1] -= x * y; //x*y
			inertiaTensor[0][2] -= x * z; //x*z

			inertiaTensor[1][0] -= x * y; //x*y
			inertiaTensor[1][1] += x * x + z * z; //x*x + z*z
			inertiaTensor[1][2] -= y * z; //y*z

			inertiaTensor[2][0] -= x * z; //x*z
			inertiaTensor[2][1] -= y * z; //y*z
			inertiaTensor[2][2] += x * x + y * y; //x*x + y*y

//			inertiaTensor[0][0] += m_hPos[i * 4 + 1] * m_hPos[i * 4 + 1] + m_hPos[i * 4 + 2] * m_hPos[i * 4 + 2]; //y*y + z*z
//			inertiaTensor[0][1] -= m_hPos[i * 4 + 1] * m_hPos[i * 4]; //x*y
//			inertiaTensor[0][2] -= m_hPos[i * 4 + 2] * m_hPos[i * 4]; //x*z
//
//			inertiaTensor[1][0] -= m_hPos[i * 4 + 1] * m_hPos[i * 4]; //x*y
//			inertiaTensor[1][1] += m_hPos[i * 4] * m_hPos[i * 4] + m_hPos[i * 4 + 2] * m_hPos[i * 4 + 2]; //x*x + z*z
//			inertiaTensor[1][2] -= m_hPos[i * 4 + 2] * m_hPos[i * 4 + 1]; //y*z
//
//			inertiaTensor[2][0] -= m_hPos[i * 4 + 2] * m_hPos[i * 4]; //x*z
//			inertiaTensor[2][1] -= m_hPos[i * 4 + 2] * m_hPos[i * 4 + 1]; //y*z
//			inertiaTensor[2][2] += m_hPos[i * 4 + 1] * m_hPos[i * 4 + 1] + m_hPos[i * 4] * m_hPos[i * 4]; //x*x + y*y

			maxDistance = maxDistance > x ? maxDistance : x;
			maxDistance = maxDistance > y ? maxDistance : y;
			maxDistance = maxDistance > z ? maxDistance : z;
			//std::cout << "Position: (" << x << ", " << y << ", " << z <<")" << std::endl;
		}

		if (!initializedNow)
			m_numParticles += particles;
		inertiaTensor = glm::inverse(inertiaTensor);

		std::cout << "Number of particles after newest addition: " << m_numParticles << std::endl;
		//reallocate client (GPU) memory to fit new data

		cudaMalloc((void**)&teapotRelativePositions, particles * sizeof(float) * 4);
		cudaMemcpy(teapotRelativePositions, &m_hPos[4*start], particles * sizeof(float) * 4, cudaMemcpyHostToDevice);

		//new per particle values
		float *newRelativePos;
		checkCudaErrors(cudaMalloc((void**)&newRelativePos, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newRelativePos, &m_hPos[4 * start], 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newParticleVelocity;
		checkCudaErrors(cudaMalloc((void**)&newParticleVelocity, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticleVelocity, &m_hVel[4 * start], 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newPerParticleValues = new float[4 * particles];
		memset(newPerParticleValues, 0, 4 * sizeof(float) * particles);
		float *newParticleForce;
		checkCudaErrors(cudaMalloc((void**)&newParticleForce, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticleForce, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newParticleTorque;
		checkCudaErrors(cudaMalloc((void**)&newParticleTorque, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticleTorque, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		float *newParticlePosition;
		checkCudaErrors(cudaMalloc((void**)&newParticlePosition, 4 * sizeof(float) * particles));
		checkCudaErrors(cudaMemcpy(newParticlePosition, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

		int *newIndexArray = new int[particles];
		for(int i = 0; i < particles; i++)
			newIndexArray[i] = numRigidBodies; //numRigidBodies has not yet increased
		int *newParticleIndex;
		checkCudaErrors(cudaMalloc((void**)&newParticleIndex, sizeof(int) * particles));
		checkCudaErrors(cudaMemcpy(newParticleIndex, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

		int *newCountARCollions;
		checkCudaErrors(cudaMalloc((void**)&newCountARCollions, sizeof(int) * particles));
		memset(newIndexArray, 0, sizeof(int) * particles); //reset values to zero
		checkCudaErrors(cudaMemcpy(newCountARCollions, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

		delete newIndexArray;
		delete newPerParticleValues;

		glm::mat3 *newInverseInertia;
		checkCudaErrors(cudaMalloc((void**)&newInverseInertia, sizeof(glm::mat3)));
		checkCudaErrors(cudaMemcpy(newInverseInertia, &inertiaTensor, sizeof(glm::mat3), cudaMemcpyHostToDevice));

		glm::vec3 *newRigidBodyAngularAcceleration;
		glm::vec3 newwdot(0.f, 0.f, 0.f);
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularAcceleration, sizeof(glm::vec3)));
		checkCudaErrors(cudaMemcpy(newRigidBodyAngularAcceleration, &newwdot, sizeof(glm::vec3), cudaMemcpyHostToDevice));

		glm::quat *newRigidBodyQuaternion;
		glm::quat newQ(1.f, 0.f, 0.f, 0.f);
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyQuaternion, sizeof(glm::quat)));
		checkCudaErrors(cudaMemcpy(newRigidBodyQuaternion, &newQ, sizeof(glm::quat), cudaMemcpyHostToDevice));


		float *newRigidBodyCM;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyCM, 4 * sizeof(float)));
		float4 newCM = make_float4(pos.x, pos.y, pos.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
		float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyAngularVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
		float4 newAngVel = make_float4(0, 0, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyAngularVelocity, &newAngVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyForce;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyForce, 4 * sizeof(float)));
		float4 newForce = make_float4(0, 0, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyForce, &newForce, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyAngularMomentum;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularMomentum, 4 * sizeof(float)));
		float4 newL = make_float4(0, 0.f, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyAngularMomentum, &newL, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyLinearMomentum;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyLinearMomentum, 4 * sizeof(float)));
		float4 newP = make_float4(vel.x, vel.y, vel.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyLinearMomentum, &newP, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyTorque;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyTorque, 4 * sizeof(float)));
		float4 newTorque = make_float4(0, 0, 0, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyTorque, &newTorque, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyRadius;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyRadius, sizeof(float)));
		float newRadius = maxDistance;
		checkCudaErrors(cudaMemcpy(newRigidBodyRadius, &newRadius, sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyMass;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyMass, sizeof(float)));
		float newMass = 1.f; //ISSUE: change this to simulate rigid bodies of different mass - after changing it also change inertia
		checkCudaErrors(cudaMemcpy(newRigidBodyMass, &newMass, sizeof(float), cudaMemcpyHostToDevice));

		addRigidBody(start,
				particles,
				newRelativePos, //new relative position - 4 * particlesAdded
				newParticleVelocity, //new particle velocity - 4 * particlesAdded
				newInverseInertia, //new inverse inertia tensor - 1
				newRigidBodyCM, //new rigid body center of mass - 4
				newRigidBodyVelocity, //new rigid body velocity - 4
				newRigidBodyAngularVelocity, //new rigid body angular velocity - 4
				newRigidBodyAngularAcceleration, //1
				newRigidBodyQuaternion, //new rigid body quaternion - 4
				newRigidBodyForce, //new rigid body force - 4
				newRigidBodyMass, //1
				newRigidBodyAngularMomentum, //4
				newRigidBodyLinearMomentum, //4
				newRigidBodyTorque, //4
				newRigidBodyRadius, //1
				newParticleForce, //4 * particlesAdded
				newParticleTorque, //4 * particlesAdded
				newParticlePosition, //4 * particlesAdded
				newCountARCollions, //particlesAdded
				newParticleIndex, //particlesAdded
				true);

		myfile.close();
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//		unregisterGLBufferObject(m_cuda_posvbo_resource); //unregister old CUDA-GL interop buffer
//		unregisterGLBufferObject(m_cuda_colorvbo_resource); //unregister old CUDA-GL interop buffer
//		unsigned int memSize = sizeof(float) * 4 * m_numParticles;
//
//		glGenVertexArrays(1, &m_virtualVAO);
//		glBindVertexArray(m_virtualVAO);
//
//		glGenBuffers(1, &m_posVbo);
//		glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
//		glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
//		glEnableVertexAttribArray(0);
//		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);
//
//		glGenBuffers(1, &m_colorVBO);
//		glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
//		glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
//		glEnableVertexAttribArray(1);
//		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);
//
//		glBindVertexArray(0);
//		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
//		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		//re-allocate memory to fit new data
//
//		cudaMalloc((void**)&teapotRelativePositions, particles * sizeof(float) * 4);
//		cudaMemcpy(teapotRelativePositions, &m_hPos[4*start], particles * sizeof(float) * 4, cudaMemcpyHostToDevice);
//
//		reAllocateMemory(&relativePos, 4 * m_numParticles, m_hPos, 4 * m_numParticles, 0); //create new relative-actual particle position array
//
//		reAllocateMemory(&m_dVel, 4 * m_numParticles, m_hVel, 4 * (m_numParticles - start), 4 * start); //new particle velocity array
//
//
//		firstTeapotIndex = numRigidBodies++; //increase number of rigid bodies
//		std::cout << "Number of rigid bodies after newest addition: " << numRigidBodies << std::endl;
//
//		float4 newValue = make_float4(pos.x, pos.y, pos.z, 0);
//		reAllocateMemory(&rbPositions, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body center of mass array
//
//		newValue = make_float4(vel.x, vel.y, vel.z, 0);
//		reAllocateMemory(&rbVelocities, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1));//new rigid body velocity array
//
//		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbAngularVelocity, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body angular velocity array
//
//		glm::vec3 newAngularAcceleration(0, 0, 0);
//		reAllocateMemory(&rbAngularAcceleration, numRigidBodies, &newAngularAcceleration, 1, numRigidBodies - 1); //new rigid body angular velocity array
//		reAllocateMemory(&rbInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
//		reAllocateMemory(&rbCurrentInertia, numRigidBodies, &inertiaTensor, 1, (numRigidBodies - 1));//new rigid body inertia array
//
//		glm::quat newQuatValue(1, 0, 0, 0);
//		reAllocateMemory(&rbQuaternion, numRigidBodies, &newQuatValue, 1, (numRigidBodies - 1)); //new rigid body quaternion array
//
//
//		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbForces, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//
//		//ISSUE: rigid bodies have the same mass as particles
//		float newMass = 1.f;// / 15.f;//(float)(m_numParticles - start); //all rigid bodies have a mass of 1
//		reAllocateMemory(&rbMass, numRigidBodies, &newMass, 1, (numRigidBodies - 1)); //new rigid body mass array
//		newValue = make_float4(0.0, 0.0, 0.0, 0);
//		reAllocateMemory(&rbAngularMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//		newValue = make_float4(vel.x , vel.y, vel.z, 0) / newMass;
////		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbLinearMomentum, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body force array
//		newValue = make_float4(0, 0, 0, 0);
//		reAllocateMemory(&rbTorque, 4 * numRigidBodies, &newValue, 4, 4 * (numRigidBodies - 1)); //new rigid body torque array - possibly not needed
//
//		float newRadius = m_params.particleRadius*2.0f*10;
//		reAllocateMemory(&rbRadii, numRigidBodies, &newRadius, 1, (numRigidBodies - 1)); //new rigid body radius array
//
//
//
//		int *newparticlesPerObjectThrown = new int[objectsThrown];
//		memcpy(newparticlesPerObjectThrown, particlesPerObjectThrown, sizeof(int) * (objectsThrown - 1));
//		if (particlesPerObjectThrown)
//			delete particlesPerObjectThrown;
//		particlesPerObjectThrown = newparticlesPerObjectThrown;
//		int newParticles = m_numParticles - start;
//		memcpy(&particlesPerObjectThrown[(objectsThrown - 1)], &newParticles, sizeof(int));
//
//		bool *newIsRigidBody = new bool[objectsThrown];
//		memcpy(newIsRigidBody, isRigidBody, sizeof(bool) * (objectsThrown - 1));
//		if (isRigidBody)
//			delete isRigidBody;
//		isRigidBody = newIsRigidBody;
//		bool newObject = true;
//		memcpy(&isRigidBody[(objectsThrown - 1)], &newObject, sizeof(bool));
//
//		int *indices = new int[(m_numParticles - start)];
//		for (int i = 0; i < (m_numParticles - start); i++)
//			indices[i] = numRigidBodies - 1; //new rigid body index
//
//		reAllocateMemory(&rbIndices, m_numParticles, indices, (m_numParticles - start), start); //new rigid body index array
//		delete indices;
////		indices = new int[(m_numParticles)];
////		cudaMemcpy(indices, rbIndices, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost);
////		for (int k = 0; k < m_numParticles; k++)
////		{
////			if(k < start && indices[k] != -1)
////				std::cout << "Wrong independent index @: " << k << std::endl;
////			else if(k > start && indices[k] == -1)
////				std::cout << "Wrong rigid body index @: " << k << std::endl;
////		}
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		float *newParticleValue = new float[4 * particles]; //all zeros (I hope)
//		memset(newParticleValue, 0, 4 * particles * sizeof(float));
//		reAllocateMemory(&pForce, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//		reAllocateMemory(&pPositions, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//		reAllocateMemory(&pTorque, 4 * m_numParticles, newParticleValue, 4 * (m_numParticles - start), 4 * start);
//		delete newParticleValue;
//
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
//
//		mapActualPositionRigidBodyParticlesWrapper(
//				(float4 *)dPos, //particle positions
//				(float4 *)relativePos, //relative particle positions
//				(float4 *)rbPositions, //rigid body center of mass
//				rbIndices, //rigid body indices
//				m_numParticles,
//				numThreads);
//
//		mapActualPositionIndependentParticlesWrapper(
//				(float4 *)dPos, //particle positions
//				(float4 *)relativePos, //relative particle positions
//				rbIndices, //rigid body indices
//				m_numParticles,
//				numThreads);
//		unmapGLBufferObject(m_cuda_posvbo_resource);
//
//
//
//		reallocGridAuxiliaries();
//
//		//number of virtual particles has changed! re-initialize SoA
//		initializeVirtualSoA(); //initialize SoA variables for virtual particles
//		myfile.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;
}

void ParticleSystem::addTeapot(glm::vec3 pos, glm::vec3 vel)
{
	if (firstTeapotIndex == -1)
	{
		initTeapot(pos, vel);
		return;
	}
	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	mapRelativePositionIndependentParticlesWrapper(
			(float4 *)dPos, //particle positions
			(float4 *)relativePos, //relative particle positions
			rbIndices, //rigid body indices
			m_numParticles,
			numThreads);
	unmapGLBufferObject(m_cuda_posvbo_resource);
	int start = m_numParticles;
	m_numParticles += teapotParticles;
	int particles = teapotParticles;
	std::cout << "Number of particles after newest addition: " << m_numParticles << std::endl;
	//reallocate client (GPU) memory to fit new data
	float *newRelativePos;
	checkCudaErrors(cudaMalloc((void**)&newRelativePos, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newRelativePos, teapotRelativePositions, 4 * sizeof(float) * particles, cudaMemcpyDeviceToDevice));

	float *newParticleVelocity;
	checkCudaErrors(cudaMalloc((void**)&newParticleVelocity, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticleVelocity, &m_hVel[4 * start], 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	float *newPerParticleValues = new float[4 * particles];
	memset(newPerParticleValues, 0, 4 * sizeof(float) * particles);
	float *newParticleForce;
	checkCudaErrors(cudaMalloc((void**)&newParticleForce, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticleForce, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	float *newParticleTorque;
	checkCudaErrors(cudaMalloc((void**)&newParticleTorque, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticleTorque, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	float *newParticlePosition;
	checkCudaErrors(cudaMalloc((void**)&newParticlePosition, 4 * sizeof(float) * particles));
	checkCudaErrors(cudaMemcpy(newParticlePosition, newPerParticleValues, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

	int *newIndexArray = new int[particles];
	for(int i = 0; i < particles; i++)
		newIndexArray[i] = numRigidBodies; //numRigidBodies has not yet increased
	int *newParticleIndex;
	checkCudaErrors(cudaMalloc((void**)&newParticleIndex, sizeof(int) * particles));
	checkCudaErrors(cudaMemcpy(newParticleIndex, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

	int *newCountARCollions;
	checkCudaErrors(cudaMalloc((void**)&newCountARCollions, sizeof(int) * particles));
	memset(newIndexArray, 0, sizeof(int) * particles); //reset values to zero
	checkCudaErrors(cudaMemcpy(newCountARCollions, newIndexArray, sizeof(int) * particles, cudaMemcpyHostToDevice));

	delete newIndexArray;
	delete newPerParticleValues;
	glm::mat3 *newInverseInertia;
	checkCudaErrors(cudaMalloc((void**)&newInverseInertia, sizeof(glm::mat3)));
	checkCudaErrors(cudaMemcpy(newInverseInertia, &rbInertia[firstTeapotIndex], sizeof(glm::mat3), cudaMemcpyDeviceToDevice));

	glm::vec3 *newRigidBodyAngularAcceleration;
	glm::vec3 newwdot(0.f, 0.f, 0.f);
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularAcceleration, sizeof(glm::vec3)));
	checkCudaErrors(cudaMemcpy(newRigidBodyAngularAcceleration, &newwdot, sizeof(glm::vec3), cudaMemcpyHostToDevice));

	glm::quat *newRigidBodyQuaternion;
	glm::quat newQ(1.f, 0.f, 0.f, 0.f);
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyQuaternion, sizeof(glm::quat)));
	checkCudaErrors(cudaMemcpy(newRigidBodyQuaternion, &newQ, sizeof(glm::quat), cudaMemcpyHostToDevice));


	float *newRigidBodyCM;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyCM, 4 * sizeof(float)));
	float4 newCM = make_float4(pos.x, pos.y, pos.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
	float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyAngularVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
	float4 newAngVel = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyAngularVelocity, &newAngVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyForce;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyForce, 4 * sizeof(float)));
	float4 newForce = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyForce, &newForce, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyAngularMomentum;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularMomentum, 4 * sizeof(float)));
	float4 newL = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyAngularMomentum, &newL, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyLinearMomentum;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyLinearMomentum, 4 * sizeof(float)));
	float4 newP = make_float4(vel.x, vel.y, vel.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyLinearMomentum, &newP, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyTorque;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyTorque, 4 * sizeof(float)));
	float4 newTorque = make_float4(0, 0, 0, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyTorque, &newTorque, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyRadius;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyRadius, sizeof(float)));
	checkCudaErrors(cudaMemcpy(newRigidBodyRadius, &rbRadii[firstTeapotIndex], sizeof(float), cudaMemcpyDeviceToDevice));

	float *newRigidBodyMass;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyMass, sizeof(float)));
	float newMass = 1.f; //ISSUE: change this to simulate rigid bodies of different mass - after changing it also change inertia
	checkCudaErrors(cudaMemcpy(newRigidBodyMass, &newMass, sizeof(float), cudaMemcpyHostToDevice));

	addRigidBody(start,
			particles,
			newRelativePos, //new relative position - 4 * particlesAdded
			newParticleVelocity, //new particle velocity - 4 * particlesAdded
			newInverseInertia, //new inverse inertia tensor - 1
			newRigidBodyCM, //new rigid body center of mass - 4
			newRigidBodyVelocity, //new rigid body velocity - 4
			newRigidBodyAngularVelocity, //new rigid body angular velocity - 4
			newRigidBodyAngularAcceleration, //1
			newRigidBodyQuaternion, //new rigid body quaternion - 4
			newRigidBodyForce, //new rigid body force - 4
			newRigidBodyMass, //1
			newRigidBodyAngularMomentum, //4
			newRigidBodyLinearMomentum, //4
			newRigidBodyTorque, //4
			newRigidBodyRadius, //1
			newParticleForce, //4 * particlesAdded
			newParticleTorque, //4 * particlesAdded
			newParticlePosition, //4 * particlesAdded
			newCountARCollions, //particlesAdded
			newParticleIndex, //particlesAdded
			true);
}
