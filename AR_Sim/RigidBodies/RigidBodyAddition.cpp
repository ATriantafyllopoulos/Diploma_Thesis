#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
#include "BVHcreation.h"

#include <iostream>
#include <fstream>
#include <sstream>

template <typename T1, typename T2>
void allocateMemory(T1 **oldArray, uint size, T2 *initValue);
template <typename T1, typename T2>
void reAllocateMemory(T1 **oldArray, uint size, T2 *initValue, uint newElements, uint oldElements,
	cudaMemcpyKind TransferType = cudaMemcpyHostToDevice);

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

	glm::mat4 *newModelMatrix = new glm::mat4[numRigidBodies];
	memcpy(newModelMatrix, modelMatrix, sizeof(glm::mat4) * (numRigidBodies - 1));
	glm::mat4 newMatrix = glm::mat4(1.f);
	glm::quat newQuat;
	checkCudaErrors(cudaMemcpy(&newQuat, newRigidBodyQuaternion, sizeof(glm::quat), cudaMemcpyDeviceToHost));
	glm::mat3 rot = mat3_cast(newQuat);
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
			newMatrix[row][col] = rot[row][col];
	float4 newPos;
	checkCudaErrors(cudaMemcpy(&newPos, newRigidBodyCM, sizeof(float) * 4, cudaMemcpyDeviceToHost));
	newMatrix[3][0] = newPos.x;
	newMatrix[3][1] = newPos.y;
	newMatrix[3][2] = newPos.z;
	newModelMatrix[numRigidBodies - 1] = newMatrix;
	if (modelMatrix)
		delete modelMatrix;
	modelMatrix = newModelMatrix;

	glm::quat *newCumulativeQuaternion = new glm::quat[numRigidBodies];
	memcpy(newCumulativeQuaternion, cumulativeQuaternion, sizeof(glm::quat) * (numRigidBodies - 1));
	newCumulativeQuaternion[numRigidBodies - 1] = newQuat;
	if (cumulativeQuaternion)
		delete cumulativeQuaternion;
	cumulativeQuaternion = newCumulativeQuaternion;

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
	reAllocateMemory(&rbInertia, numRigidBodies, newInverseInertia, 1, (numRigidBodies - 1), cudaMemcpyDeviceToDevice);
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
	if (newRelativePos)cudaFree(newRelativePos);
	if (newParticleVelocity)cudaFree(newParticleVelocity);
	if (newInverseInertia)cudaFree(newInverseInertia);
	if (newRigidBodyCM)cudaFree(newRigidBodyCM);
	if (newRigidBodyVelocity)cudaFree(newRigidBodyVelocity);
	if (newRigidBodyAngularVelocity)cudaFree(newRigidBodyAngularVelocity);
	if (newRigidBodyAngularAcceleration)cudaFree(newRigidBodyAngularAcceleration);
	if (newRigidBodyQuaternion)cudaFree(newRigidBodyQuaternion);
	if (newRigidBodyForce)cudaFree(newRigidBodyForce);
	if (newRigidBodyMass)cudaFree(newRigidBodyMass);
	if (newRigidBodyAngularMomentum)cudaFree(newRigidBodyAngularMomentum);
	if (newRigidBodyLinearMomentum)cudaFree(newRigidBodyLinearMomentum);
	if (newRigidBodyTorque)cudaFree(newRigidBodyTorque);
	if (newRigidBodyRadius)cudaFree(newRigidBodyRadius);
	if (newParticleForce)cudaFree(newParticleForce);
	if (newParticleTorque)cudaFree(newParticleTorque);
	if (newParticlePosition)cudaFree(newParticlePosition);
	if (newParticleIndex)cudaFree(newParticleIndex);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::initBunny(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale)
{
	std::string line;
	std::string fileName("Data/OBJparticles/bunny/bunny_");
	if (abs(scale - 1.0 < 0.01))
		fileName += "1_0";
	else if (abs(scale - 1.5 < 0.01))
		fileName += "1_5";
	else if (abs(scale - 2.0 < 0.01))
		fileName += "2_0";
	else if (abs(scale - 2.5 < 0.01))
		fileName += "2_5";
	else 
		fileName += "1_5";
	fileName += ".txt";
	std::ifstream myfile(fileName);
	
	if (myfile.is_open())
	{
		bool initializedNow = false;
		std::getline(myfile, line);
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
			std::getline(myfile, line);
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

		cm = cm / (float)particles;
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
		cudaMemcpy(bunnyRelativePositions, &m_hPos[4 * start], particles * sizeof(float) * 4, cudaMemcpyHostToDevice);

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
		for (int i = 0; i < particles; i++)
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
		float4 newCM = make_float4(pos.x, pos.y, pos.z, 0.f);
		checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
		float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyAngularVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
		float4 newAngVel = make_float4(ang.x, ang.y, ang.z, 0);
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
	}
	else
		std::cout << "Unable to open file" << std::endl;
}

void ParticleSystem::addBunny(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale)
{
	if (firstBunnyIndex == -1)
	{
		initBunny(pos, vel, ang, scale);
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
	// initialization is not important, these values will change according to input velocity
	//checkCudaErrors(cudaMemcpy(newParticleVelocity, bunnyRelativePositions, 4 * sizeof(float) * particles, cudaMemcpyHostToDevice));

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
	for (int i = 0; i < particles; i++)
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
	float4 newCM = make_float4(pos.x, pos.y, pos.z, 0.f);
	checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
	float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyAngularVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
	float4 newAngVel = make_float4(ang.x, ang.y, ang.z, 0);
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
}

void ParticleSystem::initTeapot(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale)
{
	std::string line;
	std::string fileName("Data/OBJparticles/teapot/teapot_");
	if (abs(scale - 1.0 < 0.01))
		fileName += "1_0";
	else if (abs(scale - 1.5 < 0.01))
		fileName += "1_5";
	else if (abs(scale - 2.0 < 0.01))
		fileName += "2_0";
	else if (abs(scale - 2.5 < 0.01))
		fileName += "2_5";
	else
		fileName += "1_5";
	fileName += ".txt";
	std::ifstream myfile(fileName);

	if (myfile.is_open())
	{
		bool initializedNow = false;
		std::getline(myfile, line);
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

		glm::vec3 cm(0, 0, 0);
		
		for (int i = start; i < start + particles; i++)
		{
			std::getline(myfile, line);
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

		cm = cm / (float)particles;
		std::cout << "Obj particles model center of mass: (" << cm.x << ", " << cm.y << ", " << cm.z << ")" << std::endl;
		glm::vec3 test(0, 0, 0);
		for (int i = start; i < start + particles; i++)
		{
			m_hPos[4 * i] -= cm.x;
			m_hPos[4 * i + 1] -= cm.y + m_params.particleRadius;
			m_hPos[4 * i + 2] -= cm.z;
			m_hPos[4 * i + 3] = 0.f;

			float x = m_hPos[4 * i];
			float y = m_hPos[4 * i + 1];
			float z = m_hPos[4 * i + 2];

			test += glm::vec3(x, y, z);

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
		test /= (float)particles;
		std::cout << "Obj particles model corrected center of mass: (" << test.x << ", " << test.y << ", " << test.z << ")" << std::endl;
		if (!initializedNow)
			m_numParticles += particles;
		inertiaTensor = glm::inverse(inertiaTensor);
		std::cout << "Teapot inverse inertia tensor: " << std::endl;
		for (int row = 0; row < 3; row++)
		{
			for (int col = 0; col < 3; col++)
				std::cout << inertiaTensor[row][col] << " ";
			std::cout << std::endl;
		}
		std::cout << "Number of particles after newest addition: " << m_numParticles << std::endl;
		//reallocate client (GPU) memory to fit new data

		cudaMalloc((void**)&teapotRelativePositions, particles * sizeof(float) * 4);
		cudaMemcpy(teapotRelativePositions, &m_hPos[4 * start], particles * sizeof(float) * 4, cudaMemcpyHostToDevice);

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
		for (int i = 0; i < particles; i++)
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
		float4 newCM = make_float4(pos.x, pos.y, pos.z, 0.f);
		checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
		float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
		checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

		float *newRigidBodyAngularVelocity;
		checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
		float4 newAngVel = make_float4(ang.x, ang.y, ang.z, 0);
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
	}
	else
		std::cout << "Unable to open file" << std::endl;
}

void ParticleSystem::addTeapot(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale)
{
	if (firstTeapotIndex == -1)
	{
		initTeapot(pos, vel, ang, scale);
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
	for (int i = 0; i < particles; i++)
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
	float4 newCM = make_float4(pos.x, pos.y, pos.z, 0.f);
	checkCudaErrors(cudaMemcpy(newRigidBodyCM, &newCM, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyVelocity, 4 * sizeof(float)));
	float4 newVel = make_float4(vel.x, vel.y, vel.z, 0);
	checkCudaErrors(cudaMemcpy(newRigidBodyVelocity, &newVel, 4 * sizeof(float), cudaMemcpyHostToDevice));

	float *newRigidBodyAngularVelocity;
	checkCudaErrors(cudaMalloc((void**)&newRigidBodyAngularVelocity, 4 * sizeof(float)));
	float4 newAngVel = make_float4(ang.x, ang.y, ang.z, 0);
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