#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
#include "BVHcreation.h"

void ParticleSystem::updateRigidBodies(float deltaTime)
{
	assert(m_bInitialized);

	float *dPos, *dCol = NULL;

	if (m_bUseOpenGL)
	{
		dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
		dCol = (float *)mapGLBufferObject(&m_cuda_colorvbo_resource);
	}
	else
	{
		dPos = (float *)m_cudaPosVBO;
	}

	// update constants
	setParameters(&m_params);

	// integrate
	/*integrateSystem(
	dPos,
	m_dVel,
	deltaTime,
	minPos,
	maxPos,
	m_numParticles);*/
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	integrateSystemRigidBodies((float4 *)rbPositions, //rigid body center of mass
		(float4 *)rbVelocities, //velocity of rigid body
		(float4 *)rbForces, //total force applied to rigid body due to previous collisions
		(float4 *)rbAngularVelocity, //contains angular velocities for each rigid body
		rbQuaternion, //contains current quaternion for each rigid body
		(float4 *)rbTorque, //torque applied to rigid body due to previous collisions
		(float4 *)rbAngularMomentum, //cumulative angular momentum of the rigid body
		(float4 *)rbLinearMomentum, //cumulative linear momentum of the rigid body
		rbInertia, //original moment of inertia for each rigid body - 9 values per RB
		rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
		rbAngularAcceleration, //current angular acceleration due to misaligned angular momentum and velocity
		deltaTime, //dt
		rbRadii, //radius chosen for each rigid body sphere
		rbMass, //total mass of rigid body
		minPos, //smallest coordinate of scene's bounding box
		maxPos, //largest coordinate of scene's bounding box
		numRigidBodies, //number of rigid bodies
		m_params, //simulation parameters
		numThreads);

	flushAndPrintRigidBodyParameters();


	//	float *cmTest = new float[4 * numRigidBodies];
	//	checkCudaErrors(cudaMemcpy(cmTest, rbQuaternion, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	//	glm::mat3 *inertia = new glm::mat3[numRigidBodies];
	//	checkCudaErrors(cudaMemcpy(inertia, rbInertia, sizeof(glm::mat3) * numRigidBodies, cudaMemcpyDeviceToHost));
	//	for(int i = 0; i < numRigidBodies; i++)
	//	{
	//		glm::mat3 localInertia = inertia[i];
	//		std::cout << "Rigid body #" << i + 1 << " quaternion: (" << cmTest[4*i] << " " <<
	//				cmTest[4*i+1] << " " << cmTest[4*i+2] << " " << cmTest[4*i+3] << ")" << std::endl;
	//		glm::quat q(cmTest[4*i+3], cmTest[4*i], cmTest[4*i+1], cmTest[4*i+2]);
	//		glm::mat3 rot = mat3_cast(q);
	//		glm::mat3 currentInertia = rot * localInertia * transpose(rot);
	//		q = normalize(q);
	//		std::cout << "Rigid body #" << i + 1 << " normalized quaternion: (" << q.x << " " <<
	//
	//				q.y << " " << q.z << " " << q.w << ")" << std::endl;
	//		std::cout << "Rigid body #" << i + 1 << " rotation matrix:" << std::endl;
	//		for (int row = 0; row < 3; row++)
	//		{
	//			for (int col = 0; col < 3; col++)
	//				std::cout << rot[row][col] << " ";
	//			std::cout << std::endl;
	//		}
	//		std::cout << "Rigid body #" << i + 1 << " inertia matrix:" << std::endl;
	//		for (int row = 0; row < 3; row++)
	//		{
	//			for (int col = 0; col < 3; col++)
	//				std::cout << localInertia[row][col] << " ";
	//			std::cout << std::endl;
	//		}
	//	}
	//	delete inertia;
	//	delete cmTest;


	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	computeGlobalAttributesWrapper((float4 *)rbPositions, //rigid body's center of mass
		(float4 *)rbVelocities, //rigid body's velocity
		(float4 *)relativePos, //particle's relative position
		(float4 *)dPos, //particle's global position
		(float4 *)m_dVel, //particle's world velocity
		rbQuaternion, //contains current quaternion for each rigid body
		(float4 *)rbAngularVelocity, //contains angular velocities for each rigid body
		rbIndices, //index of associated rigid body
		m_numParticles, //number of particles
		numThreads);

	//	float *relTest = new float[4 * m_numParticles];
	//	checkCudaErrors(cudaMemcpy(relTest, dPos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost));
	//	std::cout << "Actual positions of particles belonging to latest rigid body" << std::endl;
	//	for(int i = m_numParticles - 1024; i < m_numParticles; i++)
	//	{
	//		std::cout << "Particle #" << i + 1 << " relative position (" << relTest[4*i] << " " <<
	//				relTest[4*i+1] << " " << relTest[4*i+2] << " " << relTest[4*i+3] << ")" << std::endl;
	//	}
	//	delete relTest;
	//	float *relTest = new float[4 * m_numParticles];
	//	checkCudaErrors(cudaMemcpy(relTest, relativePos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost));
	//	std::cout << "Relative positions of particles belonging to latest rigid body" << std::endl;
	//	for(int i = m_numParticles - 1024; i < m_numParticles; i++)
	//	{
	//		std::cout << "Particle #" << i + 1 << " relative position (" << relTest[4*i] << " " <<
	//				relTest[4*i+1] << " " << relTest[4*i+2] << " " << relTest[4*i+3] << ")" << std::endl;
	//	}
	//	delete relTest;


	//integrate independent virtual particles
	integrateSystem(
		dPos,
		m_dVel,
		deltaTime,
		minPos,
		maxPos,
		rbIndices,
		m_numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	static double totalRadixTime = 0;
	static double totalLeafTime = 0;
	static double totalInternalTime = 0;
	static double totalCollisionTime = 0;
	static double totalInitTime = 0;
	static double totalSortTime = 0;
	static int iterations = 0;
	clock_t start = clock();

	clock_t end = clock();
	totalInitTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	start = clock();
	checkCudaErrors(createMortonCodes((float4 *)dPos,
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices,
		m_numParticles,
		numThreads));
	end = clock();
	totalSortTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	start = clock();
	wrapperConstructRadixTreeSoA(
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		sortedMortonCodes,
		numThreads,
		m_numParticles);
	end = clock();
	totalRadixTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	start = clock();
	initializeRadiiWrapper(radii,
		m_params.particleRadius,
		m_numParticles,
		numThreads);
	wrapperConstructLeafNodesSoA(
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		(float4 *)dPos, //original positions
		m_params.particleRadius, //common radius parameter
		numThreads,
		m_numParticles
		);
	end = clock();
	totalLeafTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	start = clock();
	wrapperConstructInternalNodesSoA(
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		numThreads,
		m_numParticles);
	end = clock();
	totalInternalTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//	PreloadRigidBodyVariablesWrapper(
	//			(float4 *)rbForces, //Input: rigid body forces - one element per rigid body
	//			(float4 *)rbTorque, //Input: rigid body torques - one element per rigid body
	//			(float4 *)rbPositions, //Input: rigid body positions - one element per rigid body
	//			(float4 *)pForce, //Output: rigid body forces - one element per particle
	//			(float4 *)pTorque, //Output: rigid body torques - one element per particle
	//			(float4 *)pPositions, //Output: rigid body positions - one element per particle
	//			rbIndices, //Auxil.: indices of corresponding rigid bodies - one element per particle
	//			m_numParticles, //Auxil.: number of particles
	//			numThreads); //number of threads to use
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	start = clock();
	collideBVHSoARigidBodyWrapper((float4 *)dCol, //particle's color, only used for testing purposes
		(float4 *)pForce, //total force applied to rigid body - uniquely tied to each particle
		rbIndices, //index of the rigid body each particle belongs to
		(float4 *)pPositions, //rigid body center of mass - uniquely tied to each particle
		(float4 *)dPos, //unsorted particle positions
		(float4 *)relativePos, //particle's relative position
		(float4 *)pTorque,  //rigid body angular momentum - uniquely tied to each particle
		(float4 *)m_dVel, //particles original velocity, updated after all collisions are handled
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		m_numParticles, //number of virtual particles
		numThreads,
		m_params); //simulation parameters



	//
	//	collideBVHSoARigidBodyOnlyWrapper((float4 *)dCol, //particle's color, only used for testing purposes
	//		rbMass, //inverse mass of each rigid body
	//		rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
	//		(float4 *)rbPositions, //rigid body center of mass
	//		(float4 *)pForce, //total force applied to rigid body - uniquely tied to each particle
	//		rbIndices, //index of the rigid body each particle belongs to
	//		(float4 *)pPositions, //rigid body center of mass - uniquely tied to each particle
	//		(float4 *)dPos, //unsorted particle positions
	//		(float4 *)relativePos, //particle's relative position
	//		(float4 *)pTorque,  //rigid body angular momentum - uniquely tied to each particle
	//		(float4 *)m_dVel, //particles original velocity, updated after all collisions are handled
	//		isLeaf, //array containing a flag to indicate whether node is leaf
	//		leftIndices, //array containing indices of the left children of each node
	//		rightIndices, //array containing indices of the right children of each node
	//		minRange, //array containing minimum (sorted) leaf covered by each node
	//		maxRange, //array containing maximum (sorted) leaf covered by each node
	//		CMs, //array containing centers of mass for each leaf
	//		bounds, //array containing bounding volume for each node - currently templated Array of Structures
	//		sortedIndices, //array containing corresponding unsorted indices for each leaf
	//		radii, //radii of all nodes - currently the same for all particles
	//		m_numParticles, //number of virtual particles
	//		m_params, //simulation parameters
	//		numThreads);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	bool toExit = false;
	ReduceRigidBodyVariables(
		(float4 *)rbForces, //Output: rigid body forces - one element per rigid body
		(float4 *)rbTorque, //Output: rigid body torques - one element per rigid body
		(float4 *)rbPositions, //Output: rigid body positions - one element per rigid body
		(float4 *)pForce, //Input: rigid body forces - one element per particle
		(float4 *)pTorque, //Input: rigid body torques - one element per particle
		(float4 *)pPositions, //Input: rigid body positions - one element per particle
		particlesPerObjectThrown, //Auxil.: number of particles for each rigid body - one element per thrown objects
		isRigidBody, //Auxil.: flag indicating whether thrown object is a rigid body
		objectsThrown, //Auxil.: number of objects thrown - rigid bodies AND point sprites
		numRigidBodies, //Auxil.: number of rigid bodies
		numThreads,
		m_numParticles, //number of threads to use
		&toExit);
	//
	if (toExit)
	{
		std::cout << "Output of collision detection and handling routine is wrong." << std::endl;
		float *velTest = new float[4 * numRigidBodies];
		checkCudaErrors(cudaMemcpy(velTest, rbVelocities, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
		glm::mat3 *inertia = new glm::mat3[numRigidBodies];
		checkCudaErrors(cudaMemcpy(inertia, rbCurrentInertia, sizeof(glm::mat3) * numRigidBodies, cudaMemcpyDeviceToHost));
		float *mass = new float[numRigidBodies];
		checkCudaErrors(cudaMemcpy(mass, rbMass, sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
		//		for(int i = 0; i < numRigidBodies; i++)
		//		{
		//			glm::mat3 localInertia = inertia[i];
		//			std::cout << "Rigid body #" << i + 1 << " velocity: (" << velTest[4*i] << " " <<
		//					velTest[4*i+1] << " " << velTest[4*i+2] << " " << velTest[4*i+3] << ")" << std::endl;
		//
		//			std::cout << "Rigid body #" << i + 1 << " inertia matrix:" << std::endl;
		//			for (int row = 0; row < 3; row++)
		//			{
		//				for (int col = 0; col < 3; col++)
		//					std::cout << localInertia[row][col] << " ";
		//				std::cout << std::endl;
		//			}
		//			std::cout << "Mass: " << mass[i] << std::endl;
		//			std::cout << std::endl;
		//		}

		float4 *cpuVel = new float4[m_numParticles];
		cudaMemcpy(cpuVel, m_dVel, sizeof(float4) * m_numParticles, cudaMemcpyDeviceToHost);

		float4 *cpuRelative = new float4[m_numParticles];
		cudaMemcpy(cpuRelative, relativePos, sizeof(float4) * m_numParticles, cudaMemcpyDeviceToHost);

		float4 *cpuPos = new float4[m_numParticles];
		cudaMemcpy(cpuPos, dPos, sizeof(float4) * m_numParticles, cudaMemcpyDeviceToHost);
		int *cpuIndex = new int[m_numParticles];
		cudaMemcpy(cpuIndex, rbIndices, sizeof(int) * m_numParticles, cudaMemcpyDeviceToHost);
		int numCollisions = 0;
		for (int i = 0; i < m_numParticles; i++)
		{
			for (int j = 0; j < m_numParticles; j++)
			{
				if (i != j && cpuIndex[i] != cpuIndex[j] && length(make_float3(cpuPos[i] - cpuPos[j])) < 2 * m_params.particleRadius)
				{
					numCollisions++;
					std::cout << "Collision between particles #" << i << " and #" << j << std::endl;
					//compute collision force
					glm::mat3 currentInertia_A = inertia[cpuIndex[i]];
					glm::mat3 currentInertia_B = inertia[cpuIndex[j]];
					float mass_A = mass[cpuIndex[i]];
					float mass_B = mass[cpuIndex[j]];
					float radius_A = m_params.particleRadius;
					float radius_B = m_params.particleRadius;
					float4 contact_A = cpuPos[i];
					float4 contact_B = cpuPos[j];
					float3 vel_A = make_float3(cpuVel[i]);
					float3 vel_B = make_float3(cpuVel[j]);
					float4 relative_A = cpuRelative[i];
					float4 relative_B = cpuRelative[j];
					float3 relPos = make_float3(contact_B - contact_A);

					float dist = length(relPos); //distance between two centers
					float collideDist = radius_A + radius_B; //sum of radii
					float3 norm = relPos / dist; //norm points from A(queryParticle) to B(leafParticle)
					glm::vec3 n(norm.x, norm.y, norm.z);

					// relative velocity
					float3 relVel = vel_B - vel_A;

					float e = 0.99; //restitution
					float a = mass_A + mass_B;
					float3 CM_A = make_float3(contact_A - relative_A);
					glm::vec3 r_A(relative_A.x, relative_A.y, relative_A.z);
					float3 CM_B = make_float3(contact_B - relative_B);
					glm::vec3 r_B(relative_B.x, relative_B.y, relative_B.z);
					float b = glm::dot(glm::cross(currentInertia_A * glm::cross(r_A, n), r_A), n);
					float c = glm::dot(glm::cross(currentInertia_B * glm::cross(r_B, n), r_B), n);
					float d = a + b + c;
					float sign = -1.f;
					if (length(CM_A - make_float3(contact_B)) < length(CM_B - make_float3(contact_A)))
						sign = 1.f;
					float3 j = -1 * (1 + e) * dot(relVel, norm) * norm / d;
					float3 localForce = j;
					float3 localTorque = cross(make_float3(relative_A), j);
					std::cout << "CPU results for collision are: " << std::endl;
					std::cout << "Force: (" << localForce.x << " " << localForce.y << " " << localForce.z << ")" << std::endl;
					std::cout << "Torque: (" << localTorque.x << " " << localTorque.y << " " << localTorque.z << ")" << std::endl;


				}
			}
		}
		std::cout << "Total number of collisions: " << numCollisions / 2 << std::endl;
		delete cpuRelative;
		delete cpuVel;
		delete inertia;
		delete velTest;
		delete cpuIndex;
		delete cpuPos;
		if (toExit)exit(1);
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();

	totalCollisionTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last 1000 iterations..." << std::endl;
		std::cout << "Average time spent on initialization: " << totalInitTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on sorting: " << totalSortTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on radix tree creation: " << totalRadixTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on leaf nodes creation: " << totalLeafTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on internal nodes creation: " << totalInternalTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on collision detection and handling: " << totalCollisionTime / iterations << " (ms)" << std::endl;
	}

	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_colorvbo_resource);
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}

}

void ParticleSystem::staticUpdateRigidBodies(float deltaTime)
{
	assert(m_bInitialized);

	float *dPos, *dCol;

	if (m_bUseOpenGL)
	{
		dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
		dCol = (float *)mapGLBufferObject(&m_cuda_colorvbo_resource);
	}
	else
	{
		dPos = (float *)m_cudaPosVBO;
	}


	//arrays used by leaf nodes only
	static double totalRadixTime = 0;
	static double totalLeafTime = 0;
	static double totalInternalTime = 0;
	static double totalCollisionTime = 0;
	static double totalInitTime = 0;
	static double totalSortTime = 0;
	static int iterations = 0;
	clock_t start = clock();

	clock_t end = clock();
	totalInitTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	start = clock();
	//create Morton codes for the static particles and sort them
	checkCudaErrors(createMortonCodes((float4 *)staticPos,
		&r_mortonCodes,
		&r_indices,
		&r_sortedMortonCodes,
		&r_sortedIndices,
		numberOfRangeData,
		numThreads));
	end = clock();
	totalSortTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	start = clock();
	//construct the radix tree for the static particles
	wrapperConstructRadixTreeSoA(
		r_isLeaf, //array containing a flag to indicate whether node is leaf
		r_leftIndices, //array containing indices of the left children of each node
		r_rightIndices, //array containing indices of the right children of each node
		r_parentIndices, //array containing indices of the parent of each node
		r_minRange, //array containing minimum (sorted) leaf covered by each node
		r_maxRange, //array containing maximum (sorted) leaf covered by each node
		r_sortedMortonCodes,
		numThreads,
		numberOfRangeData);
	end = clock();
	totalRadixTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	start = clock();
	//construct the leaf nodes of the BVH
	wrapperConstructLeafNodesSoA(
		r_isLeaf, //array containing a flag to indicate whether node is leaf
		r_leftIndices, //array containing indices of the left children of each node
		r_rightIndices, //array containing indices of the right children of each node
		r_parentIndices, //array containing indices of the parent of each node
		r_minRange, //array containing minimum (sorted) leaf covered by each node
		r_maxRange, //array containing maximum (sorted) leaf covered by each node
		r_CMs, //array containing centers of mass for each leaf
		r_bounds, //array containing bounding volume for each node - currently templated Array of Structures
		r_sortedIndices, //array containing corresponding unsorted indices for each leaf
		r_radii, //radii of all nodes - currently the same for all particles
		(float4 *)staticPos, //original positions
		m_params.particleRadius, //common radius parameter
		numThreads,
		numberOfRangeData
		);
	end = clock();
	totalLeafTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	start = clock();
	//construct the internal nodes of the BVH
	wrapperConstructInternalNodesSoA(
		r_leftIndices, //array containing indices of the left children of each node
		r_rightIndices, //array containing indices of the right children of each node
		r_parentIndices, //array containing indices of the parent of each node
		r_bounds, //array containing bounding volume for each node - currently templated Array of Structures
		numThreads,
		numberOfRangeData);
	end = clock();
	totalInternalTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	start = clock();
	//detect collisions between virtual and static particles
	staticCollideBVHSoARigidBodyWrapper((float4 *)dCol, //virtual particle colors
		(float4 *)dPos, //virtual particle positions
		(float4 *)relativePos, //particle's relative position
		(float4 *)rbTorque,  //rigid body angular momentum
		(float4 *)rbForces, //total force applied to rigid body
		rbMass, //total mass of rigid body
		rbIndices, //index of the rigid body each particle belongs to
		(float4 *)rbPositions, //rigid body center of mass
		(float4 *)m_dVel, //particles original velocity, updated after all collisions are handled
		(float4 *)staticNorm, //normals computed for each real particle using its 8-neighborhood
		r_isLeaf, //array containing a flag to indicate whether node is leaf
		r_leftIndices, //array containing indices of the left children of each node
		r_rightIndices, //array containing indices of the right children of each node
		r_minRange, //array containing minimum (sorted) leaf covered by each node
		r_maxRange, //array containing maximum (sorted) leaf covered by each node
		r_CMs, //array containing centers of mass for each leaf
		r_bounds, //array containing bounding volume for each node - currently templated Array of Structures
		r_sortedIndices, //array containing corresponding unsorted indices for each leaf
		r_radii, //radii of all nodes - currently the same for all particles
		m_numParticles, //number of virtual particles
		numberOfRangeData, //number of static data
		numThreads, //number of threads
		m_params); //simulation parameters
	end = clock();
	totalCollisionTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last 1000 iterations regarding static particles..." << std::endl;
		std::cout << "Average time spent on initialization: " << totalInitTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on sorting: " << totalSortTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on radix tree creation: " << totalRadixTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on leaf nodes creation: " << totalLeafTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on internal nodes creation: " << totalInternalTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on collision detection and handling: " << totalCollisionTime / iterations << " (ms)" << std::endl;
	}
	float3 localMin, localMax;
	cudaMemcpy(&localMin, &(r_bounds[numberOfRangeData].min), sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(&localMax, &(r_bounds[numberOfRangeData].max), sizeof(float3), cudaMemcpyDeviceToHost);

	minPos.x = minPos.x < localMin.x ? minPos.x : localMin.x;
	minPos.y = minPos.y < localMin.y ? minPos.y : localMin.y;
	minPos.z = minPos.z < localMin.z ? minPos.z : localMin.z;

	maxPos.x = maxPos.x > localMax.x ? maxPos.x : localMax.x;
	maxPos.y = maxPos.y > localMax.y ? maxPos.y : localMax.y;
	maxPos.z = maxPos.z > localMax.z ? maxPos.z : localMax.z;
	//	std::cout << "Bounding box: " << std::endl;
	//	std::cout << "Min: (" << minPos.x << ", " << minPos.y << ", " << minPos.z << ")" << std::endl;
	//	std::cout << "Max: (" << maxPos.x << ", " << maxPos.y << ", " << maxPos.z << ")" << std::endl;
	//cudaFree everything

	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_colorvbo_resource);
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}
}

void ParticleSystem::Find_Rigid_Body_Collisions_BVH()
{
	checkCudaErrors(createMortonCodes((float4 *)dPos,
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices,
		m_numParticles,
		numThreads));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	wrapperConstructRadixTreeSoA(
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		sortedMortonCodes,
		numThreads,
		m_numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	initializeRadiiWrapper(radii,
		m_params.particleRadius,
		m_numParticles,
		numThreads);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	wrapperConstructLeafNodesSoA(
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		(float4 *)dPos, //original positions
		m_params.particleRadius, //common radius parameter
		numThreads,
		m_numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	wrapperConstructInternalNodesSoA(
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		parentIndices, //array containing indices of the parent of each node
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		numThreads,
		m_numParticles);


	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// cudaMemset is mandatory if cudaMalloc takes place once
	checkCudaErrors(cudaMemset(contactDistance, 0, sizeof(float) * m_numParticles));

	FindRigidBodyCollisionsBVHWrapper(
		(float4 *)dCol, // Input: particle's color, only used for testing purposes
		rbIndices, // Input: index of the rigid body each particle belongs to
		isLeaf, // Input: array containing a flag to indicate whether node is leaf
		leftIndices, // Input:  array containing indices of the left children of each node
		rightIndices, // Input: array containing indices of the right children of each node
		minRange, // Input: array containing minimum (sorted) leaf covered by each node
		maxRange, // Input: array containing maximum (sorted) leaf covered by each node
		CMs, // Input: array containing centers of mass for each leaf
		bounds, // Input: array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, // Input: array containing corresponding unsorted indices for each leaf
		radii, // Input: radii of all nodes - currently the same for all particles
		numThreads, // Input: number of threads to use
		m_numParticles, // Input: number of virtual particles
		m_params, // Input: simulation parameters
		contactDistance, // Output: distance between particles presenting largest penetration
		collidingParticleIndex, // Output: particle of most important contact
		collidingRigidBodyIndex); // Output: rigid body of most important contact

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

void ParticleSystem::Find_Augmented_Reality_Collisions_BVH()
{
	// calculate grid hash
	calcHash(
		m_dGridParticleHash,
		m_dGridParticleIndex,
		dPos,
		m_numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// sort particles based on hash
	sortParticles(&m_dGridParticleHash, &m_dGridParticleIndex, m_numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(
		rbIndices, //index of the rigid body each particle belongs to
		m_dCellStart,
		m_dCellEnd,
		m_dSortedPos,
		m_dSortedVel,
		m_dGridParticleHash,
		m_dGridParticleIndex,
		dPos,
		m_dVel,
		m_numParticles,
		m_numGridCells);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// calculate grid hash
	calcHash(
		staticGridParticleHash,
		staticGridParticleIndex,
		staticPos,
		numberOfRangeData);
	// sort particles based on hash
	sortParticles(&staticGridParticleHash, &staticGridParticleIndex, numberOfRangeData);
	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(rbIndices, //index of the rigid body each particle belongs to
		staticCellStart,
		staticCellEnd,
		staticSortedPos,
		staticSortedVel,
		staticGridParticleHash,
		staticGridParticleIndex,
		staticPos,
		staticVel,
		numberOfRangeData,
		m_numGridCells);

	// cudaMemset is mandatory if cudaMalloc takes place once
	checkCudaErrors(cudaMemset(contactDistance, 0, sizeof(float) * m_numParticles));

	FindAugmentedRealityCollisionsUniformGridWrapper(
		collidingParticleIndex, // index of particle of contact
		contactDistance, // penetration distance
		(float4 *)dCol, // particle color
		(float4 *)m_dSortedPos, // sorted positions
		(float4 *)staticSortedPos, // sorted augmented reality positions
		m_dGridParticleIndex, // sorted particle indices
		staticGridParticleIndex, // sorted scene particle indices
		staticCellStart,
		staticCellEnd,
		m_numParticles,
		numberOfRangeData,
		m_params,
		numThreads);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

void ParticleSystem::updateBVHExperimental(float deltaTime)
{
	assert(m_bInitialized);

	//float *dPos;
	//float *dCol;
	if (m_bUseOpenGL)
	{
		dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
		dCol = (float *)mapGLBufferObject(&m_cuda_colorvbo_resource);
	}
	else
	{
		dPos = (float *)m_cudaPosVBO;
	}

	// update constants
	setParameters(&m_params);

	// integrate system of rigid bodies
	Integrate_RB_System(deltaTime);

	// find and handle wall collisions
	Handle_Wall_Collisions();

	if (simulateAR)
	{
		// find collisions between rigid bodies and real scene
		Find_Augmented_Reality_Collisions_BVH();

		// handle collisions between rigid bodies and real scene
		Handle_Augmented_Reality_Collisions_Baraff_CPU();
	}

	// find collisions between rigid bodies
	Find_Rigid_Body_Collisions_BVH();

	// handle collisions between rigid bodies
	Handle_Rigid_Body_Collisions_Baraff_CPU();

	// note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_colorvbo_resource);
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}

}
