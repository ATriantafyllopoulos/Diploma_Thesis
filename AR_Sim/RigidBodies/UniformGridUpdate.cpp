#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
#include "BVHcreation.h"
void integrateRigidBodyCPU(float4 *CMs, //rigid body center of mass
	float4 *vel, //velocity of rigid body
	float4 *force, //force applied to rigid body due to previous collisions
	float4 *rbAngularVelocity, //contains angular velocities for each rigid body
	glm::quat *rbQuaternion, //contains current quaternion for each rigid body
	float4 *rbTorque, //torque applied to rigid body due to previous collisions
	float4 *rbAngularMomentum, //cumulative angular momentum of the rigid body
	float4 *rbLinearMomentum, //cumulative linear momentum of the rigid body
	glm::mat3 *rbInertia, //original moment of inertia for each rigid body - 9 values per RB
	glm::mat3 *rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
	glm::vec3 *rbAngularAcceleration, //current angular acceleration due to misaligned angular momentum and velocity
	float deltaTime, //dt
	float *rbRadii, //radius chosen for each rigid body sphere
	float *rbMass, //inverse of total mass of rigid body
	float3 minPos, //smallest coordinate of scene's bounding box
	float3 maxPos, //largest coordinate of scene's bounding box
	int numBodies, //number of rigid bodies
	SimParams params) //simulation parameters
{

	//	std::cout << "Wtf" << std::endl;
	//	std::cout << "Integrating rigid bodies on the CPU" << std::endl;
	float4 *CMs_CPU = new float4[numBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numBodies];  //velocity of rigid body
	float4 *force_CPU = new float4[numBodies];  //force applied to rigid body due to previous collisions
	float4 *rbAngularVelocity_CPU = new float4[numBodies];  //contains angular velocities for each rigid body
	glm::quat *rbQuaternion_CPU = new glm::quat[numBodies]; //contains current quaternion for each rigid body
	float4 *rbTorque_CPU = new float4[numBodies];  //torque applied to rigid body due to previous collisions
	float4 *rbAngularMomentum_CPU = new float4[numBodies];  //cumulative angular momentum of the rigid body
	float4 *rbLinearMomentum_CPU = new float4[numBodies];  //cumulative linear momentum of the rigid body
	glm::mat3 *rbInertia_CPU = new glm::mat3[numBodies];  //original moment of inertia for each rigid body - 9 values per RB
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numBodies];  //current moment of inertia for each rigid body - 9 values per RB
	glm::vec3 *rbAngularAcceleration_CPU = new glm::vec3[numBodies];  //current angular acceleration due to misaligned angular momentum and velocity
	float *rbRadii_CPU = new float[numBodies];  //radius chosen for each rigid body sphere
	float *rbMass_CPU = new float[numBodies];  //inverse of total mass of rigid body



	checkCudaErrors(cudaMemcpy(CMs_CPU, CMs, numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, vel, numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(force_CPU, force, numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbQuaternion_CPU, rbQuaternion, numBodies * sizeof(glm::quat), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbTorque_CPU, rbTorque, numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularMomentum_CPU, rbAngularMomentum, numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbLinearMomentum_CPU, rbLinearMomentum, numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbInertia_CPU, rbInertia, numBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularAcceleration_CPU, rbAngularAcceleration, numBodies * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbRadii_CPU, rbRadii, numBodies * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numBodies * sizeof(float), cudaMemcpyDeviceToHost));

	for (int index = 0; index < numBodies; index++)
	{
		float4 locLinearMomentum = rbLinearMomentum_CPU[index];
		locLinearMomentum += force_CPU[index];
		locLinearMomentum += make_float4(params.gravity, 0) * deltaTime;

		//		maxPos.x = maxPos.x + 0.001;
		//		maxPos.y = maxPos.y + 0.001;
		//		maxPos.z = maxPos.z + 0.001;
		//
		//		minPos.x = minPos.x - 0.001;
		//		minPos.y = minPos.y - 0.001;
		//		minPos.z = minPos.z - 0.001;

		float4 locPos = CMs_CPU[index];
		float locMass = rbMass_CPU[index];
		float sphereRadius = rbRadii_CPU[index];
		if (locPos.x > maxPos.x - sphereRadius)
		{
			locPos.x = maxPos.x - sphereRadius;
			locLinearMomentum.x *= params.boundaryDamping;
		}

		if (locPos.x < minPos.x + sphereRadius)
		{
			locPos.x = minPos.x + sphereRadius;
			locLinearMomentum.x *= params.boundaryDamping;
		}

		if (locPos.y > maxPos.y - sphereRadius && locLinearMomentum.y > 0)
		{
			locPos.y = maxPos.y - sphereRadius;
			locLinearMomentum.y *= params.boundaryDamping;
		}

		if (locPos.y < minPos.y + sphereRadius)
		{
			locPos.y = minPos.y + sphereRadius;
			locLinearMomentum.y *= params.boundaryDamping;
		}

		if (locPos.z > maxPos.z - sphereRadius)
		{
			locPos.z = maxPos.z - sphereRadius;
			locLinearMomentum.z *= params.boundaryDamping;
		}

		if (locPos.z < minPos.z + sphereRadius)
		{
			locPos.z = minPos.z + sphereRadius;
			locLinearMomentum.z *= params.boundaryDamping;
		}

		locLinearMomentum *= params.globalDamping;
		float4 locVel = locLinearMomentum / locMass;
		rbLinearMomentum_CPU[index] = locLinearMomentum;
		//locVel += make_float4(params.gravity, 0) * locMass * deltaTime;
		//locVel *= params.globalDamping;

		locPos += locVel * deltaTime;

		locPos.w = 0.f;
		locVel.w = 0.f;
		CMs_CPU[index] = locPos;
		vel_CPU[index] = locVel;
		force_CPU[index] = make_float4(0, 0, 0, 0); //reset force to zero

		//now consider rotational motion
		glm::mat3 inertia = rbInertia_CPU[index]; //each inertia matrix has 9 elements

		glm::quat quaternion = rbQuaternion_CPU[index];
		glm::mat3 currentInertia = rbCurrentInertia_CPU[index];
		float4 angularMomentum = rbAngularMomentum_CPU[index];
		float4 torque = rbTorque_CPU[index];
		angularMomentum += torque;
		glm::vec3 L(rbAngularMomentum_CPU[index].x,
			rbAngularMomentum_CPU[index].y,
			rbAngularMomentum_CPU[index].z);
		glm::vec3 Ldot(torque.x, torque.y, torque.z);
		Ldot *= -1.f;
		//		glm::vec3 Ldot(0, 0, 0);
		glm::vec3 currentMomentum = glm::vec3(angularMomentum.x, angularMomentum.y, angularMomentum.z);
		glm::vec3 newVelocity = currentInertia * currentMomentum;


		//
		glm::vec3 angularAcceleration = currentInertia * glm::cross(currentMomentum, newVelocity);
		angularMomentum *= 0.9999f;

		glm::vec3 normalizedVel = normalize(newVelocity);
		float theta = glm::length(newVelocity);
		if (theta > 0.00001)
		{
			quaternion.w = cos(theta / 2.f);
			quaternion.x = sin(theta / 2.f) * normalizedVel.x;
			quaternion.y = sin(theta / 2.f) * normalizedVel.y;
			quaternion.z = sin(theta / 2.f) * normalizedVel.z;
		}
		else
		{
			quaternion.w = 1.f;
			quaternion.x = 0.f;
			quaternion.y = 0.f;
			quaternion.z = 0.f;
		}
		//		quaternion = glm::rotate(quaternion, theta, glm::normalize(newVelocity));
		//	float angular_speed = glm::length(newVelocity);
		//	float rotation_angle = angular_speed*deltaTime;
		//	glm::vec3 rotationAxis = normalize(newVelocity);
		//	glm::quat dq(cos(rotation_angle / 2), sin(rotation_angle / 2) * rotationAxis.x, sin(rotation_angle / 2) * rotationAxis.y, sin(rotation_angle / 2) * rotationAxis.z);
		//	quaternion = glm::cross(dq, quaternion);
		quaternion = normalize(quaternion);
		//		if (myfileQuat.is_open())
		//		{
		//			myfileQuat << quaternion.w << " " << quaternion.x << " " << quaternion.y << " " << quaternion.z << '\n';
		//		}
		//		if (++iterations == 100000)
		//			exit(1);
		glm::mat3 rot = mat3_cast(quaternion);
		currentInertia = rot * inertia * transpose(rot);
		//		newVelocity -= angularAcceleration * deltaTime;

		rbAngularAcceleration_CPU[index] = angularAcceleration;
		rbCurrentInertia_CPU[index] = currentInertia;
		rbAngularMomentum_CPU[index] = angularMomentum;
		rbQuaternion_CPU[index] = quaternion;
		rbAngularVelocity_CPU[index] = make_float4(newVelocity.x, newVelocity.y, newVelocity.z, 0);
		rbTorque_CPU[index] = make_float4(0, 0, 0, 0); //reset torque to zero

	}
	checkCudaErrors(cudaMemcpy(CMs, CMs_CPU, numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vel, vel_CPU, numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(force, force_CPU, numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbQuaternion, rbQuaternion_CPU, numBodies * sizeof(glm::quat), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbTorque, rbTorque_CPU, numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularMomentum, rbAngularMomentum_CPU, numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbLinearMomentum, rbLinearMomentum_CPU, numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbInertia, rbInertia_CPU, numBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularAcceleration, rbAngularAcceleration_CPU, numBodies * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbRadii, rbRadii_CPU, numBodies * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete force_CPU;
	delete rbAngularVelocity_CPU;
	delete rbQuaternion_CPU;
	delete rbTorque_CPU;
	delete rbAngularMomentum_CPU;
	delete rbLinearMomentum_CPU;
	delete rbInertia_CPU;
	delete rbCurrentInertia_CPU;
	delete rbAngularAcceleration_CPU;
	delete rbRadii_CPU;
	delete rbMass_CPU;
}

// step the simulation
void
ParticleSystem::updateGrid(float deltaTime)
{
	assert(m_bInitialized);

	float *dPos;
	float *dCol;
	if (m_bUseOpenGL)
	{
		dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
		dCol = (float *)mapGLBufferObject(&m_cuda_colorvbo_resource);
	}
	else
	{
		dPos = (float *)m_cudaPosVBO;
	}

	//update constants
	setParameters(&m_params);
	//flushAndPrintRigidBodyParameters();
	//	flushAndPrintRigidBodyParameters();
	//	integrateSystemRigidBodies((float4 *)rbPositions, //rigid body center of mass
	//		(float4 *)rbVelocities, //velocity of rigid body
	//		(float4 *)rbForces, //total force applied to rigid body due to previous collisions
	//		(float4 *)rbAngularVelocity, //contains angular velocities for each rigid body
	//		rbQuaternion, //contains current quaternion for each rigid body
	//		(float4 *)rbTorque, //torque applied to rigid body due to previous collisions
	//		(float4 *)rbAngularMomentum, //cumulative angular momentum of the rigid body
	//		(float4 *)rbLinearMomentum, //cumulative linear momentum of the rigid body
	//		rbInertia, //original moment of inertia for each rigid body - 9 values per RB
	//		rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
	//		rbAngularAcceleration, //current angular acceleration due to misaligned angular momentum and velocity
	//		deltaTime, //dt
	//		rbRadii, //radius chosen for each rigid body sphere
	//		rbMass, //total mass of rigid body
	//		minPos, //smallest coordinate of scene's bounding box
	//		maxPos, //largest coordinate of scene's bounding box
	//		numRigidBodies, //number of rigid bodies
	//		m_params, //simulation parameters
	//		numThreads);

	integrateRigidBodyCPU((float4 *)rbPositions, //rigid body center of mass
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
		m_params); //simulation parameters
	//	unmapGLBufferObject(m_cuda_posvbo_resource);
	//	integrateRigidBodyCPU_RK(deltaTime);
	//	dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);

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

	//integrate
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

	//calculate grid hash
	calcHash(
		m_dGridParticleHash,
		m_dGridParticleIndex,
		dPos,
		m_numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//sort particles based on hash
	sortParticles(&m_dGridParticleHash, &m_dGridParticleIndex, m_numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//reorder particle arrays into sorted order and
	//find start and end of each cell
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
	//process collisions
	//	collide(
	//		(float4 *)pForce, //total force applied to rigid body
	//		rbIndices, //index of the rigid body each particle belongs to
	//		(float4 *)relativePos, //particle's relative position
	//		(float4 *)pTorque,  //rigid body angular momentum
	//		dCol,
	//		m_dVel,
	//		m_dSortedPos,
	//		m_dSortedVel,
	//		m_dGridParticleIndex,
	//		m_dCellStart,
	//		m_dCellEnd,
	//		m_numParticles,
	//		m_numGridCells);
	collideUniformGridRigidBodiesWrapper(
		(float4 *)pForce, //total force applied to rigid body
		rbIndices, //index of the rigid body each particle belongs to
		(float4 *)relativePos, //particle's relative position
		(float4 *)rbPositions, //rigid body center of mass
		(float4 *)rbAngularVelocity, //rigid body angular velocity
		(float4 *)rbVelocities, //rigid body linear velocity
		(float4 *)pTorque,  //rigid body angular momentum
		rbCurrentInertia,
		rbMass,
		(float4 *)dCol,
		(float4 *)m_dVel,
		(float4 *)m_dSortedPos,
		(float4 *)m_dSortedVel,
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_params,
		numThreads);

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
	if (toExit)
	{
		std::cerr << "Reduction gone wrong." << std::endl;
		exit(1);
	}

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// note: do unmap at end here to avoid unnecessary graphics/CUDA context switch

	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_colorvbo_resource);
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}

}

void ParticleSystem::updateStaticParticles(float deltaTime)
{
	float *dCol = (float *)mapGLBufferObject(&m_cuda_colorvbo_resource);
	setParameters(&m_params);
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
	// process collisions
	//	staticCollide(
	//		(float4 *)dCol,
	//		(float4 *)pForce, //total force applied to rigid body
	//		rbIndices, //index of the rigid body each particle belongs to
	//		(float4 *)relativePos, //particle's relative position
	//		(float4 *)pTorque,  //rigid body angular momentum
	//		r_radii, //radii of all scene particles
	//		m_dVel,
	//		m_dSortedPos,
	//		m_dSortedVel,
	//		staticSortedPos,
	//		m_dGridParticleIndex,
	//		staticCellStart,
	//		staticCellEnd,
	//		m_numParticles,
	//		m_numGridCells);
	ARcollisionsUniformGridWrapper(
		pCountARCollions, //count AR collisions per particle
		(float4 *)pForce, //total force applied to rigid body - per particle
		rbIndices, //index of the rigid body each particle belongs to
		(float4 *)relativePos, //particle's relative position
		(float4 *)pTorque,  //rigid body angular momentum - per particle
		rbCurrentInertia, //current moment of inertia of rigid body
		rbMass, //mass of rigid body
		(float4 *)dCol,
		r_radii, //radii of all scene particles
		(float4 *)m_dVel,               // output: new velocity
		(float4 *)m_dSortedPos,               // input: sorted positions
		(float4 *)m_dSortedVel,               // input: sorted velocities
		(float4 *)staticSortedPos, //positions of AR particles
		(float4 *)staticNorm, //normals associated with each AR particle
		m_dGridParticleIndex,    // input: sorted particle indices
		staticGridParticleIndex,//sorted AR particle indices
		staticCellStart,
		staticCellEnd,
		m_numParticles,
		m_params,
		numThreads);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	bool toExit = false;
	//	ReduceRigidBodyVariables(
	//			(float4 *)rbForces, //Output: rigid body forces - one element per rigid body
	//			(float4 *)rbTorque, //Output: rigid body torques - one element per rigid body
	//			(float4 *)rbPositions, //Output: rigid body positions - one element per rigid body
	//			(float4 *)pForce, //Input: rigid body forces - one element per particle
	//			(float4 *)pTorque, //Input: rigid body torques - one element per particle
	//			(float4 *)pPositions, //Input: rigid body positions - one element per particle
	//			particlesPerObjectThrown, //Auxil.: number of particles for each rigid body - one element per thrown objects
	//			isRigidBody, //Auxil.: flag indicating whether thrown object is a rigid body
	//			objectsThrown, //Auxil.: number of objects thrown - rigid bodies AND point sprites
	//			numRigidBodies, //Auxil.: number of rigid bodies
	//			numThreads,
	//			m_numParticles, //number of threads to use
	//			&toExit);
	ReduceRigidBodyARVariables(
		(float4 *)rbForces, //Output: rigid body forces - one element per rigid body
		(float4 *)rbTorque, //Output: rigid body torques - one element per rigid body
		(float4 *)rbPositions, //Output: rigid body positions - one element per rigid body
		(float4 *)pForce, //Input: rigid body forces - one element per particle
		(float4 *)pTorque, //Input: rigid body torques - one element per particle
		(float4 *)pPositions, //Input: rigid body positions - one element per particle
		pCountARCollions, //Input: AR collisions - one element per particle
		particlesPerObjectThrown, //Auxil.: number of particles for each rigid body - one element per thrown objects
		isRigidBody, //Auxil.: flag indicating whether thrown object is a rigid body
		objectsThrown, //Auxil.: number of objects thrown - rigid bodies AND point sprites
		numRigidBodies, //Auxil.: number of rigid bodies
		numThreads,
		m_numParticles, //number of threads to use
		&toExit);
	if (toExit)
	{
		std::cerr << "Reduction gone wrong." << std::endl;
		exit(1);
	}
	unmapGLBufferObject(m_cuda_colorvbo_resource);
}