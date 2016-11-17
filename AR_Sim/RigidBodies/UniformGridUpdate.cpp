#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
#include "BVHcreation.h"

void integrateRigidBodyCPU(
	glm::quat *cumulativeQuaternion,
	glm::mat4 *modelMatrixArray, // model matrix array used for rendering
	float4 *CMs, //rigid body center of mass
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
	SimParams params);

// step the simulation
void ParticleSystem::updateGrid(float deltaTime)
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

	integrateRigidBodyCPU(
		cumulativeQuaternion,
		modelMatrix, // model matrix array used for rendering
		(float4 *)rbPositions, //rigid body center of mass
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

void ParticleSystem::Find_Rigid_Body_Collisions_Uniform_Grid()
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

	// cudaMemset is mandatory if cudaMalloc takes place once
	checkCudaErrors(cudaMemset(contactDistance, 0, sizeof(float) * m_numParticles));

	FindRigidBodyCollisionsUniformGridWrapper(
		rbIndices, // index of the rigid body each particle belongs to
		collidingRigidBodyIndex, // index of rigid body of contact
		collidingParticleIndex, // index of particle of contact
		contactDistance, // penetration distance
		(float4 *)dCol, // particle color
		(float4 *)m_dSortedPos,  // sorted positions
		m_dGridParticleIndex, // sorted particle indices
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_params,
		numThreads);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::Find_Augmented_Reality_Collisions_Uniform_Grid()
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
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

void ParticleSystem::updateUniformGrid(float deltaTime)
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
	//Handle_Wall_Collisions();

	if (simulateAR)
	{
		// find collisions between rigid bodies and real scene
		Find_Augmented_Reality_Collisions_Uniform_Grid();

		// handle collisions between rigid bodies and real scene
		//Handle_Augmented_Reality_Collisions_Baraff_CPU();
		Handle_Augmented_Reality_Collisions_Catto_CPU();
	}

	// find collisions between rigid bodies
	//Find_Rigid_Body_Collisions_Uniform_Grid();

	// handle collisions between rigid bodies
	//Handle_Rigid_Body_Collisions_Baraff_CPU();

	//// cudaFree contact info variables - uncomment if no collision handling routine is used
	//checkCudaErrors(cudaFree(collidingRigidBodyIndex));
	//checkCudaErrors(cudaFree(collidingParticleIndex));
	//checkCudaErrors(cudaFree(contactDistance));
	// note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_colorvbo_resource);
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}

}