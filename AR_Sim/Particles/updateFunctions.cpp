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
	SimParams params) //simulation parameters
{
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

		//rbAngularMomentum_CPU[index] *= 0.99;
		//rbLinearMomentum_CPU[index] *= 0.99;

		float4 locPos = CMs_CPU[index];
		float locMass = rbMass_CPU[index];
		glm::quat quaternion = rbQuaternion_CPU[index];
		glm::mat3 inertia = rbInertia_CPU[index];
		glm::mat3 currentInertia = rbCurrentInertia_CPU[index];
		float4 locVel = vel_CPU[index];
		float4 locAng = rbAngularVelocity_CPU[index];

		//locVel += make_float4(0, -0.981, 0, 0) * deltaTime;

		//float4 locVel = rbLinearMomentum_CPU[index] / locMass;
		//float4 locMomentum = rbAngularMomentum_CPU[index];
		//glm::vec3 glmMomentum(locMomentum.x, locMomentum.y, locMomentum.z);
		//glm::vec3 glmAng = currentInertia * glmMomentum;
		//float4 locAng = make_float4(glmAng.x, glmAng.y, glmAng.z, 0);

		locPos += locVel * deltaTime;

		glm::vec3 newVelocity(locAng.x, locAng.y, locAng.z);
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
		quaternion = normalize(quaternion);
		glm::mat3 rot = mat3_cast(quaternion);
		currentInertia = rot * inertia * transpose(rot);

		/*std::cout << "New velocity for rigid body #" << index + 1 << " is: (" <<
			newVelocity.x << ", " << newVelocity.y << ", " << newVelocity.z << ")" << std::endl;*/
		/*std::cout << "New quaternion for rigid body #" << index + 1 << " is: (" << quaternion.w << ", " <<
			quaternion.x << ", " << quaternion.y << ", " << quaternion.z << ")" << std::endl;*/
		/*std::cout << "New center of mass for rigid body #" << index + 1 << " is: (" <<
			locPos.x << ", " << locPos.y << ", " << locPos.z << ")" << std::endl;*/

		/*glm::quat quatVelocity(0, locAng.x, locAng.y, locAng.z);
		glm::quat qdot = 0.5f * quatVelocity * quaternion;
		cumulativeQuaternion[index] += qdot * deltaTime;*/
		cumulativeQuaternion[index] = quaternion * cumulativeQuaternion[index];
		cumulativeQuaternion[index] = normalize(cumulativeQuaternion[index]);
		rot = mat3_cast(cumulativeQuaternion[index]);
		glm::mat4 modelMatrix = glm::mat4(1.f);
		for (int row = 0; row < 3; row++)
			for (int col = 0; col < 3; col++)
				modelMatrix[row][col] = rot[row][col];

		
		modelMatrix[3][0] = locPos.x;
		modelMatrix[3][1] = locPos.y;
		modelMatrix[3][2] = locPos.z;

		modelMatrixArray[index] = modelMatrix;

		locVel += make_float4(params.gravity, 0);
		CMs_CPU[index] = locPos;
		vel_CPU[index] = locVel;
		rbCurrentInertia_CPU[index] = currentInertia;
		rbQuaternion_CPU[index] = quaternion;
		rbAngularVelocity_CPU[index] = make_float4(newVelocity.x, newVelocity.y, newVelocity.z, 0);
		rbTorque_CPU[index] = make_float4(0, 0, 0, 0); // reset torque to zero

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

void ParticleSystem::flushAndPrintRigidBodyParameters()
{
	glm::quat *quatTest = new glm::quat[4 * numRigidBodies];
	checkCudaErrors(cudaMemcpy(quatTest, rbQuaternion, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	float *posTest = new float[4 * numRigidBodies];
	checkCudaErrors(cudaMemcpy(posTest, rbPositions, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	float *velTest = new float[4 * numRigidBodies];
	checkCudaErrors(cudaMemcpy(velTest, rbVelocities, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	float *angTest = new float[4 * numRigidBodies];
	checkCudaErrors(cudaMemcpy(angTest, rbAngularVelocity, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	glm::mat3 *inertia = new glm::mat3[numRigidBodies];
	checkCudaErrors(cudaMemcpy(inertia, rbCurrentInertia, sizeof(glm::mat3) * numRigidBodies, cudaMemcpyDeviceToHost));
	float *mass = new float[numRigidBodies];
	checkCudaErrors(cudaMemcpy(mass, rbMass, sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	float *torqueTest = new float[4 * numRigidBodies];
	//checkCudaErrors(cudaMemcpy(torqueTest, rbTorque, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	float *LTest = new float[4 * numRigidBodies];
	//checkCudaErrors(cudaMemcpy(LTest, rbAngularMomentum, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	glm::vec3 *ldot = new glm::vec3[numRigidBodies];
	//checkCudaErrors(cudaMemcpy(ldot, rbAngularAcceleration, sizeof(glm::vec3) * numRigidBodies, cudaMemcpyDeviceToHost));
	for(int i = 0; i < numRigidBodies; i++)
	{

		std::cout << "Rigid body #" << i + 1 << std::endl;
		glm::mat3 localInertia = inertia[i];
//		std::cout <<  "Position: (" << posTest[4*i] << " " <<
//				posTest[4*i+1] << " " << posTest[4*i+2] << " " << posTest[4*i+3] << ")" << std::endl;
//		std::cout <<  "Velocity: (" << velTest[4*i] << " " <<
//				velTest[4*i+1] << " " << velTest[4*i+2] << " " << velTest[4*i+3] << ")" << std::endl;
		glm::vec3 L(LTest[4*i], LTest[4*i + 1], LTest[4*i + 2]);
		glm::vec3 w(angTest[4*i], angTest[4*i + 1], angTest[4*i + 2]);
		glm::vec3 wdot = localInertia * glm::cross(L, w);
		glm::vec3 GPUwdot = ldot[i];
		std::cout <<  "GPU Angular acceleration: (" << GPUwdot.x << " " <<
				GPUwdot.y << " " << GPUwdot.z  << ")" << std::endl;
		std::cout <<  "Angular acceleration: (" << wdot.x << " " <<
				wdot.y << " " << wdot.z  << ")" << std::endl;
		std::cout <<  "Angular momentum: (" << LTest[4*i] << " " <<
				LTest[4*i+1] << " " << LTest[4*i+2] << " " << LTest[4*i+3] << ")" << std::endl;
		std::cout <<  "Angular velocity: (" << angTest[4*i] << " " <<
				angTest[4*i+1] << " " << angTest[4*i+2] << " " << angTest[4*i+3] << ")" << std::endl;
		std::cout <<  "Corrected angular velocity: (" << angTest[4*i] - wdot.x * 0.01 << " " <<
				angTest[4*i+1] - wdot.y * 0.01 << " " << angTest[4*i+2] - wdot.z * 0.01 << " " << angTest[4*i+3] << ")" << std::endl;

//		std::cout <<  "Torque: (" << torqueTest[4*i] << " " <<
//				torqueTest[4*i+1] << " " << torqueTest[4*i+2] << " " << torqueTest[4*i+3] << ")" << std::endl;
//		std::cout <<  "Quaternion: (" << quatTest[i].x << " " <<
//				quatTest[i].y << " " << quatTest[i].z << " " << quatTest[i].w << ")" << std::endl;
//
//		std::cout << "Inertia matrix:" << std::endl;
//		for (int row = 0; row < 3; row++)
//		{
//			for (int col = 0; col < 3; col++)
//				std::cout << localInertia[row][col] << " ";
//			std::cout << std::endl;
//		}
//		std::cout << "Mass: " << mass[i] << std::endl;
		std::cout << std::endl;


		if (posTest[4*i] != posTest[4*i] || posTest[4*i + 1] != posTest[4*i + 1] || posTest[4*i + 2] != posTest[4*i + 2])
			exit(1);
		if (angTest[4*i] != angTest[4*i] || angTest[4*i + 1] != angTest[4*i + 1] || angTest[4*i + 2] != angTest[4*i + 2])
			exit(1);
		if (torqueTest[4*i] != torqueTest[4*i] || torqueTest[4*i + 1] != torqueTest[4*i + 1] || torqueTest[4*i + 2] != torqueTest[4*i + 2])
			exit(1);
	}
	delete LTest;
	delete ldot;
	delete torqueTest;
	delete angTest;
	delete velTest;
	delete inertia;
	delete mass;
	delete posTest;
	delete quatTest;
}

void ParticleSystem::Handle_Wall_Collisions()
{
	WallCollisionWrapper(
		(float4 *)dPos, // particle positions
		(float4 *)rbPositions, // rigid body center of mass
		minPos, // scene AABB's smallest coordinates
		maxPos, // scene AABB's largest coordinates
		(float4 *)rbVelocities, // rigid body linear velocity
		(float4 *)rbAngularVelocity, // rigid body angular velocity
		//(float4 *)rbLinearMomentum, // rigid body linear momentum
		//(float4 *)rbAngularMomentum, // rigid body angular momentum
		NULL,
		NULL,
		rbCurrentInertia, // current rigid body inverse inertia tensor
		rbMass, // rigid body mass
		rbIndices, // index showing where each particle belongs
		particlesPerObjectThrown, // number of particles per rigid body
		numRigidBodies, // total number of scene's rigid bodies
		m_numParticles, // number of particles to test
		numThreads, // number of threads to use
		m_params);

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
}

void ParticleSystem::Handle_Rigid_Body_Collisions_Baraff_CPU()
{

	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body
	
	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingRigidBodyIndex_CPU = new int[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingRigidBodyIndex_CPU, collidingRigidBodyIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	int current_particle = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				
				int rigidBodyIndex = collidingRigidBodyIndex_CPU[current_particle];
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				//std::cout << current_particle << " collides with " << particleIndex << std::endl;
				if ((index < rigidBodyIndex || collisionMethod == M_BVH) &&
					testParticleCollision(CMs_CPU[index] + relative_CPU[current_particle],
					CMs_CPU[rigidBodyIndex] + relative_CPU[particleIndex],
					m_params.particleRadius,
					m_params.particleRadius,
					CMs_CPU[index]))
				{
					float4 cp, cn;
					findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
						CMs_CPU[rigidBodyIndex] + relative_CPU[particleIndex],
						m_params.particleRadius,
						m_params.particleRadius,
						cp, cn);
					/*float4 r1 = cp - CMs_CPU[index];
					float4 r2 = cp - CMs_CPU[rigidBodyIndex];*/
					float4 r1 = relative_CPU[current_particle];
					float4 r2 = relative_CPU[particleIndex];

					glm::mat3 IinvA = rbCurrentInertia_CPU[index];
					glm::mat3 IinvB = rbCurrentInertia_CPU[rigidBodyIndex];
					float mA = rbMass_CPU[index];
					float mB = rbMass_CPU[rigidBodyIndex];
					float impulse = computeImpulseMagnitude(
						vel_CPU[index], vel_CPU[rigidBodyIndex],
						rbAngularVelocity_CPU[index], rbAngularVelocity_CPU[rigidBodyIndex],
						r1, r2,	IinvA, IinvB, mA, mB, cn);

					float4 impulseVector = cn * impulse;

					/*std::cout << "Collision normal: (" << cn.x << ", " <<
						cn.y << ", " << cn.z << ", " << cn.w << ")" << std::endl;
					std::cout << "r1: (" << r1.x << ", " <<
						r1.y << ", " << r1.z << ", " << r1.w << ")" << std::endl;
					std::cout << "r2: (" << r2.x << ", " <<
						r2.y << ", " << r2.z << ", " << r2.w << ")" << std::endl;*/

					/*glm::vec3 vA(vel_CPU[index].x, vel_CPU[index].y, vel_CPU[index].z);
					glm::vec3 vB(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);

					glm::vec3 wA(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);
					glm::vec3 wB(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);

					glm::vec3 rAA(r1.x, r1.y, r1.z);
					glm::vec3 rBB(r2.x, r2.y, r2.z);

					glm::vec3 norm(cn.x, cn.y, cn.z);

					glm::vec3 velA = vA + glm::cross(wA, rAA);
					glm::vec3 velB = vB + glm::cross(wB, rBB);

					float numerator = glm::dot(velA - velB, norm);
					std::cout << "Relative velocity: " << numerator << std::endl;

					std::cout << "Iinv1:" << std::endl;
					for (int row = 0; row < 3; row++)
					{
						for (int col = 0; col < 3; col++)
							std::cout << IinvA[row][col] << " ";
						std::cout << std::endl;
					}

					std::cout << "Iinv2:" << std::endl;
					for (int row = 0; row < 3; row++)
					{
						for (int col = 0; col < 3; col++)
							std::cout << IinvB[row][col] << " ";
						std::cout << std::endl;
					}

					std::cout << "V1 before impulse: (" << vel_CPU[index].x << ", " <<
						vel_CPU[index].y << ", " << vel_CPU[index].z << ")" << std::endl;
					std::cout << "V2 before impulse: (" << vel_CPU[rigidBodyIndex].x << ", " <<
						vel_CPU[rigidBodyIndex].y << ", " << vel_CPU[rigidBodyIndex].z << ")" << std::endl;*/

					// apply linear impulse
					vel_CPU[index] += impulseVector / mA;
					vel_CPU[rigidBodyIndex] -= impulseVector / mB;
					/*std::cout << "V1 after impulse: (" << vel_CPU[index].x << ", " <<
						vel_CPU[index].y << ", " << vel_CPU[index].z << ")" << std::endl;
					std::cout << "V2 after impulse: (" << vel_CPU[rigidBodyIndex].x << ", " <<
						vel_CPU[rigidBodyIndex].y << ", " << vel_CPU[rigidBodyIndex].z << ")" << std::endl;*/
					// compute auxiliaries for angular impulse
					glm::vec3 rA(r1.x, r1.y, r1.z);
					glm::vec3 rB(r2.x, r2.y, r2.z);
					glm::vec3 impulseVectorGLM(impulseVector.x, impulseVector.y, impulseVector.z);

					/*std::cout << "W1 before impulse: (" << rbAngularVelocity_CPU[index].x << ", " <<
						rbAngularVelocity_CPU[index].y << ", " << rbAngularVelocity_CPU[index].z << ")" << std::endl;
					std::cout << "W2 before impulse: (" << rbAngularVelocity_CPU[rigidBodyIndex].x << ", " <<
						rbAngularVelocity_CPU[rigidBodyIndex].y << ", " << rbAngularVelocity_CPU[rigidBodyIndex].z << ")" << std::endl;*/

					// apply angular impulse
					glm::vec3 AngularImpulse = IinvA *
						(glm::cross(glm::vec3(r1.x, r1.y, r1.z), impulseVectorGLM));
					rbAngularVelocity_CPU[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

					AngularImpulse = IinvB *
						(glm::cross(glm::vec3(r2.x, r2.y, r2.z), impulseVectorGLM * (-1.f)));
					rbAngularVelocity_CPU[rigidBodyIndex] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

					//std::cout << "W1 after impulse: (" << rbAngularVelocity_CPU[index].x << ", " <<
					//	rbAngularVelocity_CPU[index].y << ", " << rbAngularVelocity_CPU[index].z << ")" << std::endl;
					//std::cout << "W2 after impulse: (" << rbAngularVelocity_CPU[rigidBodyIndex].x << ", " <<
					//	rbAngularVelocity_CPU[rigidBodyIndex].y << ", " << rbAngularVelocity_CPU[rigidBodyIndex].z << ")" << std::endl;
					//std::cout << std::endl;
				}
			}
			current_particle++;
		}
		
	}

	checkCudaErrors(cudaMemcpy(rbPositions, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbVelocities, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	delete rbCurrentInertia_CPU;
	delete rbMass_CPU;

	delete relative_CPU;

	delete contactDistance_CPU;
	delete collidingRigidBodyIndex_CPU;
	delete collidingParticleIndex_CPU;

}

void ParticleSystem::Handle_Rigid_Body_Collisions_Catto_CPU()
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *particlePosition_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(particlePosition_CPU, dPos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	// copy contact info to CPU - one contact per particle
	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingRigidBodyIndex_CPU = new int[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingRigidBodyIndex_CPU, collidingRigidBodyIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	// pre-processing step
	// count total number of collisions

	int current_particle = 0;
	int collision_counter = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			int rigidBodyIndex = collidingRigidBodyIndex_CPU[current_particle];
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				collision_counter++;
			}
			current_particle++;
		}
	}

	//#define PRINT_COLLISIONS
#ifdef PRINT_COLLISIONS
	std::cout << "Number of collisions: " << collision_counter << std::endl;
	std::ofstream file("collisions.txt");
#endif
	// initialize auxiliary contact variables
	float4 *contactNormal = new float4[collision_counter]; // store one normal per collision
	float4 *contactPoint = new float4[collision_counter]; // store one contact point per collision
	float *contactAccumulatedImpulse = new float[collision_counter]; // store the accumulated impulses per collision
	int *contactRigidBody_1 = new int[collision_counter]; // index to colliding rigid body
	int *contactRigidBody_2 = new int[collision_counter]; // index to colliding rigid body
	float *contactBias = new float[collision_counter]; // bias at each contact point
	memset(contactBias, 0, sizeof(float) * collision_counter);
	memset(contactNormal, 0, sizeof(float) * 4 * collision_counter);
	memset(contactPoint, 0, sizeof(float) * 4 * collision_counter);
	memset(contactAccumulatedImpulse, 0, sizeof(float) * collision_counter);
	memset(contactRigidBody_1, 0, sizeof(int) * collision_counter);
	memset(contactRigidBody_2, 0, sizeof(int) * collision_counter);

	collision_counter = 0;
	current_particle = 0;

	float epsilon = 1.f;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			int rigidBodyIndex = collidingRigidBodyIndex_CPU[current_particle];
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				
				float4 cp, cn;
				findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
					CMs_CPU[rigidBodyIndex] + relative_CPU[particleIndex],
					m_params.particleRadius,
					m_params.particleRadius,
					cp, cn);

				contactNormal[collision_counter] = cn;
				contactPoint[collision_counter] = cp;
				contactRigidBody_1[collision_counter] = index;
				contactRigidBody_2[collision_counter] = rigidBodyIndex;

				float3 v1 = make_float3(vel_CPU[index].x, vel_CPU[index].y, vel_CPU[index].z);
				float3 w1 = make_float3(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);

				float3 v2 = make_float3(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);
				float3 w2 = make_float3(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);
				
				float v_rel = dot(v1 + cross(w1, make_float3(relative_CPU[current_particle])), make_float3(contactNormal[collision_counter])) - 
					dot(v2 + cross(w2, make_float3(relative_CPU[particleIndex])), make_float3(contactNormal[collision_counter])); // relative velocity at current contact
				
				contactBias[collision_counter] = epsilon * v_rel;
#ifdef PRINT_COLLISIONS
				std::cout << "Collision #" << collision_counter + 1 << " initial bias: " << contactBias[collision_counter] << std::endl;
				float4 r = relative_CPU[current_particle];//position_CPU[particleIndex] - CMs_CPU[index];
				//std::cout << "Collision #" << collision_counter + 1 << ": (" << r.x << ", " << r.y << ", " << r.z << ")" << std::endl;
				file << r.x << " " << r.y << " " << r.z << " " << std::endl;
#endif
				collision_counter++;
			}
			current_particle++;
		}
	}
#ifdef PRINT_COLLISIONS
	file.close();
#endif
	// solve contacts using SIS

	const int iterations = 8; // number of iterations per simulation step
	const int UPPER_BOUND = 100; // upper bound for accumulated impulse
	for (int k = 0; k < iterations; k++)
	{
		for (int c = 0; c < collision_counter; c++)
		{
			glm::vec3 n(contactNormal[c].x, contactNormal[c].y, contactNormal[c].z); // collision normal
			//glm::vec3 n(0, sqrt(2.f) / 2, sqrt(2.f) / 2);
			//glm::vec3 n(0, 1, 0);
			float4 point = contactPoint[c];
			int rigidBodyIndex1 = contactRigidBody_1[c];
			int rigidBodyIndex2 = contactRigidBody_2[c];

			float4 r1 = point - CMs_CPU[rigidBodyIndex1];
			glm::vec3 p1(r1.x, r1.y, r1.z); // contact to be processed at this iteration
			glm::mat3 Iinv1 = rbCurrentInertia_CPU[rigidBodyIndex1];
			float m1 = rbMass_CPU[rigidBodyIndex1];
			glm::vec3 v1(vel_CPU[rigidBodyIndex1].x, vel_CPU[rigidBodyIndex1].y, vel_CPU[rigidBodyIndex1].z);
			glm::vec3 w1(rbAngularVelocity_CPU[rigidBodyIndex1].x, rbAngularVelocity_CPU[rigidBodyIndex1].y, rbAngularVelocity_CPU[rigidBodyIndex1].z);

			float4 r2 = point - CMs_CPU[rigidBodyIndex2];
			glm::vec3 p2(r2.x, r2.y, r2.z); // contact to be processed at this iteration
			glm::mat3 Iinv2 = rbCurrentInertia_CPU[rigidBodyIndex2];
			float m2 = rbMass_CPU[rigidBodyIndex2];
			glm::vec3 v2(vel_CPU[rigidBodyIndex2].x, vel_CPU[rigidBodyIndex2].y, vel_CPU[rigidBodyIndex2].z);
			glm::vec3 w2(rbAngularVelocity_CPU[rigidBodyIndex2].x, rbAngularVelocity_CPU[rigidBodyIndex2].y, rbAngularVelocity_CPU[rigidBodyIndex2].z);

			float mc = 1 / m1 + 1 / m2 + glm::dot(glm::cross(Iinv1 * glm::cross(p1, n), p1), n) + 
				glm::dot(glm::cross(Iinv2 * glm::cross(p2, n), p2), n); // active mass at current collision
			if (abs(mc) < 0.00001) mc = 1.f;
			float v_rel = glm::dot(v1 + glm::cross(w1, p1), n) - glm::dot(v2 + glm::cross(w2, p2), n); // relative velocity at current contact
			float corrective_impulse = -(v_rel + contactBias[c]) / mc; // corrective impulse magnitude
#ifdef PRINT_COLLISIONS
			std::cout << "Iteration: " << k << std::endl;
			std::cout << "Contact: " << c << std::endl;
			std::cout << "Collision normal: (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
			std::cout << "Relative linear velocity: " << glm::dot(v, n) << std::endl;
			std::cout << "Relative angular velocity: " << glm::dot(glm::cross(w, p), n) << std::endl;
			std::cout << "Total relative velocity: " << v_rel << std::endl;
#endif
			//if (corrective_impulse < 0)
			//	std::cout << "Negative corrective impulse encountered: " << corrective_impulse << std::endl;

			float temporary_impulse = contactAccumulatedImpulse[c]; // make a copy of old accumulated impulse
			temporary_impulse = temporary_impulse + corrective_impulse; // add corrective impulse to accumulated impulse
			//clamp new accumulated impulse
			if (temporary_impulse < 0)
				temporary_impulse = 0; // allow no negative accumulated impulses
			else if (temporary_impulse > UPPER_BOUND)
				temporary_impulse = UPPER_BOUND; // max upper bound for accumulated impulse
			// compute difference between old and new impulse
			corrective_impulse = temporary_impulse - contactAccumulatedImpulse[c];
			contactAccumulatedImpulse[c] = temporary_impulse; // store new clamped accumulated impulse
			// apply new clamped corrective impulse difference to velocity
			glm::vec3 impulse_vector = corrective_impulse * n;
			v1 = v1 + impulse_vector / m1;
			w1 = w1 + Iinv1 * glm::cross(p1, impulse_vector);

			vel_CPU[rigidBodyIndex1] = make_float4(v1.x, v1.y, v1.z, 0);
			rbAngularVelocity_CPU[rigidBodyIndex1] = make_float4(w1.x, w1.y, w1.z, 0);

			v2 = v2 - impulse_vector / m2;
			w2 = w2 - Iinv2 * glm::cross(p2, impulse_vector);
			vel_CPU[rigidBodyIndex2] = make_float4(v2.x, v2.y, v2.z, 0);
			rbAngularVelocity_CPU[rigidBodyIndex2] = make_float4(w2.x, w2.y, w2.z, 0);

#ifdef PRINT_COLLISIONS	
			std::cout << "Applied impulse: " << corrective_impulse << std::endl;
			std::cout << "New linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
			std::cout << "New angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;
			std::cout << std::endl;
#endif
		}
	}

	checkCudaErrors(cudaMemcpy(rbPositions, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbVelocities, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	delete rbCurrentInertia_CPU;
	delete rbMass_CPU;

	delete relative_CPU;
	delete particlePosition_CPU;

	delete contactDistance_CPU;
	delete collidingParticleIndex_CPU;
	delete collidingRigidBodyIndex_CPU;

	delete contactNormal;
	delete contactAccumulatedImpulse;
	delete contactRigidBody_1;
	delete contactRigidBody_2;
	delete contactPoint;
	delete contactBias;

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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::Handle_Rigid_Body_Collisions_Baraff_GPU()
{
	HandleRigidBodyCollisionWrapper(
		(float4 *)dPos, // particle positions
		(float4 *)rbPositions, // rigid body center of mass
		(float4 *)rbVelocities, // rigid body linear velocity
		(float4 *)rbAngularVelocity, // rigid body angular velocity
		//(float4 *)rbLinearMomentum, // rigid body linear momentum
		//(float4 *)rbAngularMomentum, // rigid body angular momentum
		NULL,
		NULL,
		rbCurrentInertia, // current rigid body inverse inertia tensor
		rbMass, // rigid body mass
		rbIndices, // index showing where each particle belongs
		particlesPerObjectThrown, // number of particles per rigid body
		collidingRigidBodyIndex, // index of rigid body of contact
		collidingParticleIndex, // index of particle of contact
		contactDistance, // penetration distance
		numRigidBodies, // total number of scene's rigid bodies
		m_numParticles, // number of particles to test
		numThreads, // number of threads to use
		m_params); // simulation parameters

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// cudaFree contact info variables
	/*checkCudaErrors(cudaFree(collidingRigidBodyIndex));
	checkCudaErrors(cudaFree(collidingParticleIndex));
	checkCudaErrors(cudaFree(contactDistance));*/
}

void ParticleSystem::Handle_Augmented_Reality_Collisions_Baraff_GPU()
{
	HandleAugmentedRealityCollisionWrapper(
		(float4 *)dPos, // particle positions
		(float4 *)staticPos, // scene particle positions
		(float4 *)rbPositions, // rigid body center of mass
		(float4 *)rbVelocities, // rigid body linear velocity
		(float4 *)rbAngularVelocity, // rigid body angular velocity
		//(float4 *)rbLinearMomentum, // rigid body linear momentum
		//(float4 *)rbAngularMomentum, // rigid body angular momentum
		NULL,
		NULL,
		rbCurrentInertia, // current rigid body inverse inertia tensor
		rbMass, // rigid body mass
		rbIndices, // index showing where each particle belongs
		particlesPerObjectThrown, // number of particles per rigid body
		collidingParticleIndex, // index of particle of contact
		contactDistance, // penetration distance
		numRigidBodies, // total number of scene's rigid bodies
		m_numParticles, // number of particles to test
		numThreads, // number of threads to use
		m_params); // simulation parameters

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// cudaFree contact info variables
	/*checkCudaErrors(cudaFree(collidingRigidBodyIndex));
	checkCudaErrors(cudaFree(collidingParticleIndex));
	checkCudaErrors(cudaFree(contactDistance));*/
}

void ParticleSystem::Handle_Augmented_Reality_Collisions_Baraff_CPU()
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *particlePosition_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(particlePosition_CPU, dPos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *position_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(position_CPU, staticPos, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *normal_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(normal_CPU, staticNorm, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));

	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	int current_particle = 0;
	//static int iterations = 0;
	
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				/*if (iterations < 120)
				{
					std::ofstream file("collision_iteration.txt");
					file << iterations << std::endl;
					file.close();
				}*/
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				if (testParticleCollision(CMs_CPU[index] + relative_CPU[current_particle],
					position_CPU[particleIndex],
					m_params.particleRadius,
					m_params.particleRadius,
					CMs_CPU[index]))
				{
					float4 cp, cn;
					findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
						position_CPU[particleIndex],
						m_params.particleRadius,
						m_params.particleRadius,
						cp, cn);
					cn = normal_CPU[particleIndex];
					//cn = make_float4(0, 1, 0, 0);
					// customly added for collision with horizontal plane
					// TODO: remove
					cp = relative_CPU[current_particle];
					/*cp.y = 1.5;
					cn = make_float4(0, 1, 0, 0);
					CMs_CPU[index].y = 1.5 - relative_CPU[current_particle].y;*/
					float4 r1 = relative_CPU[current_particle];
					//float4 r1 = cp - CMs_CPU[index];
					
					glm::mat3 IinvA = rbCurrentInertia_CPU[index];
				
					float mA = rbMass_CPU[index];
					float impulse = computeImpulseMagnitude(vel_CPU[index], rbAngularVelocity_CPU[index], r1, IinvA, mA, cn);

					float4 impulseVector = cn * impulse;
//#define DEBUG_BARAFF
					// apply linear impulse
#ifdef DEBUG_BARAFF
					std::cout << "V before impulse: (" << vel_CPU[index].x << ", " <<
						vel_CPU[index].y << ", " << vel_CPU[index].z << ")" << std::endl;
					std::cout << "Collision normal: (" << cn.x << ", " <<
						cn.y << ", " << cn.z << ", " << cn.w << ")" << std::endl;
					std::cout << "Linear impulse applied: (" << impulseVector.x << ", " <<
						impulseVector.y << ", " << impulseVector.z << ", " << impulseVector.w << ")" << std::endl;
#endif
					vel_CPU[index] += impulseVector / mA;
					vel_CPU[index].w = 0;
#ifdef DEBUG_BARAFF
					std::cout << "V after impulse: (" << vel_CPU[index].x << ", " <<
						vel_CPU[index].y << ", " << vel_CPU[index].z << ")" << std::endl;
					std::cout << std::endl;
#endif
					// compute auxiliaries for angular impulse
					glm::vec3 rA(r1.x, r1.y, r1.z);
					glm::vec3 impulseVectorGLM(impulseVector.x, impulseVector.y, impulseVector.z);

					// apply angular impulse
					glm::vec3 AngularImpulse = IinvA *
						(glm::cross(glm::vec3(r1.x, r1.y, r1.z), impulseVectorGLM));

					glm::vec3 MomentumAdded = (glm::cross(glm::vec3(r1.x, r1.y, r1.z), impulseVectorGLM));
#ifdef DEBUG_BARAFF
					std::cout << "Angular momentum added: (" << MomentumAdded.x << ", " <<
						MomentumAdded.y << ", " << MomentumAdded.z << ")" << std::endl;

					std::cout << "Angular impulse applied: (" << AngularImpulse.x << ", " <<
						AngularImpulse.y << ", " << AngularImpulse.z  << ")" << std::endl;

					std::cout << "W before impulse: (" << rbAngularVelocity_CPU[index].x << ", " <<
						rbAngularVelocity_CPU[index].y << ", " << rbAngularVelocity_CPU[index].z << ")" << std::endl;
#endif
					rbAngularVelocity_CPU[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);
					rbAngularVelocity_CPU[index].w = 0;
#ifdef DEBUG_BARAFF
					std::cout << "W after impulse: (" << rbAngularVelocity_CPU[index].x << ", " <<
						rbAngularVelocity_CPU[index].y << ", " << rbAngularVelocity_CPU[index].z << ")" << std::endl;
					std::cout << std::endl;
#endif

				}
			}
			current_particle++;
		}

	}
	//iterations++;
	checkCudaErrors(cudaMemcpy(rbPositions, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbVelocities, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	delete rbCurrentInertia_CPU;
	delete rbMass_CPU;

	delete relative_CPU;
	delete position_CPU;
	delete normal_CPU;
	delete particlePosition_CPU;

	delete contactDistance_CPU;
	delete collidingParticleIndex_CPU;

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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

/*
* Rigid body collision handling based on Erin Catto's 2009 GDC paper
* titled "Modeling and solving constraints". Based on this paper we
* implement a Sequential Impulse Solver for handling contacts between
* virtual objects and our Augmented Reality Scene. For now, position
* correction is not implemented. Each rigid body is handled independently.
* If this method works, it can substitute Baraff's method for collision
* between virtual objects as well. In the future, collision reaction from
* both virtual and augmented reality contacts could be unified.
*/
void ParticleSystem::Handle_Augmented_Reality_Collisions_Catto_CPU()
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *particlePosition_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(particlePosition_CPU, dPos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *position_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(position_CPU, staticPos, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *normal_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(normal_CPU, staticNorm, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));

	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	// pre-processing step
	// count total number of collisions

	int current_particle = 0;
	int collision_counter = 0;
	//static int iteration = 1;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				collision_counter++;
				/*if (iteration < 120)
				{
					std::ofstream file("collision_iteration.txt");
					file << iteration << std::endl;
					file.close();
				}*/
			}
			current_particle++;
		}
	}
	//iteration++;
//#define PRINT_COLLISIONS
#ifdef PRINT_COLLISIONS
	std::cout << "Number of collisions: " << collision_counter << std::endl;
	std::ofstream file("collisions.txt");
#endif
	// initialize auxiliary contact variables
	float4 *contactNormal = new float4[collision_counter]; // store one normal per collision
	float4 *contactPoint = new float4[collision_counter]; // store one contact point per collision
	float *contactAccumulatedImpulse = new float[collision_counter]; // store the accumulated impulses per collision
	int *contactRigidBody = new int[collision_counter]; // index to colliding rigid body
	float *contactBias = new float[collision_counter]; // bias at each contact point
	memset(contactBias, 0, sizeof(float) * collision_counter);
	memset(contactNormal, 0, sizeof(float) * 4 * collision_counter);
	memset(contactPoint, 0, sizeof(float) * 4 * collision_counter);
	memset(contactAccumulatedImpulse, 0, sizeof(float) * collision_counter);
	memset(contactRigidBody, 0, sizeof(int) * collision_counter);
	
	collision_counter = 0;
	current_particle = 0;

	const float epsilon = m_params.ARrestitution;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				contactNormal[collision_counter] = normal_CPU[particleIndex]; // scene's normal at collision point
				contactNormal[collision_counter] = make_float4(0, 1, 0, 0);
				/*float4 cp, cn;
				findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
					position_CPU[particleIndex],
					m_params.particleRadius,
					m_params.particleRadius,
					cp, cn);*/

				//contactPoint[collision_counter] = position_CPU[particleIndex]; // exact contact point
				contactPoint[collision_counter] = relative_CPU[current_particle] + CMs_CPU[index];
				contactRigidBody[collision_counter] = index;
				float3 v = make_float3(vel_CPU[index].x, vel_CPU[index].y, vel_CPU[index].z);
				float3 w = make_float3(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);
				float v_rel = dot(v + cross(w, make_float3(relative_CPU[current_particle])), make_float3(contactNormal[collision_counter])); // relative velocity at current contact
				contactBias[collision_counter] = epsilon * v_rel;
#ifdef PRINT_COLLISIONS
				std::cout << "Collision #" << collision_counter + 1 << " initial bias: " << contactBias[collision_counter] << std::endl;
				float4 r = relative_CPU[current_particle];//position_CPU[particleIndex] - CMs_CPU[index];
				//std::cout << "Collision #" << collision_counter + 1 << ": (" << r.x << ", " << r.y << ", " << r.z << ")" << std::endl;
				file << r.x << " " << r.y << " " << r.z << " " << std::endl;
#endif
				collision_counter++;
			}
			current_particle++;
		}
	}
#ifdef PRINT_COLLISIONS
	file.close();
#endif
	// solve contacts using SIS
	
	const int iterations = 8; // number of iterations per simulation step
	const int UPPER_BOUND = 100; // upper bound for accumulated impulse
	for (int k = 0; k < iterations; k++)
	{
		for (int c = 0; c < collision_counter; c++)
		{
			glm::vec3 n(contactNormal[c].x, contactNormal[c].y, contactNormal[c].z); // collision normal
			//glm::vec3 n(0, sqrt(2.f) / 2, sqrt(2.f) / 2);
			//glm::vec3 n(0, 1, 0);
			float4 point = contactPoint[c];
			int rigidBodyIndex = contactRigidBody[c];
			float4 r = point - CMs_CPU[rigidBodyIndex];
			glm::vec3 p(r.x, r.y, r.z); // contact to be processed at this iteration
			glm::mat3 Iinv = rbCurrentInertia_CPU[rigidBodyIndex];
			float m = rbMass_CPU[rigidBodyIndex];
			glm::vec3 v(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);
			glm::vec3 w(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);
			
			float mc = 1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, n), p), n); // active mass at current collision
			if (abs(mc) < 0.00001) mc = 1.f;
			float v_rel = glm::dot(v + glm::cross(w, p), n); // relative velocity at current contact
			float corrective_impulse = -(v_rel + contactBias[c]) / mc; // corrective impulse magnitude
#ifdef PRINT_COLLISIONS
			std::cout << "Iteration: " << k << std::endl;
			std::cout << "Contact: " << c << std::endl;
			std::cout << "Collision normal: (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
			std::cout << "Relative linear velocity: " << glm::dot(v, n) << std::endl;
			std::cout << "Relative angular velocity: " << glm::dot(glm::cross(w, p), n) << std::endl;
			std::cout << "Total relative velocity: " << v_rel << std::endl;
#endif
			//if (corrective_impulse < 0)
			//	std::cout << "Negative corrective impulse encountered: " << corrective_impulse << std::endl;

			float temporary_impulse = contactAccumulatedImpulse[c]; // make a copy of old accumulated impulse
			temporary_impulse = temporary_impulse + corrective_impulse; // add corrective impulse to accumulated impulse
			//clamp new accumulated impulse
			if (temporary_impulse < 0)
				temporary_impulse = 0; // allow no negative accumulated impulses
			else if (temporary_impulse > UPPER_BOUND)
				temporary_impulse = UPPER_BOUND; // max upper bound for accumulated impulse
			// compute difference between old and new impulse
			corrective_impulse = temporary_impulse - contactAccumulatedImpulse[c];
			contactAccumulatedImpulse[c] = temporary_impulse; // store new clamped accumulated impulse
			// apply new clamped corrective impulse difference to velocity
			glm::vec3 impulse_vector = corrective_impulse * n;
			v = v + impulse_vector / m;
			w = w + Iinv * glm::cross(p, impulse_vector);

			vel_CPU[rigidBodyIndex] = make_float4(v.x, v.y, v.z, 0);
			rbAngularVelocity_CPU[rigidBodyIndex] = make_float4(w.x, w.y, w.z, 0);

			vel_CPU[rigidBodyIndex].x = abs(vel_CPU[rigidBodyIndex].x) < 0.02 ? 0 : vel_CPU[rigidBodyIndex].x;
			vel_CPU[rigidBodyIndex].y = abs(vel_CPU[rigidBodyIndex].y) < 0.02 ? 0 : vel_CPU[rigidBodyIndex].y;
			vel_CPU[rigidBodyIndex].z = abs(vel_CPU[rigidBodyIndex].z) < 0.02 ? 0 : vel_CPU[rigidBodyIndex].z;

			rbAngularVelocity_CPU[rigidBodyIndex].x = abs(rbAngularVelocity_CPU[rigidBodyIndex].x) < 0.00008 ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].x;
			rbAngularVelocity_CPU[rigidBodyIndex].y = abs(rbAngularVelocity_CPU[rigidBodyIndex].y) < 0.00008 ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].y;
			rbAngularVelocity_CPU[rigidBodyIndex].z = abs(rbAngularVelocity_CPU[rigidBodyIndex].z) < 0.00008 ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].z;

#ifdef PRINT_COLLISIONS	
			std::cout << "Applied impulse: " << corrective_impulse << std::endl;
			std::cout << "New linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
			std::cout << "New angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;
			std::cout << std::endl;
#endif
		}
	}


	checkCudaErrors(cudaMemcpy(rbPositions, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbVelocities, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	delete rbCurrentInertia_CPU;
	delete rbMass_CPU;

	delete relative_CPU;
	delete position_CPU;
	delete normal_CPU;
	delete particlePosition_CPU;

	delete contactDistance_CPU;
	delete collidingParticleIndex_CPU;

	delete contactNormal;
	delete contactAccumulatedImpulse;
	delete contactRigidBody;
	delete contactPoint;
	delete contactBias;

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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//for (int iteration = 0; iteration < 8; iteration++) // hard-coded number of iterations for now
	//{
	//	for (int collision = 0; collision < collision_counter; collision++) // iterate over all collisions
	//	{
	//		float4 normal = contactNormal[collision];
	//		float4 point = contactPoint[collision];
	//		int rigidBodyIndex = contactRigidBody[collision];

	//		float4 r = point - CMs_CPU[rigidBodyIndex];
	//		glm::mat3 Iinv = rbCurrentInertia_CPU[rigidBodyIndex];
	//		float m = rbMass_CPU[rigidBodyIndex];
	//		float correctiveImpulse = computeImpulseMagnitude(vel_CPU[rigidBodyIndex], rbAngularVelocity_CPU[rigidBodyIndex], r, Iinv, m, normal);

	//		float accumulatedImpulse = contactAccumulatedImpulse[collision];
	//		// add corrective impulse to accumulated impulse
	//		accumulatedImpulse += correctiveImpulse;

	//		// clamp accumulated impulse
	//		if (accumulatedImpulse < 0)
	//			accumulatedImpulse = 0;
	//		if (accumulatedImpulse > 100)
	//			accumulatedImpulse = 100;

	//		// use difference of clamped and previous accumulated impulse
	//		float deltaImpulse = accumulatedImpulse - contactAccumulatedImpulse[collision];
	//		contactAccumulatedImpulse[collision] = accumulatedImpulse;

	//		/*std::cout << "Collision normal: (" << normal.x << ", " <<
	//			normal.y << ", " << normal.z << ", " << normal.w << ")" << std::endl;

	//		std::cout << "Corrective impulse: " << correctiveImpulse << std::endl;
	//		std::cout << "Clamped accumulated impulse: " << accumulatedImpulse << std::endl;

	//		std::cout << "V before impulse: (" << vel_CPU[rigidBodyIndex].x << ", " <<
	//			vel_CPU[rigidBodyIndex].y << ", " << vel_CPU[rigidBodyIndex].z << ")" << std::endl;*/
	//		

	//		float4 impulseVector = normal * deltaImpulse;
	//		vel_CPU[rigidBodyIndex] += impulseVector / m;
	//		vel_CPU[rigidBodyIndex].w = 0;

	//		/*std::cout << "V after impulse: (" << vel_CPU[rigidBodyIndex].x << ", " <<
	//			vel_CPU[rigidBodyIndex].y << ", " << vel_CPU[rigidBodyIndex].z << ")" << std::endl;*/

	//		// compute auxiliaries for angular impulse
	//		glm::vec3 rA(r.x, r.y, r.z);
	//		glm::vec3 impulseVectorGLM(impulseVector.x, impulseVector.y, impulseVector.z);
	//		
	//		/*std::cout << "W before impulse: (" << rbAngularVelocity_CPU[rigidBodyIndex].x << ", " <<
	//			rbAngularVelocity_CPU[rigidBodyIndex].y << ", " << rbAngularVelocity_CPU[rigidBodyIndex].z << ")" << std::endl;*/

	//		// apply angular impulse
	//		glm::vec3 AngularImpulse = Iinv * (glm::cross(rA, impulseVectorGLM));
	//		rbAngularVelocity_CPU[rigidBodyIndex] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);
	//		rbAngularVelocity_CPU[rigidBodyIndex].w = 0;

	//		/*std::cout << "W after impulse: (" << rbAngularVelocity_CPU[rigidBodyIndex].x << ", " <<
	//			rbAngularVelocity_CPU[rigidBodyIndex].y << ", " << rbAngularVelocity_CPU[rigidBodyIndex].z << ")" << std::endl;
	//		std::cout << std::endl;*/
	//	}
	//}

}

void ParticleSystem::Handle_Augmented_Reality_Collisions_Catto_Baumgarte_CPU(float deltaTime)
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *particlePosition_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(particlePosition_CPU, dPos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *position_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(position_CPU, staticPos, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *normal_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(normal_CPU, staticNorm, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));

	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	// pre-processing step
	// count total number of collisions

	int current_particle = 0;
	int collision_counter = 0;
	//static int iteration = 1;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				collision_counter++;
				/*if (iteration < 120)
				{
				std::ofstream file("collision_iteration.txt");
				file << iteration << std::endl;
				file.close();
				}*/
			}
			current_particle++;
		}
	}
	//iteration++;
	//#define PRINT_COLLISIONS
#ifdef PRINT_COLLISIONS
	std::cout << "Number of collisions: " << collision_counter << std::endl;
	std::ofstream file("collisions.txt");
#endif
	// initialize auxiliary contact variables
	float4 *contactNormal = new float4[collision_counter]; // store one normal per collision
	float4 *contactPoint = new float4[collision_counter]; // store one contact point per collision
	float *contactAccumulatedImpulse = new float[collision_counter]; // store the accumulated impulses per collision
	int *contactRigidBody = new int[collision_counter]; // index to colliding rigid body
	float *contactBias = new float[collision_counter]; // bias at each contact point
	memset(contactBias, 0, sizeof(float) * collision_counter);
	memset(contactNormal, 0, sizeof(float) * 4 * collision_counter);
	memset(contactPoint, 0, sizeof(float) * 4 * collision_counter);
	memset(contactAccumulatedImpulse, 0, sizeof(float) * collision_counter);
	memset(contactRigidBody, 0, sizeof(int) * collision_counter);

	collision_counter = 0;
	current_particle = 0;

	const float epsilon = 0.5f;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				contactNormal[collision_counter] = normal_CPU[particleIndex]; // scene's normal at collision point
				//contactNormal[collision_counter] = make_float4(0, 1, 0, 0);
				/*float4 cp, cn;
				findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
				position_CPU[particleIndex],
				m_params.particleRadius,
				m_params.particleRadius,
				cp, cn);*/

				//contactPoint[collision_counter] = position_CPU[particleIndex]; // exact contact point
				contactPoint[collision_counter] = relative_CPU[current_particle];
				contactRigidBody[collision_counter] = index;
				float3 v = make_float3(vel_CPU[index].x, vel_CPU[index].y, vel_CPU[index].z);
				float3 w = make_float3(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);
				float v_rel = dot(v + cross(w, make_float3(relative_CPU[current_particle])), make_float3(contactNormal[collision_counter])); // relative velocity at current contact
				contactBias[collision_counter] = epsilon * v_rel;
#ifdef PRINT_COLLISIONS
				std::cout << "Collision #" << collision_counter + 1 << " initial bias: " << contactBias[collision_counter] << std::endl;
				float4 r = relative_CPU[current_particle];//position_CPU[particleIndex] - CMs_CPU[index];
				//std::cout << "Collision #" << collision_counter + 1 << ": (" << r.x << ", " << r.y << ", " << r.z << ")" << std::endl;
				file << r.x << " " << r.y << " " << r.z << " " << std::endl;
#endif
				collision_counter++;
			}
			current_particle++;
		}
	}
#ifdef PRINT_COLLISIONS
	file.close();
#endif
	// solve contacts using SIS
	
	const int iterations = 8; // number of iterations per simulation step
	const int UPPER_BOUND = 100; // upper bound for accumulated impulse
	for (int k = 0; k < iterations; k++)
	{
		for (int c = 0; c < collision_counter; c++)
		{
			glm::vec3 n(contactNormal[c].x, contactNormal[c].y, contactNormal[c].z); // collision normal
			//glm::vec3 n(0, sqrt(2.f) / 2, sqrt(2.f) / 2);
			//glm::vec3 n(0, 1, 0);
			float4 point = contactPoint[c];
			int rigidBodyIndex = contactRigidBody[c];
			float4 r = point;
			glm::vec3 p(r.x, r.y, r.z); // contact to be processed at this iteration
			glm::mat3 Iinv = rbCurrentInertia_CPU[rigidBodyIndex];
			float m = rbMass_CPU[rigidBodyIndex];
			glm::vec3 v(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);
			glm::vec3 w(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);

			float mc = 1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, n), p), n); // active mass at current collision
			if (abs(mc) < 0.00001) mc = 1.f;
			float v_rel = glm::dot(v + glm::cross(w, p), n); // relative velocity at current contact
			float corrective_impulse = -(v_rel + contactBias[c]) / mc; // corrective impulse magnitude
#ifdef PRINT_COLLISIONS
			std::cout << "Iteration: " << k << std::endl;
			std::cout << "Contact: " << c << std::endl;
			std::cout << "Collision normal: (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
			std::cout << "Relative linear velocity: " << glm::dot(v, n) << std::endl;
			std::cout << "Relative angular velocity: " << glm::dot(glm::cross(w, p), n) << std::endl;
			std::cout << "Total relative velocity: " << v_rel << std::endl;
#endif
			//if (corrective_impulse < 0)
			//	std::cout << "Negative corrective impulse encountered: " << corrective_impulse << std::endl;

			float temporary_impulse = contactAccumulatedImpulse[c]; // make a copy of old accumulated impulse
			temporary_impulse = temporary_impulse + corrective_impulse; // add corrective impulse to accumulated impulse
			//clamp new accumulated impulse
			if (temporary_impulse < 0)
				temporary_impulse = 0; // allow no negative accumulated impulses
			else if (temporary_impulse > UPPER_BOUND)
				temporary_impulse = UPPER_BOUND; // max upper bound for accumulated impulse
			// compute difference between old and new impulse
			corrective_impulse = temporary_impulse - contactAccumulatedImpulse[c];
			contactAccumulatedImpulse[c] = temporary_impulse; // store new clamped accumulated impulse
			// apply new clamped corrective impulse difference to velocity
			glm::vec3 impulse_vector = corrective_impulse * n;
			v = v + impulse_vector / m;
			w = w + Iinv * glm::cross(p, impulse_vector);

			vel_CPU[rigidBodyIndex] = make_float4(v.x, v.y, v.z, 0);
			rbAngularVelocity_CPU[rigidBodyIndex] = make_float4(w.x, w.y, w.z, 0);

			const float Baumgarte_factor = 0.6 / deltaTime;
			CMs_CPU[rigidBodyIndex] = CMs_CPU[rigidBodyIndex] * exp(-Baumgarte_factor*deltaTime) - contactBias[c] / Baumgarte_factor;
#ifdef PRINT_COLLISIONS	
			std::cout << "Applied impulse: " << corrective_impulse << std::endl;
			std::cout << "New linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
			std::cout << "New angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;
			std::cout << std::endl;
#endif
		}
	}


	checkCudaErrors(cudaMemcpy(rbPositions, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbVelocities, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	delete rbCurrentInertia_CPU;
	delete rbMass_CPU;

	delete relative_CPU;
	delete position_CPU;
	delete normal_CPU;
	delete particlePosition_CPU;

	delete contactDistance_CPU;
	delete collidingParticleIndex_CPU;

	delete contactNormal;
	delete contactAccumulatedImpulse;
	delete contactRigidBody;
	delete contactPoint;
	delete contactBias;

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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::GatherRigidBodyCollisions()
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	
	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *particlePosition_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(particlePosition_CPU, dPos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	// copy contact info to CPU - one contact per particle
	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingRigidBodyIndex_CPU = new int[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingRigidBodyIndex_CPU, collidingRigidBodyIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));


	int current_particle = 0;
	const float epsilon = m_params.RBrestitution;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			int rigidBodyIndex = collidingRigidBodyIndex_CPU[current_particle];
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				int particleIndex = collidingParticleIndex_CPU[current_particle];

				float4 cp, cn;
				findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
					CMs_CPU[rigidBodyIndex] + relative_CPU[particleIndex],
					m_params.particleRadius,
					m_params.particleRadius,
					cp, cn);

				ContactNormal.push_back(cn);
				ContactPoint.push_back(cp);
				ContactRigidBody_1.push_back(index);
				ContactRigidBody_2.push_back(rigidBodyIndex);

				float3 v1 = make_float3(vel_CPU[index].x, vel_CPU[index].y, vel_CPU[index].z);
				float3 w1 = make_float3(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);

				float3 v2 = make_float3(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);
				float3 w2 = make_float3(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);

				float v_rel = dot(v1 + cross(w1, make_float3(relative_CPU[current_particle])), make_float3(cn)) -
					dot(v2 + cross(w2, make_float3(relative_CPU[particleIndex])), make_float3(cn)); // relative velocity at current contact

				ContactBias.push_back(epsilon * v_rel);
				ContactAccumulatedImpulse.push_back(0);
				ContactAccumulatedFriction.push_back(0);
				ContactAccumulatedFriction_2.push_back(0);
			}
			current_particle++;
		}
	}

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	
	delete relative_CPU;
	delete particlePosition_CPU;

	delete contactDistance_CPU;
	delete collidingParticleIndex_CPU;
	delete collidingRigidBodyIndex_CPU;
}

void ParticleSystem::GatherAugmentedRealityCollisions()
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));

	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *normal_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(normal_CPU, staticNorm, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));

	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];
	float4 *contact_normal_CPU = new float4[m_numParticles];
	
	checkCudaErrors(cudaMemcpy(contact_normal_CPU, contact_normal, m_numParticles * 4 * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	// pre-processing step
	// count total number of collisions
	int current_particle = 0;
	const float epsilon = m_params.ARrestitution;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				ContactNormal.push_back(normal_CPU[particleIndex]); // scene's normal at collision point
				//ContactNormal.push_back(contact_normal_CPU[current_particle]);
				ContactPoint.push_back(relative_CPU[current_particle] + CMs_CPU[index]);
				ContactRigidBody_1.push_back(index);
				ContactRigidBody_2.push_back(-1);
				float3 v = make_float3(vel_CPU[index].x, vel_CPU[index].y, vel_CPU[index].z);
				float3 w = make_float3(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);
				float v_rel = dot(v + cross(w, make_float3(relative_CPU[current_particle])), make_float3(normal_CPU[particleIndex])); // relative velocity at current contact
				//float v_rel = dot(v + cross(w, make_float3(relative_CPU[current_particle])), make_float3(contact_normal_CPU[current_particle]));
				ContactBias.push_back(epsilon * v_rel);
				ContactAccumulatedImpulse.push_back(0);
				ContactAccumulatedFriction.push_back(0);
				ContactAccumulatedFriction_2.push_back(0);
			}
			current_particle++;
		}
	}
#ifdef PRINT_COLLISIONS
	file.close();
#endif
	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;

	delete relative_CPU;
	delete normal_CPU;
	delete contact_normal_CPU;

	delete contactDistance_CPU;
	delete collidingParticleIndex_CPU;
}

void ParticleSystem::SequentialImpulseSolver()
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	int current_particle = 0;
	const float linear_bound = 0.02;
	const float angular_bound = 0.00002;
	const int iterations = 18; // number of iterations per simulation step
	const int UPPER_BOUND = 100; // upper bound for accumulated impulse
	const int collision_counter = ContactNormal.size();

	// solve contacts using SIS
	for (int k = 0; k < iterations; k++)
	{
		for (int c = 0; c < collision_counter; c++)
		{
			glm::vec3 n(ContactNormal[c].x, ContactNormal[c].y, ContactNormal[c].z); // collision normal
			float4 point = ContactPoint[c];

			int rigidBodyIndex = ContactRigidBody_1[c];
			int rigidBodyIndex2 = ContactRigidBody_2[c];
			if (rigidBodyIndex2 == -1)
			{
				float4 r = point - CMs_CPU[rigidBodyIndex];
				glm::vec3 p(r.x, r.y, r.z); // contact to be processed at this iteration
				glm::mat3 Iinv = rbCurrentInertia_CPU[rigidBodyIndex];
				float m = rbMass_CPU[rigidBodyIndex];
				glm::vec3 v(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);
				glm::vec3 w(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);

				float mc = 1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, n), p), n); // active mass at current collision
				if (abs(mc) < 0.00001) mc = 1.f;
				float v_rel = glm::dot(v + glm::cross(w, p), n); // relative velocity at current contact
				float corrective_impulse = -(v_rel + ContactBias[c]) / mc; // corrective impulse magnitude
#ifdef PRINT_COLLISIONS
				std::cout << "Iteration: " << k << std::endl;
				std::cout << "Contact: " << c << std::endl;
				std::cout << "Collision normal: (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
				std::cout << "Relative linear velocity: " << glm::dot(v, n) << std::endl;
				std::cout << "Relative angular velocity: " << glm::dot(glm::cross(w, p), n) << std::endl;
				std::cout << "Total relative velocity: " << v_rel << std::endl;
#endif
				float temporary_impulse = ContactAccumulatedImpulse[c]; // make a copy of old accumulated impulse
				temporary_impulse = temporary_impulse + corrective_impulse; // add corrective impulse to accumulated impulse
				//clamp new accumulated impulse
				if (temporary_impulse < 0)
					temporary_impulse = 0; // allow no negative accumulated impulses
				else if (temporary_impulse > UPPER_BOUND)
					temporary_impulse = UPPER_BOUND; // max upper bound for accumulated impulse
				// compute difference between old and new impulse
				corrective_impulse = temporary_impulse - ContactAccumulatedImpulse[c];
				ContactAccumulatedImpulse[c] = temporary_impulse; // store new clamped accumulated impulse

				// compute friction
				//const float friction_bound = friction_coefficient * temporary_impulse;
				const float friction_bound = m_params.ARfriction * mc * abs(m_params.gravity.y);
				glm::vec3 vel = v + glm::cross(w, p);

				glm::vec3 tangential_direction(1, 0, 0);
				if (abs(tangential_direction.x - n.x) < 0.0001 && abs(tangential_direction.x - n.x) < 0.0001 && abs(tangential_direction.x - n.x) < 0.0001)
					tangential_direction = glm::vec3(0, 1, 0);
				else if(abs(tangential_direction.x + n.x) < 0.0001 && abs(tangential_direction.x + n.x) < 0.0001 && abs(tangential_direction.x + n.x) < 0.0001)
					tangential_direction = glm::vec3(0, 1, 0);

				tangential_direction = tangential_direction - glm::dot(tangential_direction, n) * n;
				tangential_direction = glm::normalize(tangential_direction);
				float k_t = (1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, tangential_direction), p), tangential_direction));
				float corrective_friction = -glm::dot(vel, tangential_direction) / k_t;


				float temporary_friction = ContactAccumulatedFriction[c];
				temporary_friction = temporary_friction + corrective_friction;
				if (temporary_friction < -friction_bound)
					temporary_friction = -friction_bound; // allow no negative accumulated impulses
				else if (temporary_friction > friction_bound)
					temporary_friction = friction_bound; // max upper bound for accumulated impulse
				corrective_friction = temporary_friction - ContactAccumulatedFriction[c];
				ContactAccumulatedFriction[c] = temporary_friction; // store new clamped accumulated impulse


				glm::vec3 tangential_direction_2 = glm::cross(tangential_direction, n);
				tangential_direction_2 = glm::normalize(tangential_direction_2);
				float k_t_2 = (1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, tangential_direction_2), p), tangential_direction_2));
				float corrective_friction_2 = -glm::dot(vel, tangential_direction_2) / k_t_2;
				float temporary_friction_2 = ContactAccumulatedFriction_2[c];
				temporary_friction_2 = temporary_friction_2 + corrective_friction_2;
				if (temporary_friction_2 < -friction_bound)
					temporary_friction_2 = -friction_bound; // allow no negative accumulated impulses
				else if (temporary_friction_2 > friction_bound)
					temporary_friction_2 = friction_bound; // max upper bound for accumulated impulse
				corrective_friction_2 = temporary_friction_2 - ContactAccumulatedFriction_2[c];
				ContactAccumulatedFriction_2[c] = temporary_friction_2; // store new clamped accumulated impulse

				glm::vec3 normal_impulse = corrective_impulse * n;
				glm::vec3 friction_impulse_1 = corrective_friction * tangential_direction;
				glm::vec3 friction_impulse_2 = corrective_friction_2 * tangential_direction_2;

				glm::vec3 impulse_vector = corrective_impulse * n + corrective_friction * tangential_direction +
					corrective_friction_2 * tangential_direction_2;
				v = v + impulse_vector / m;
				w = w + Iinv * glm::cross(p, impulse_vector);

				vel_CPU[rigidBodyIndex] = make_float4(v.x, v.y, v.z, 0);
				rbAngularVelocity_CPU[rigidBodyIndex] = make_float4(w.x, w.y, w.z, 0);

				//vel_CPU[rigidBodyIndex].x = abs(vel_CPU[rigidBodyIndex].x) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].x;
				//vel_CPU[rigidBodyIndex].y = abs(vel_CPU[rigidBodyIndex].y) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].y;
				//vel_CPU[rigidBodyIndex].z = abs(vel_CPU[rigidBodyIndex].z) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].z;

				//rbAngularVelocity_CPU[rigidBodyIndex].x = abs(rbAngularVelocity_CPU[rigidBodyIndex].x) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].x;
				//rbAngularVelocity_CPU[rigidBodyIndex].y = abs(rbAngularVelocity_CPU[rigidBodyIndex].y) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].y;
				//rbAngularVelocity_CPU[rigidBodyIndex].z = abs(rbAngularVelocity_CPU[rigidBodyIndex].z) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].z;

#ifdef PRINT_COLLISIONS	
				std::cout << "Applied impulse: " << corrective_impulse << std::endl;
				std::cout << "New linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
				std::cout << "New angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;
				std::cout << std::endl;
#endif
			}
			else
			{
				float4 r1 = point - CMs_CPU[rigidBodyIndex];
				glm::vec3 p1(r1.x, r1.y, r1.z); // contact to be processed at this iteration
				glm::mat3 Iinv1 = rbCurrentInertia_CPU[rigidBodyIndex];
				float m1 = rbMass_CPU[rigidBodyIndex];
				glm::vec3 v1(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);
				glm::vec3 w1(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);

				float4 r2 = point - CMs_CPU[rigidBodyIndex2];
				glm::vec3 p2(r2.x, r2.y, r2.z); // contact to be processed at this iteration
				glm::mat3 Iinv2 = rbCurrentInertia_CPU[rigidBodyIndex2];
				float m2 = rbMass_CPU[rigidBodyIndex2];
				glm::vec3 v2(vel_CPU[rigidBodyIndex2].x, vel_CPU[rigidBodyIndex2].y, vel_CPU[rigidBodyIndex2].z);
				glm::vec3 w2(rbAngularVelocity_CPU[rigidBodyIndex2].x, rbAngularVelocity_CPU[rigidBodyIndex2].y, rbAngularVelocity_CPU[rigidBodyIndex2].z);

				float mc = 1 / m1 + 1 / m2 + glm::dot(glm::cross(Iinv1 * glm::cross(p1, n), p1), n) + 
					glm::dot(glm::cross(Iinv2 * glm::cross(p2, n), p2), n); // active mass at current collision
				if (abs(mc) < 0.00001) mc = 1.f;
				float v_rel = glm::dot(v1 + glm::cross(w1, p1), n) - glm::dot(v2 + glm::cross(w2, p2), n); // relative velocity at current contact
				float corrective_impulse = -(v_rel + ContactBias[c]) / mc; // corrective impulse magnitude
#ifdef PRINT_COLLISIONS
				std::cout << "Iteration: " << k << std::endl;
				std::cout << "Contact: " << c << std::endl;
				std::cout << "Collision normal: (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
				std::cout << "Relative linear velocity: " << glm::dot(v, n) << std::endl;
				std::cout << "Relative angular velocity: " << glm::dot(glm::cross(w, p), n) << std::endl;
				std::cout << "Total relative velocity: " << v_rel << std::endl;
#endif

				float temporary_impulse = ContactAccumulatedImpulse[c]; // make a copy of old accumulated impulse
				temporary_impulse = temporary_impulse + corrective_impulse; // add corrective impulse to accumulated impulse
				//clamp new accumulated impulse
				if (temporary_impulse < 0)
					temporary_impulse = 0; // allow no negative accumulated impulses
				else if (temporary_impulse > UPPER_BOUND)
					temporary_impulse = UPPER_BOUND; // max upper bound for accumulated impulse
				// compute difference between old and new impulse
				corrective_impulse = temporary_impulse - ContactAccumulatedImpulse[c];
				ContactAccumulatedImpulse[c] = temporary_impulse; // store new clamped accumulated impulse
	
				const float friction_bound = m_params.RBfriction * mc * abs(m_params.gravity.y);
				glm::vec3 vel = -v1 + glm::cross(w1, p1) + v2 + glm::cross(w2, p2);

				glm::vec3 tangential_direction(1, 0, 0);
				if (abs(tangential_direction.x - n.x) < 0.0001 && abs(tangential_direction.x - n.x) < 0.0001 && abs(tangential_direction.x - n.x) < 0.0001)
					tangential_direction = glm::vec3(0, 1, 0);
				tangential_direction = tangential_direction - glm::dot(tangential_direction, n) * n;
				tangential_direction = glm::normalize(tangential_direction);
				float k_t = (1 / m1 + glm::dot(glm::cross(Iinv1 * glm::cross(p1, tangential_direction), p1), tangential_direction)) + 
					(1 / m2 + glm::dot(glm::cross(Iinv2 * glm::cross(p2, tangential_direction), p2), tangential_direction));
				float corrective_friction = -glm::dot(vel, tangential_direction) / k_t;


				float temporary_friction = ContactAccumulatedFriction[c];
				temporary_friction = temporary_friction + corrective_friction;
				if (temporary_friction < -friction_bound)
					temporary_friction = -friction_bound; // allow no negative accumulated impulses
				else if (temporary_friction > friction_bound)
					temporary_friction = friction_bound; // max upper bound for accumulated impulse
				corrective_friction = temporary_friction - ContactAccumulatedFriction[c];
				ContactAccumulatedFriction[c] = temporary_friction; // store new clamped accumulated impulse

				glm::vec3 tangential_direction_2 = glm::cross(tangential_direction, n);
				tangential_direction_2 = glm::normalize(tangential_direction_2);
				float k_t_2 = (1 / m1 + glm::dot(glm::cross(Iinv1 * glm::cross(p1, tangential_direction_2), p1), tangential_direction_2)) +
					(1 / m2 + glm::dot(glm::cross(Iinv2 * glm::cross(p2, tangential_direction_2), p2), tangential_direction_2));
				float corrective_friction_2 = -glm::dot(vel, tangential_direction_2) / k_t_2;
				float temporary_friction_2 = ContactAccumulatedFriction_2[c];
				temporary_friction_2 = temporary_friction_2 + corrective_friction_2;
				if (temporary_friction_2 < -friction_bound)
					temporary_friction_2 = -friction_bound; // allow no negative accumulated impulses
				else if (temporary_friction_2 > friction_bound)
					temporary_friction_2 = friction_bound; // max upper bound for accumulated impulse
				corrective_friction_2 = temporary_friction_2 - ContactAccumulatedFriction_2[c];
				ContactAccumulatedFriction_2[c] = temporary_friction_2; // store new clamped accumulated impulse

				// apply new clamped corrective impulse difference to velocity
				glm::vec3 normal_impulse = corrective_impulse * n;
				glm::vec3 friction_impulse_1 = corrective_friction * tangential_direction;
				glm::vec3 friction_impulse_2 = corrective_friction_2 * tangential_direction_2;
				glm::vec3 impulse_vector = normal_impulse;// +friction_impulse_1 + friction_impulse_2;
				

				//std::cout << "Relative velocity (" << vel.x << ", " << vel.y << ", " << vel.z << ")" << std::endl;
				//std::cout << "Velocity 1 (" << (v1 + glm::cross(w1, p1)).x << ", " << (v1 + glm::cross(w1, p1)).y << ", " << (v1 + glm::cross(w1, p1)).z << ")" << std::endl;
				//std::cout << "Velocity 2 (" << (v2 + glm::cross(w2, p2)).x << ", " << (v2 + glm::cross(w2, p2)).y << ", " << (v2 + glm::cross(w2, p2)).z << ")" << std::endl;
				//std::cout << "Normal (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
				//std::cout << "Normal impulse (" << normal_impulse.x << ", " << normal_impulse.y << ", " << normal_impulse.z << ")" << std::endl;
				//std::cout << "Tangential 1 (" << tangential_direction.x << ", " << tangential_direction.y << ", " << tangential_direction.z << ")" << std::endl;
				//std::cout << "Friction impulse 1 (" << friction_impulse_1.x << ", " << friction_impulse_1.y << ", " << friction_impulse_1.z << ")" << std::endl;
				//std::cout << "Tangential 2 (" << tangential_direction_2.x << ", " << tangential_direction_2.y << ", " << tangential_direction_2.z << ")" << std::endl;
				//std::cout << "Friction impulse 2 (" << friction_impulse_2.x << ", " << friction_impulse_2.y << ", " << friction_impulse_2.z << ")" << std::endl;
				//std::cout << std::endl;


				//glm::vec3 impulse_vector = corrective_impulse * n;
				
				v1 = v1 + impulse_vector / m1;
				w1 = w1 + Iinv1 * glm::cross(p1, impulse_vector);

				vel_CPU[rigidBodyIndex] = make_float4(v1.x, v1.y, v1.z, 0);
				rbAngularVelocity_CPU[rigidBodyIndex] = make_float4(w1.x, w1.y, w1.z, 0);

				v2 = v2 - impulse_vector / m2;
				w2 = w2 - Iinv2 * glm::cross(p2, impulse_vector);
				vel_CPU[rigidBodyIndex2] = make_float4(v2.x, v2.y, v2.z, 0);
				rbAngularVelocity_CPU[rigidBodyIndex2] = make_float4(w2.x, w2.y, w2.z, 0);

				//vel_CPU[rigidBodyIndex].x = abs(vel_CPU[rigidBodyIndex].x) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].x;
				//vel_CPU[rigidBodyIndex].y = abs(vel_CPU[rigidBodyIndex].y) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].y;
				//vel_CPU[rigidBodyIndex].z = abs(vel_CPU[rigidBodyIndex].z) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].z;

				//rbAngularVelocity_CPU[rigidBodyIndex].x = abs(rbAngularVelocity_CPU[rigidBodyIndex].x) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].x;
				//rbAngularVelocity_CPU[rigidBodyIndex].y = abs(rbAngularVelocity_CPU[rigidBodyIndex].y) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].y;
				//rbAngularVelocity_CPU[rigidBodyIndex].z = abs(rbAngularVelocity_CPU[rigidBodyIndex].z) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].z;
				//
				//vel_CPU[rigidBodyIndex2].x = abs(vel_CPU[rigidBodyIndex2].x) < linear_bound ? 0 : vel_CPU[rigidBodyIndex2].x;
				//vel_CPU[rigidBodyIndex2].y = abs(vel_CPU[rigidBodyIndex2].y) < linear_bound ? 0 : vel_CPU[rigidBodyIndex2].y;
				//vel_CPU[rigidBodyIndex2].z = abs(vel_CPU[rigidBodyIndex2].z) < linear_bound ? 0 : vel_CPU[rigidBodyIndex2].z;

				//rbAngularVelocity_CPU[rigidBodyIndex2].x = abs(rbAngularVelocity_CPU[rigidBodyIndex2].x) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex2].x;
				//rbAngularVelocity_CPU[rigidBodyIndex2].y = abs(rbAngularVelocity_CPU[rigidBodyIndex2].y) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex2].y;
				//rbAngularVelocity_CPU[rigidBodyIndex2].z = abs(rbAngularVelocity_CPU[rigidBodyIndex2].z) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex2].z;




#ifdef PRINT_COLLISIONS	
				std::cout << "Applied impulse: " << corrective_impulse << std::endl;
				std::cout << "New linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
				std::cout << "New angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;
				std::cout << std::endl;
#endif
			}
		}
	}
	checkCudaErrors(cudaMemcpy(rbPositions, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbVelocities, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	delete rbCurrentInertia_CPU;
	delete rbMass_CPU;

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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

void ParticleSystem::Handle_Augmented_Reality_Collisions_Catto_Friction_CPU()
{
	// copy rigid body variables to CPU
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body

	checkCudaErrors(cudaMemcpy(CMs_CPU, rbPositions, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, rbVelocities, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	// copy particle variables to CPU
	float4 *relative_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(relative_CPU, relativePos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *particlePosition_CPU = new float4[m_numParticles];
	checkCudaErrors(cudaMemcpy(particlePosition_CPU, dPos, m_numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *position_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(position_CPU, staticPos, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));
	float4 *normal_CPU = new float4[numberOfRangeData];
	checkCudaErrors(cudaMemcpy(normal_CPU, staticNorm, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));

	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	// pre-processing step
	// count total number of collisions

	int current_particle = 0;
	int collision_counter = 0;
	//static int iteration = 1;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				collision_counter++;
				/*if (iteration < 120)
				{
				std::ofstream file("collision_iteration.txt");
				file << iteration << std::endl;
				file.close();
				}*/
			}
			current_particle++;
		}
	}
	//iteration++;
	//#define PRINT_COLLISIONS
#ifdef PRINT_COLLISIONS
	std::cout << "Number of collisions: " << collision_counter << std::endl;
	std::ofstream file("collisions.txt");
#endif
	// initialize auxiliary contact variables
	float4 *contactNormal = new float4[collision_counter]; // store one normal per collision
	float4 *contactPoint = new float4[collision_counter]; // store one contact point per collision
	float *contactAccumulatedImpulse = new float[collision_counter]; // store the accumulated impulses per collision
	float *contactAccumulatedFriction = new float[collision_counter]; // store the accumulated impulses per collision
	float *contactAccumulatedFriction_2 = new float[collision_counter]; // store the accumulated impulses per collision
	int *contactRigidBody = new int[collision_counter]; // index to colliding rigid body
	float *contactBias = new float[collision_counter]; // bias at each contact point
	memset(contactBias, 0, sizeof(float) * collision_counter);
	memset(contactNormal, 0, sizeof(float) * 4 * collision_counter);
	memset(contactPoint, 0, sizeof(float) * 4 * collision_counter);
	memset(contactAccumulatedImpulse, 0, sizeof(float) * collision_counter);
	memset(contactAccumulatedFriction, 0, sizeof(float) * collision_counter);
	memset(contactAccumulatedFriction_2, 0, sizeof(float) * collision_counter);
	memset(contactRigidBody, 0, sizeof(int) * collision_counter);

	collision_counter = 0;
	current_particle = 0;

	const float epsilon = 0.5f;
	const float friction_coefficient = 0.3;
	const float linear_bound = 0.02;
	const float angular_bound = 0.00002;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				contactNormal[collision_counter] = normal_CPU[particleIndex]; // scene's normal at collision point
				//contactNormal[collision_counter] = make_float4(0, 1, 0, 0);
				/*float4 cp, cn;
				findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
				position_CPU[particleIndex],
				m_params.particleRadius,
				m_params.particleRadius,
				cp, cn);*/

				//contactPoint[collision_counter] = position_CPU[particleIndex]; // exact contact point
				contactPoint[collision_counter] = relative_CPU[current_particle] + CMs_CPU[index];
				contactRigidBody[collision_counter] = index;
				float3 v = make_float3(vel_CPU[index].x, vel_CPU[index].y, vel_CPU[index].z);
				float3 w = make_float3(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);
				float v_rel = dot(v + cross(w, make_float3(relative_CPU[current_particle])), make_float3(contactNormal[collision_counter])); // relative velocity at current contact
				contactBias[collision_counter] = epsilon * v_rel;
#ifdef PRINT_COLLISIONS
				std::cout << "Collision #" << collision_counter + 1 << " initial bias: " << contactBias[collision_counter] << std::endl;
				float4 r = relative_CPU[current_particle];//position_CPU[particleIndex] - CMs_CPU[index];
				//std::cout << "Collision #" << collision_counter + 1 << ": (" << r.x << ", " << r.y << ", " << r.z << ")" << std::endl;
				file << r.x << " " << r.y << " " << r.z << " " << std::endl;
#endif
				collision_counter++;
			}
			current_particle++;
		}
	}
#ifdef PRINT_COLLISIONS
	file.close();
#endif
	// solve contacts using SIS
	
	const int iterations = 8; // number of iterations per simulation step
	const int UPPER_BOUND = 100; // upper bound for accumulated impulse
	for (int k = 0; k < iterations; k++)
	{
		for (int c = 0; c < collision_counter; c++)
		{
			glm::vec3 n(contactNormal[c].x, contactNormal[c].y, contactNormal[c].z); // collision normal
			//glm::vec3 n(0, sqrt(2.f) / 2, sqrt(2.f) / 2);
			//glm::vec3 n(0, 1, 0);
			float4 point = contactPoint[c];
			int rigidBodyIndex = contactRigidBody[c];
			float4 r = point - CMs_CPU[rigidBodyIndex];
			glm::vec3 p(r.x, r.y, r.z); // contact to be processed at this iteration
			glm::mat3 Iinv = rbCurrentInertia_CPU[rigidBodyIndex];
			float m = rbMass_CPU[rigidBodyIndex];
			glm::vec3 v(vel_CPU[rigidBodyIndex].x, vel_CPU[rigidBodyIndex].y, vel_CPU[rigidBodyIndex].z);
			glm::vec3 w(rbAngularVelocity_CPU[rigidBodyIndex].x, rbAngularVelocity_CPU[rigidBodyIndex].y, rbAngularVelocity_CPU[rigidBodyIndex].z);

			float mc = 1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, n), p), n); // active mass at current collision
			if (abs(mc) < 0.00001) mc = 1.f;
			float v_rel = glm::dot(v + glm::cross(w, p), n); // relative velocity at current contact
			float corrective_impulse = -(v_rel + contactBias[c]) / mc; // corrective impulse magnitude
#ifdef PRINT_COLLISIONS
			std::cout << "Iteration: " << k << std::endl;
			std::cout << "Contact: " << c << std::endl;
			std::cout << "Collision normal: (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
			std::cout << "Relative linear velocity: " << glm::dot(v, n) << std::endl;
			std::cout << "Relative angular velocity: " << glm::dot(glm::cross(w, p), n) << std::endl;
			std::cout << "Total relative velocity: " << v_rel << std::endl;
#endif
			//if (corrective_impulse < 0)
			//	std::cout << "Negative corrective impulse encountered: " << corrective_impulse << std::endl;

			float temporary_impulse = contactAccumulatedImpulse[c]; // make a copy of old accumulated impulse
			temporary_impulse = temporary_impulse + corrective_impulse; // add corrective impulse to accumulated impulse
			//clamp new accumulated impulse
			if (temporary_impulse < 0)
				temporary_impulse = 0; // allow no negative accumulated impulses
			else if (temporary_impulse > UPPER_BOUND)
				temporary_impulse = UPPER_BOUND; // max upper bound for accumulated impulse
			// compute difference between old and new impulse
			corrective_impulse = temporary_impulse - contactAccumulatedImpulse[c];
			contactAccumulatedImpulse[c] = temporary_impulse; // store new clamped accumulated impulse

			// compute friction
			//const float friction_bound = friction_coefficient * temporary_impulse;
			const float friction_bound = friction_coefficient * mc * abs(m_params.gravity.y);
			//std::cout << "Friction bound: " << friction_bound << std::endl;
			glm::vec3 vel = v + glm::cross(w, p);

			/*glm::vec3 tanVel = vel - (glm::dot(vel, n) * n);
			glm::vec3 tangential_direction = glm::normalize(tanVel);*/
			glm::vec3 tangential_direction(1, 0, 0);
			if (abs(tangential_direction.x - n.x) < 0.0001 && abs(tangential_direction.x - n.x) < 0.0001 && abs(tangential_direction.x - n.x) < 0.0001)
				tangential_direction = glm::vec3(0, 1, 0);
			tangential_direction = tangential_direction - glm::dot(tangential_direction, n) * n;
			float k_t = (1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, tangential_direction), p), tangential_direction));
			float corrective_friction = -glm::dot(vel, tangential_direction) / k_t;
			/*if (abs(tanVel.x) < 0.0001 && abs(tanVel.y) < 0.0001 && abs(tanVel.z) < 0.0001)
				corrective_friction = 0;*/
			float temporary_friction = contactAccumulatedFriction[c];
			temporary_friction = temporary_friction + corrective_friction;
			//std::cout << "Temporary friction 1: " << temporary_friction << std::endl;
			if (temporary_friction < -friction_bound)
				temporary_friction = -friction_bound; // allow no negative accumulated impulses
			else if (temporary_friction > friction_bound)
				temporary_friction = friction_bound; // max upper bound for accumulated impulse
			corrective_friction = temporary_friction - contactAccumulatedFriction[c];
			contactAccumulatedFriction[c] = temporary_friction; // store new clamped accumulated impulse


			glm::vec3 tangential_direction_2 = glm::cross(tangential_direction, n);
			float k_t_2 = (1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, tangential_direction_2), p), tangential_direction_2));
			float corrective_friction_2 = -glm::dot(vel, tangential_direction_2) / k_t_2;
			/*if (abs(tanVel.x) < 0.0001 && abs(tanVel.y) < 0.0001 && abs(tanVel.z) < 0.0001)
				corrective_friction_2 = 0;*/
			float temporary_friction_2 = contactAccumulatedFriction_2[c];
			temporary_friction_2 = temporary_friction_2 + corrective_friction_2;
			//std::cout << "Temporary friction 2: " << temporary_friction_2 << std::endl;
			if (temporary_friction_2 < -friction_bound)
				temporary_friction_2 = -friction_bound; // allow no negative accumulated impulses
			else if (temporary_friction_2 > friction_bound)
				temporary_friction_2 = friction_bound; // max upper bound for accumulated impulse
			corrective_friction_2 = temporary_friction_2 - contactAccumulatedFriction_2[c];
			contactAccumulatedFriction_2[c] = temporary_friction_2; // store new clamped accumulated impulse

			//corrective_impulse += corrective_friction + corrective_friction_2;
			glm::vec3 normal_impulse = corrective_impulse * n;
			glm::vec3 friction_impulse_1 = corrective_friction * tangential_direction;
			glm::vec3 friction_impulse_2 = corrective_friction_2 * tangential_direction_2;

			/*std::cout << "Normal (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
			std::cout << "Tangential 1 (" << tangential_direction.x << ", " << tangential_direction.y << ", " << tangential_direction.z << ")" << std::endl;
			std::cout << "Tangential 2 (" << tangential_direction_2.x << ", " << tangential_direction_2.y << ", " << tangential_direction_2.z << ")" << std::endl;
			std::cout << "Velocity (" << vel.x << ", " << vel.y << ", " << vel.z << ")" << std::endl;
			std::cout << "Normal impulse (" << normal_impulse.x << ", " << normal_impulse.y << ", " << normal_impulse.z << ")" << std::endl;
			std::cout << "Friction impulse 1 (" << friction_impulse_1.x << ", " << friction_impulse_1.y << ", " << friction_impulse_1.z << ")" << std::endl;
			std::cout << "Friction impulse 2 (" << friction_impulse_2.x << ", " << friction_impulse_2.y << ", " << friction_impulse_2.z << ")" << std::endl;
			std::cout << std::endl;*/
			// apply new clamped corrective impulse difference to velocity
			
			glm::vec3 impulse_vector = corrective_impulse * n + corrective_friction * tangential_direction + 
				corrective_friction_2 * tangential_direction_2;
			v = v + impulse_vector / m;
			w = w + Iinv * glm::cross(p, impulse_vector);

			vel_CPU[rigidBodyIndex] = make_float4(v.x, v.y, v.z, 0);
			rbAngularVelocity_CPU[rigidBodyIndex] = make_float4(w.x, w.y, w.z, 0);

			vel_CPU[rigidBodyIndex].x = abs(vel_CPU[rigidBodyIndex].x) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].x;
			vel_CPU[rigidBodyIndex].y = abs(vel_CPU[rigidBodyIndex].y) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].y;
			vel_CPU[rigidBodyIndex].z = abs(vel_CPU[rigidBodyIndex].z) < linear_bound ? 0 : vel_CPU[rigidBodyIndex].z;

			rbAngularVelocity_CPU[rigidBodyIndex].x = abs(rbAngularVelocity_CPU[rigidBodyIndex].x) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].x;
			rbAngularVelocity_CPU[rigidBodyIndex].y = abs(rbAngularVelocity_CPU[rigidBodyIndex].y) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].y;
			rbAngularVelocity_CPU[rigidBodyIndex].z = abs(rbAngularVelocity_CPU[rigidBodyIndex].z) < angular_bound ? 0 : rbAngularVelocity_CPU[rigidBodyIndex].z;


#ifdef PRINT_COLLISIONS	
			std::cout << "Applied impulse: " << corrective_impulse << std::endl;
			std::cout << "New linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
			std::cout << "New angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;
			std::cout << std::endl;
#endif
		}
	}


	checkCudaErrors(cudaMemcpy(rbPositions, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbVelocities, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

	delete CMs_CPU;
	delete vel_CPU;
	delete rbAngularVelocity_CPU;
	delete rbCurrentInertia_CPU;
	delete rbMass_CPU;

	delete relative_CPU;
	delete position_CPU;
	delete normal_CPU;
	delete particlePosition_CPU;

	delete contactDistance_CPU;
	delete collidingParticleIndex_CPU;

	delete contactNormal;
	delete contactAccumulatedFriction;
	delete contactAccumulatedFriction_2;
	delete contactAccumulatedImpulse;
	delete contactRigidBody;
	delete contactPoint;
	delete contactBias;

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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

/*
* Collision detection algorithm based on BVH trees. BVH trees form the broadphase collision detection part of the physics engine.
* Currently, all dynamic simulations are only particle based. Particles do not form rigid bodies. We handle collisions with the DEM method.
* The BVH is constructed in parallel on the GPU using CUDA. We use a variation of BVH trees called LBVH. The algorithm consists of the following steps:
* 1) Find the AABB of the scene.
* 2) Normalize each positions so that it belongs in the unit cube [0-1] and assign a Morton code to the corresponding particle.
* 3) Use the Morton codes to sort the associated particles.
* 4) Construct a binary radix tree of the sorted particles based on Terro Karras' paper [2012].
* 5) Create a BVH on top of the binary tree using a bottom-up approach.
* 6) Use the BVH to identify collisions.
* Deprecated because we need joint support for rigid bodies and independent particles.
*/
void ParticleSystem::updateBVHSoA(float deltaTime)
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

	// update constants
	setParameters(&m_params);

	// integrate
	integrateSystem(
		dPos,
		m_dVel,
		deltaTime,
		minPos,
		maxPos,
		rbIndices,
		m_numParticles);

	static double totalRadixTime = 0;
	static double totalLeafTime = 0;
	static double totalInternalTime = 0;
	static double totalCollisionTime = 0;
	static double totalInitTime = 0;
	static int iterations = 0;
	clock_t start = clock();

	checkCudaErrors(createMortonCodes((float4 *)dPos,
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices,
		m_numParticles,
		numThreads));
	clock_t end = clock();
	totalInitTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
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

	start = clock();
	initializeRadiiWrapper(radii,
		m_params.particleRadius,
		m_numParticles,
		numThreads);

	wrapperCollideBVHSoA(
		(float4 *)dCol, //particle's color, only used for testing purposes
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
		m_params, //simulation parameters
		numThreads);
	end = clock();
	totalCollisionTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last 1000 iterations..." << std::endl;
		std::cout << "Average time spent on initialization and sorting: " << totalInitTime / iterations << " (ms)" << std::endl;
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
	if (simulateAR) updateStaticParticlesBVHSoA(deltaTime);
}

/*
* Virtual vs real particle collision detection.
* Algorithm is the same as virtual vs virtual collision detection except for the final step.
* We create a BVH for the static particles using steps 1-5.
* For the 6th step, we use the created BVH to test for collisions with the virtual particles.
* Deprecated because we need joint support for rigid bodies and independent particles.
*/
void ParticleSystem::updateStaticParticlesBVHSoA(float deltaTime)
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
	static double totalRadixTime = 0;
	static double totalLeafTime = 0;
	static double totalInternalTime = 0;
	static double totalCollisionTime = 0;
	static double totalInitTime = 0;
	static int iterations = 0;
	clock_t start = clock();

	//create Morton codes for the static particles and sort them
	checkCudaErrors(createMortonCodes((float4 *)staticPos,
		&r_mortonCodes,
		&r_indices,
		&r_sortedMortonCodes,
		&r_sortedIndices,
		numberOfRangeData,
		numThreads));
	clock_t end = clock();
	totalInitTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
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
	wrapperStaticCollideBVHSoA((float4 *)dPos, //virtual particle positions
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
		numThreads,
		m_params); //simulation parameters
	end = clock();
	totalCollisionTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last 1000 iterations regarding static particles..." << std::endl;
		std::cout << "Average time spent on initialization and sorting: " << totalInitTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on radix tree creation: " << totalRadixTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on leaf nodes creation: " << totalLeafTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on internal nodes creation: " << totalInternalTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on collision detection and handling: " << totalCollisionTime / iterations << " (ms)" << std::endl;
	}
	cudaMemcpy(&minPos, &(r_bounds[numberOfRangeData].min), sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(&maxPos, &(r_bounds[numberOfRangeData].max), sizeof(float3), cudaMemcpyDeviceToHost);

	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_colorvbo_resource);
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}
}

/*
* Legacy function. Has been substituted with SoA equivalent which is much faster.
* Collision detection algorithm based on BVH trees. BVH trees form the broadphase collision detection part of the physics engine.
* Currently, all dynamic simulations are only particle based. Particles do not form rigid bodies. We handle collisions with the DEM method.
* The BVH is constructed in parallel on the GPU using CUDA. We use a variation of BVH trees called LBVH. The algorithm consists of the following steps:
* 1) Find the AABB of the scene.
* 2) Normalize each positions so that it belongs in the unit cube [0-1] and assign a Morton code to the corresponding particle.
* 3) Use the Morton codes to sort the associated particles.
* 4) Construct a binary radix tree of the sorted particles based on Terro Karras' paper [2012].
* 5) Create a BVH on top of the binary tree using a bottom-up approach.
* 6) Use the BVH to identify collisions.
*/
void ParticleSystem::updateBVH(float deltaTime)
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

	// update constants
	setParameters(&m_params);

	// integrate
	integrateSystem(
		dPos,
		m_dVel,
		deltaTime,
		minPos,
		maxPos,
		rbIndices,
		m_numParticles);

	TreeNode<AABB> *cudaDeviceTreeNodes, *cudaDeviceTreeLeaves;
	unsigned int *mortonCodes, *sortedMortonCodes;
	int *indices, *sortedIndices;

	static double totalRadixTime = 0;
	static double totalBVHTime = 0;
	static double totalCollisionTime = 0;
	static double totalInitTime = 0;
	static int iterations = 0;

	clock_t start = clock();
	checkCudaErrors(createCUDAarrays((float4 *)dPos,
		&cudaDeviceTreeNodes,
		&cudaDeviceTreeLeaves,
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices,
		m_numParticles,
		numThreads));

	checkCudaErrors(createMortonCodes((float4 *)dPos,
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices,
		m_numParticles,
		numThreads));
	clock_t end = clock();
	totalInitTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	start = clock();
	checkCudaErrors(constructRadixTree(&cudaDeviceTreeNodes,
		&cudaDeviceTreeLeaves,
		sortedMortonCodes,
		m_numParticles,
		numThreads));
	end = clock();
	totalRadixTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	start = clock();
	checkCudaErrors(constructBVHTree(&cudaDeviceTreeNodes,
		&cudaDeviceTreeLeaves,
		dPos,
		m_params.particleRadius,
		sortedIndices,
		sortedMortonCodes,
		m_numParticles,
		numThreads));
	end = clock();
	totalBVHTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	start = clock();
	checkCudaErrors(collisionDetectionAndHandling((float4 *)dCol,
		m_dVel,
		cudaDeviceTreeNodes,
		cudaDeviceTreeLeaves,
		m_numParticles,
		numThreads,
		m_params));
	end = clock();
	totalCollisionTime += (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
	if (++iterations == 1000)
	{
		std::cout << "Average compute times for last 1000 iterations..." << std::endl;
		std::cout << "Average time spent on initialization and sorting: " << totalInitTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on radix tree creation: " << totalRadixTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on BVH tree creation: " << totalBVHTime / iterations << " (ms)" << std::endl;
		std::cout << "Average time spent on collision detection and handling: " << totalCollisionTime / iterations << " (ms)" << std::endl;
	}

	if (cudaDeviceTreeNodes)
		cudaFree(cudaDeviceTreeNodes);
	if (cudaDeviceTreeLeaves)
		cudaFree(cudaDeviceTreeLeaves);
	if (mortonCodes)
		cudaFree(mortonCodes);
	if (sortedMortonCodes)
		cudaFree(sortedMortonCodes);
	if (indices)
		cudaFree(indices);
	if (sortedIndices)
		cudaFree(sortedIndices);

	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_colorvbo_resource);
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}
	if (simulateAR) updateStaticParticlesBVH(deltaTime);
}

/*
* Legacy function. Has been substituted with SoA equivalent which is much faster.
* Virtual vs real particle collision detection.
* Algorithm is the same as virtual vs virtual collision detection except for the final step.
* We create a BVH for the static particles using steps 1-5.
* For the 6th step, we use the created BVH to test for collisions with the virtual particles.
*/
void ParticleSystem::updateStaticParticlesBVH(float deltaTime)
{
	TreeNode<AABB> *cudaDeviceTreeNodes, *cudaDeviceTreeLeaves;
	unsigned int *mortonCodes, *sortedMortonCodes;
	int *indices, *sortedIndices;

	checkCudaErrors(createCUDAarrays((float4 *)staticPos,
		&cudaDeviceTreeNodes,
		&cudaDeviceTreeLeaves,
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices,
		numberOfRangeData,
		numThreads));

	checkCudaErrors(createMortonCodes((float4 *)staticPos,
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices,
		numberOfRangeData,
		numThreads));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(constructRadixTree(&cudaDeviceTreeNodes,
		&cudaDeviceTreeLeaves,
		sortedMortonCodes,
		numberOfRangeData,
		numThreads));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(constructBVHTree(&cudaDeviceTreeNodes,
		&cudaDeviceTreeLeaves,
		staticPos,
		m_params.particleRadius,
		sortedIndices,
		sortedMortonCodes,
		numberOfRangeData,
		numThreads));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	float *dPos;
	if (m_bUseOpenGL)
	{
		dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	}
	else
	{
		dPos = (float *)m_cudaPosVBO;
	}

	//insert routine for collision detection with static particles here
	checkCudaErrors(staticCollisionDetection(dPos,
		m_dVel,
		cudaDeviceTreeNodes,
		cudaDeviceTreeLeaves,
		m_numParticles,
		numThreads,
		m_params));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	if (m_bUseOpenGL)
	{
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}


	if (cudaDeviceTreeNodes)
		cudaFree(cudaDeviceTreeNodes);
	if (cudaDeviceTreeLeaves)
		cudaFree(cudaDeviceTreeLeaves);
	if (mortonCodes)
		cudaFree(mortonCodes);
	if (sortedMortonCodes)
		cudaFree(sortedMortonCodes);
	if (indices)
		cudaFree(indices);
	if (sortedIndices)
		cudaFree(sortedIndices);
}

void ParticleSystem::initializeVirtualSoA()
{
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//cudaFree everything
	if (mortonCodes)checkCudaErrors(cudaFree(mortonCodes));
	if (sortedMortonCodes)checkCudaErrors(cudaFree(sortedMortonCodes));
	if (indices)checkCudaErrors(cudaFree(indices));
	if (sortedIndices)checkCudaErrors(cudaFree(sortedIndices));
	if (parentIndices)checkCudaErrors(cudaFree(parentIndices));
	if (leftIndices)checkCudaErrors(cudaFree(leftIndices));
	if (rightIndices)checkCudaErrors(cudaFree(rightIndices));
	if (radii)checkCudaErrors(cudaFree(radii));
	if (minRange)checkCudaErrors(cudaFree(minRange));
	if (maxRange)checkCudaErrors(cudaFree(maxRange));
	if (bounds)checkCudaErrors(cudaFree(bounds));
	if (isLeaf)checkCudaErrors(cudaFree(isLeaf));
	if (CMs)checkCudaErrors(cudaFree(CMs));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	createSoA(
		&isLeaf, //array containing a flag to indicate whether node is leaf
		&parentIndices, //array containing indices of the parent of each node
		&leftIndices, //array containing indices of the left children of each node
		&rightIndices, //array containing indices of the right children of each node
		&minRange, //array containing minimum (sorted) leaf covered by each node
		&maxRange, //array containing maximum (sorted) leaf covered by each node
		&CMs, //array containing centers of mass for each leaf
		&bounds, //array containing bounding volume for each node - currently templated Array of Structures
		&radii, //radii of all nodes - currently the same for all particles
		&mortonCodes,
		&indices,
		&sortedMortonCodes,
		&sortedIndices, //array containing corresponding unsorted indices for each leaf
		m_numParticles,
		numThreads);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::initializeRealSoA()
{
	if (r_isLeaf)
		cudaFree(r_isLeaf);
	if (r_parentIndices)
		cudaFree(r_parentIndices);
	if (r_leftIndices)
		cudaFree(r_leftIndices);
	if (r_rightIndices)
		cudaFree(r_rightIndices);
	if (r_minRange)
		cudaFree(r_minRange);
	if (r_maxRange)
		cudaFree(r_maxRange);
	if (r_CMs)
		cudaFree(r_CMs);
	if (r_bounds)
		cudaFree(r_bounds);
	if (r_radii)
		cudaFree(r_radii);
	if (r_mortonCodes)
		cudaFree(r_mortonCodes);
	if (r_indices)
		cudaFree(r_indices);
	if (r_sortedMortonCodes)
		cudaFree(r_sortedMortonCodes);
	if (r_sortedIndices)
		cudaFree(r_sortedIndices);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//create arrays for the static particles
	createSoA(
		&r_isLeaf, //array containing a flag to indicate whether node is leaf
		&r_parentIndices, //array containing indices of the parent of each node
		&r_leftIndices, //array containing indices of the left children of each node
		&r_rightIndices, //array containing indices of the right children of each node
		&r_minRange, //array containing minimum (sorted) leaf covered by each node
		&r_maxRange, //array containing maximum (sorted) leaf covered by each node
		&r_CMs, //array containing centers of mass for each leaf
		&r_bounds, //array containing bounding volume for each node - currently templated Array of Structures
		&r_radii, //radii of all nodes - currently the same for all particles
		&r_mortonCodes,
		&r_indices,
		&r_sortedMortonCodes,
		&r_sortedIndices, //array containing corresponding unsorted indices for each leaf
		numberOfRangeData,
		numThreads);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::update(float deltaTime)
{
	//updateBVHSoA(deltaTime);
	
	/*m_params.spring = 0.9f;
	m_params.damping = 0.01f;
	m_params.shear = 0.0f;*/
	//m_params.gravity.y = -0.008;
	//m_params.gravity.y = -0.012;

	// simulation parameters
	deltaTime = 0.05;
	//deltaTime = 0.01;
	m_params.gravity.y = -0.012;
	m_params.Wdamping = 0.9999;
	m_params.Vdamping = 0.9999;

	// DEM parameters
	m_params.spring = 0.5f;
	m_params.damping = 0.0f;
	m_params.shear = 0.0f;

	// SIS parameters
	m_params.ARrestitution = 0.5;
	m_params.RBrestitution = 0.7;
	m_params.ARfriction = 0.1;
	m_params.RBfriction = 0.0;

	if (m_numParticles)
	{
		//if (collisionMethod == M_UNIFORM_GRID)
  //      {
		//	updateUniformGrid(deltaTime);
		//	//updateUniformGridDEM(deltaTime);
		//}
		//else if (collisionMethod == M_BVH)
		//{
		//	updateBVHExperimental(deltaTime);
		//}
		//updateUniformGrid(deltaTime);
		/*static int iterations = 0;
		static float totalTime = 0;
		clock_t start = clock();*/
		
		/*iterations++;
		clock_t end = clock();
		totalTime += (end - start) / (CLOCKS_PER_SEC / 1000);

		if (iterations == 1000)
		{
			std::cout << "Avg time spent on uniform grid update: " << totalTime / (float)iterations << std::endl;
			std::cout << std::endl;
		}*/
		//updateUniformGrid(deltaTime);
		updateUniformGridSIS(deltaTime);
		//updateBVHExperimental(deltaTime);
		//updateUniformGridDEM(deltaTime);
		/*static int iterations = 1;
		float4 *VEL_CPU = new float4[numRigidBodies];
		float4 *ANG_CPU = new float4[numRigidBodies];
		checkCudaErrors(cudaMemcpy(VEL_CPU, rbVelocities, numRigidBodies * sizeof(float) * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(ANG_CPU, rbAngularVelocity, numRigidBodies * sizeof(float) * 4, cudaMemcpyDeviceToHost));
		for (int rb = 0; rb < numRigidBodies; rb++)
		{
			if (iterations < 20)
			{
				std::ostringstream counter;
				counter << (rb + 1);
				std::string fileName("my_body_");
				fileName += counter.str();
				fileName += "_vel_DEM.txt";
				std::ofstream file(fileName.c_str(), std::ofstream::app);
				file << VEL_CPU[rb].x << " " << VEL_CPU[rb].y << " " << VEL_CPU[rb].z << std::endl;
				file.close();

				fileName = "my_body_";
				fileName += counter.str();
				fileName += "_ang_DEM.txt";
				std::ofstream file2(fileName.c_str(), std::ofstream::app);
				file2 << ANG_CPU[rb].x << " " << ANG_CPU[rb].y << " " << ANG_CPU[rb].z << std::endl;
				file2.close();
			}
		}
		iterations++;
		delete ANG_CPU;
		delete VEL_CPU;*/
	}
    /*if (!pauseFrame)
    {
    	updateFrame();

    }*/


}

void ParticleSystem::Integrate_Rigid_Body_System_GPU(float deltaTime)
{
	// these mallocs and memcpys could be further optimized by only allocating once
	// but better to be safe than sorry
	glm::mat4 *modelMatrixGPU;
	checkCudaErrors(cudaMalloc((void**)&modelMatrixGPU, sizeof(glm::mat4) * numRigidBodies));

	glm::quat *cumulativeQuaternionGPU;
	checkCudaErrors(cudaMalloc((void**)&cumulativeQuaternionGPU, sizeof(glm::quat) * numRigidBodies));
	checkCudaErrors(cudaMemcpy(cumulativeQuaternionGPU, cumulativeQuaternion, sizeof(glm::quat) * numRigidBodies, cudaMemcpyHostToDevice));

	GPUintegratorWrapper((float4 *)rbPositions, //rigid body center of mass
		(float4 *)rbVelocities, //velocity of rigid body
		(float4 *)rbAngularVelocity, //contains angular velocities for each rigid body
		rbQuaternion, //contains current quaternion for each rigid body
		rbInertia, //original moment of inertia for each rigid body - 9 values per RB
		rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
		modelMatrixGPU, // modelMatrix used for rendering
		cumulativeQuaternionGPU,  // quaternion used to compute modelMatrix
		deltaTime, //dt
		rbMass, //inverse of total mass of rigid body
		numRigidBodies, //number of rigid bodies
		m_params,
		numThreads);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(cumulativeQuaternion, cumulativeQuaternionGPU, sizeof(glm::quat) * numRigidBodies, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(modelMatrix, modelMatrixGPU, sizeof(glm::mat4) * numRigidBodies, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(cumulativeQuaternionGPU));
	checkCudaErrors(cudaFree(modelMatrixGPU));
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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	resetQuaternionWrapper(rbQuaternion, numRigidBodies, numThreads);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


}

void ParticleSystem::Integrate_RB_System(float deltaTime)
{
	//integrateRigidBodyCPU(
	//	cumulativeQuaternion,
	//	modelMatrix, // model matrix array used for rendering
	//	(float4 *)rbPositions, //rigid body center of mass
	//	(float4 *)rbVelocities, //velocity of rigid body
	//	(float4 *)rbForces, //total force applied to rigid body due to previous collisions
	//	(float4 *)rbAngularVelocity, //contains angular velocities for each rigid body
	//	rbQuaternion, //contains current quaternion for each rigid body
	//	(float4 *)rbTorque, //torque applied to rigid body due to previous collisions
	//	(float4 *)rbAngularMomentum, //cumulative angular momentum of the rigid body
	//	(float4 *)rbLinearMomentum, //cumulative linear momentum of the rigid body
	//	rbInertia, //original moment of inertia for each rigid body - 9 values per RB
	//	rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
	//	rbAngularAcceleration, //current angular acceleration due to misaligned angular momentum and velocity
	//	deltaTime, //dt
	//	rbRadii, //radius chosen for each rigid body sphere
	//	rbMass, //total mass of rigid body
	//	minPos, //smallest coordinate of scene's bounding box
	//	maxPos, //largest coordinate of scene's bounding box
	//	numRigidBodies, //number of rigid bodies
	//	m_params); //simulation parameters

	//float4 *pos = new float4[m_numParticles];
	//checkCudaErrors(cudaMemcpy(pos, relativePos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost));
	//int total = 0;
	//for (int rb = 0; rb < numRigidBodies; rb++)
	//{
	//	for (int p = 0; p < particlesPerObjectThrown[rb]; p++)
	//	{
	//		
	//		if (abs(pos[p].x - pos[total].x) > 0.001 ||
	//			abs(pos[p].y - pos[total].y) > 0.001 ||
	//			abs(pos[p].z - pos[total].z) > 0.001)
	//		{
	//			std::cout << "Wrong relative position found @ rigid body: " << rb << " particle:" << p << std::endl;
	//			/*std::cout << "Position 1 is: (" << pos[p].x << ", " <<
	//				pos[p].y << ", " << pos[p].z << ", " << pos[p].w << ")" << std::endl;
	//			std::cout << "Position 2 is: (" << pos[total].x << ", " <<
	//				pos[total].y << ", " << pos[total].z << ", " << pos[total].w << ")" << std::endl;*/
	//		}
	//		total++;
	//	}
	//}
	
	// CPU Test
//	glm::quat *quaternions = new glm::quat[numRigidBodies];
//	checkCudaErrors(cudaMemcpy(quaternions, rbQuaternion, sizeof(glm::quat) * numRigidBodies, cudaMemcpyDeviceToHost));
//	total = 0;
//	for (int rb = 0; rb < numRigidBodies; rb++)
//	{
//		glm::mat4 rot = mat4_cast(quaternions[rb]);
//		/*std::cout << "Rotation matrix of rigid body #" << rb << " is:" << std::endl;
//		for (int row = 0; row < 4; row++)
//		{
//			for (int col = 0; col < 4; col++)
//				std::cout << rot[row][col] << " ";
//			std::cout << std::endl;
//		}*/
//
//		for (int p = 0; p < particlesPerObjectThrown[rb]; p++)
//		{
//			/*if (total >= 10240 && total <= 10415)
//				std::cout << "Position @" << total << " is: (" << pos[total].x << ", " <<
//				pos[total].y << ", " << pos[total].z << ", " << pos[total].w << ")" << std::endl;
//*/
//			glm::vec4 position = glm::vec4(pos[total].x, pos[total].y, pos[total].z, 0.f);
//			
//			position = rot * position;
//			pos[total] = make_float4(position.x, position.y, position.z, 0);
//			/*if (total >= 10240 && total <= 10415)
//				std::cout << "Rotated position @" << total << " is: (" << pos[total].x << ", " <<
//				pos[total].y << ", " << pos[total].z << ", " << pos[total].w << ")" << std::endl;*/
//			total++;
//		}
//	}
//
	

	//total = 0;
	//for (int rb = 0; rb < numRigidBodies; rb++)
	//{
	//	for (int p = 0; p < particlesPerObjectThrown[rb]; p++)
	//	{
	//		if (abs(pos[p].x - pos[total].x) > 0.001 ||
	//			abs(pos[p].y - pos[total].y) > 0.001 ||
	//			abs(pos[p].z - pos[total].z) > 0.001)
	//		{
	//			std::cout << "Wrong relative position found @ rigid body: " << rb << " particle:" << p << std::endl;
	//			/*std::cout << "Position 1 is: (" << pos[p].x << ", " <<
	//				pos[p].y << ", " << pos[p].z << ", " << pos[p].w << ")" << std::endl;
	//			std::cout << "Position 2 is: (" << pos[total].x << ", " <<
	//				pos[total].y << ", " << pos[total].z << ", " << pos[total].w << ")" << std::endl;*/
	//		}
	//		total++;
	//	}
	//}

	//int *index = new int[m_numParticles];
	//checkCudaErrors(cudaMemcpy(index, rbIndices, sizeof(int) * m_numParticles, cudaMemcpyDeviceToHost));
	//total = 0;
	//int startIndex = 0;
	//for (int rb = 0; rb < numRigidBodies; rb++)
	//{
	//	if (rb == 23)
	//		startIndex = total;
	//	for (int p = 0; p < particlesPerObjectThrown[rb]; p++)
	//	{
	//		if (rb != index[total])
	//			std::cout << "Wrong index found @ rigid body: " << rb << " particle:" << p << "(" << index[total] << ")" << std::endl;
	//		if (abs(quaternions[rb].w - quaternions[index[total]].w) > 0.001 ||
	//			abs(quaternions[rb].x - quaternions[index[total]].x) > 0.001 ||
	//			abs(quaternions[rb].y - quaternions[index[total]].y) > 0.001 ||
	//			abs(quaternions[rb].z - quaternions[index[total]].z) > 0.001)
	//		{
	//			std::cout << "Wrong quaternion found @ rigid body: " << rb << " particle:" << p << std::endl;
	//		}
	//		total++;
	//	}
	//}
	//delete index;
	//delete quaternions;
	//std::cout << "Calling DebugComputeGlobalAttributes for " << particlesPerObjectThrown[23] * 2 << " particles." << std::endl;
	//DebugComputeGlobalAttributes((float4 *)rbPositions, //rigid body's center of mass
	//	(float4 *)rbVelocities, //rigid body's velocity
	//	(float4 *)relativePos, //particle's relative position
	//	(float4 *)dPos, //particle's global position
	//	(float4 *)m_dVel, //particle's world velocity
	//	rbQuaternion, //contains current quaternion for each rigid body
	//	(float4 *)rbAngularVelocity, //contains angular velocities for each rigid body
	//	rbIndices, //index of associated rigid body
	//	startIndex,
	//	particlesPerObjectThrown[23] * 2, //number of particles
	//	numThreads); //number of threads

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

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	resetQuaternionWrapper(rbQuaternion, numRigidBodies, numThreads);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//std::cout << "Calling computeGlobalAttributesWrapper for " << m_numParticles << " particles." << std::endl;
	/*checkCudaErrors(cudaMemcpy(pos, relativePos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost));
	total = 0;
	for (int rb = 0; rb < numRigidBodies; rb++)
	{
		for (int p = 0; p < particlesPerObjectThrown[rb]; p++)
		{
			if (abs(pos[p].x - pos[total].x) > 0.001 ||
				abs(pos[p].y - pos[total].y) > 0.001 ||
				abs(pos[p].z - pos[total].z) > 0.001)
			{
				std::cout << "Wrong relative position found @ rigid body: " << rb << " particle:" << total << std::endl;
				std::cout << "Position 1 is: (" << pos[p].x << ", " <<
					pos[p].y << ", " << pos[p].z << ", " << pos[p].w << ")" << std::endl;
				std::cout << "Position 2 is: (" << pos[total].x << ", " <<
					pos[total].y << ", " << pos[total].z << ", " << pos[total].w << ")" << std::endl;
			}
			total++;
		}
	}
	delete pos;*/
	//integrate
	/*integrateSystem(
		dPos,
		m_dVel,
		deltaTime,
		minPos,
		maxPos,
		rbIndices,
		m_numParticles);*/

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
