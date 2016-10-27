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

		CMs_CPU[index] = locPos;
		vel_CPU[index] = locVel;
		rbCurrentInertia_CPU[index] = currentInertia;
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
	checkCudaErrors(cudaMemcpy(torqueTest, rbTorque, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	float *LTest = new float[4 * numRigidBodies];
	checkCudaErrors(cudaMemcpy(LTest, rbAngularMomentum, sizeof(float) * 4 * numRigidBodies, cudaMemcpyDeviceToHost));
	glm::vec3 *ldot = new glm::vec3[numRigidBodies];
	checkCudaErrors(cudaMemcpy(ldot, rbAngularAcceleration, sizeof(glm::vec3) * numRigidBodies, cudaMemcpyDeviceToHost));
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
		(float4 *)rbLinearMomentum, // rigid body linear momentum
		(float4 *)rbAngularMomentum, // rigid body angular momentum
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
				
				if (testParticleCollision(CMs_CPU[index] + relative_CPU[current_particle],
					CMs_CPU[rigidBodyIndex] + relative_CPU[particleIndex],
					m_params.particleRadius,
					m_params.particleRadius,
					CMs_CPU[index]) && index < rigidBodyIndex)
				{
					float4 cp, cn;
					findExactContactPoint(CMs_CPU[index] + relative_CPU[current_particle],
						CMs_CPU[rigidBodyIndex] + relative_CPU[particleIndex],
						m_params.particleRadius,
						m_params.particleRadius,
						cp, cn);
					float4 r1 = cp - CMs_CPU[index];
					float4 r2 = cp - CMs_CPU[rigidBodyIndex];

					glm::mat3 IinvA = rbCurrentInertia_CPU[index];
					glm::mat3 IinvB = rbCurrentInertia_CPU[rigidBodyIndex];
					float mA = rbMass_CPU[index];
					float mB = rbMass_CPU[rigidBodyIndex];
					float impulse = computeImpulseMagnitude(
						vel_CPU[index], vel_CPU[rigidBodyIndex],
						rbAngularVelocity_CPU[index], rbAngularVelocity_CPU[rigidBodyIndex],
						r1, r2,	IinvA, IinvB, mA, mB, cn);

					float4 impulseVector = cn * impulse;
					
					// apply linear impulse
					vel_CPU[index] += impulseVector / mA;
					vel_CPU[rigidBodyIndex] -= impulseVector / mB;

					// compute auxiliaries for angular impulse
					glm::vec3 rA(r1.x, r1.y, r1.z);
					glm::vec3 rB(r2.x, r2.y, r2.z);
					glm::vec3 impulseVectorGLM(impulseVector.x, impulseVector.y, impulseVector.z);

					// apply angular impulse
					glm::vec3 AngularImpulse = IinvA *
						(glm::cross(glm::vec3(r1.x, r1.y, r1.z), impulseVectorGLM));
					rbAngularVelocity_CPU[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

					AngularImpulse = IinvB *
						(glm::cross(glm::vec3(r2.x, r2.y, r2.z), impulseVectorGLM * (-1.f)));
					rbAngularVelocity_CPU[rigidBodyIndex] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);


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

	// cudaFree contact info variables
	/*checkCudaErrors(cudaFree(collidingRigidBodyIndex));
	checkCudaErrors(cudaFree(collidingParticleIndex));
	checkCudaErrors(cudaFree(contactDistance));*/

}

void ParticleSystem::Handle_Rigid_Body_Collisions_Baraff_GPU()
{
	HandleRigidBodyCollisionWrapper(
		(float4 *)dPos, // particle positions
		(float4 *)rbPositions, // rigid body center of mass
		(float4 *)rbVelocities, // rigid body linear velocity
		(float4 *)rbAngularVelocity, // rigid body angular velocity
		(float4 *)rbLinearMomentum, // rigid body linear momentum
		(float4 *)rbAngularMomentum, // rigid body angular momentum
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
		(float4 *)rbLinearMomentum, // rigid body linear momentum
		(float4 *)rbAngularMomentum, // rigid body angular momentum
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

	// copy contact info to CPU - one contact per particle
	float *contactDistance_CPU = new float[m_numParticles];
	int *collidingParticleIndex_CPU = new int[m_numParticles];

	checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, m_numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, m_numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	int current_particle = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			if (contactDistance_CPU[current_particle] > 0) // if current particle has collided
			{
				int particleIndex = collidingParticleIndex_CPU[current_particle];
				//std::cout << "Rigid body No." << index + 1 << " is colliding with particle No." << particleIndex << std::endl;
				////float3 displacementVector = make_float3(CMs_CPU[index] + relative_CPU[current_particle] - position_CPU[particleIndex]);
				//float3 displacementVector = make_float3(particlePosition_CPU[current_particle] - position_CPU[particleIndex]);
				//float displacementDistance = 2 * m_params.particleRadius - length(displacementVector);
				//std::cout << "GPU distance: " << contactDistance_CPU[current_particle] << " CPU distance: " << displacementDistance
				//for (int testParticle = 0; testParticle < numberOfRangeData; testParticle++)
				//{
				//	displacementVector = make_float3(particlePosition_CPU[current_particle] - position_CPU[testParticle]);
				//	displacementDistance = 2 * m_params.particleRadius - length(displacementVector);
				//	if (abs(displacementDistance - contactDistance_CPU[current_particle]) < 0.001)
				//	{
				//		std::cout << "Correct particle found @: " << testParticle << std::endl;
				//	}
				//	if (displacementDistance > 0)
				//	{
				//		std::cout << "Found collision @: " << testParticle << " distance is: " << displacementDistance << std::endl;
				//	}
				//}
				/*std::cout << "Rigid body state before collision:" << std::endl;
				std::cout << "Position: (" << CMs_CPU[index].x << " " << CMs_CPU[index].y << " "
					<< CMs_CPU[index].z << " " << CMs_CPU[index].w << ") " << std::endl;;
				std::cout << "Velocity: (" << vel_CPU[index].x << " " << vel_CPU[index].y << " "
					<< vel_CPU[index].z << " " << vel_CPU[index].w << ") " << std::endl;;
				std::cout << "Angular velocity: (" << rbAngularVelocity_CPU[index].x << " " << rbAngularVelocity_CPU[index].y << " "
					<< rbAngularVelocity_CPU[index].z << " " << rbAngularVelocity_CPU[index].w << ") " << std::endl;;
				std::cout << std::endl;*/
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
					float4 r1 = cp - CMs_CPU[index];
					
					glm::mat3 IinvA = rbCurrentInertia_CPU[index];
				
					float mA = rbMass_CPU[index];
					float impulse = computeImpulseMagnitude(vel_CPU[index], rbAngularVelocity_CPU[index], r1, IinvA, mA, cn);

					float4 impulseVector = cn * impulse;

					// apply linear impulse
					vel_CPU[index] += impulseVector / mA;
					vel_CPU[index].w = 0;
					// compute auxiliaries for angular impulse
					glm::vec3 rA(r1.x, r1.y, r1.z);
					glm::vec3 impulseVectorGLM(impulseVector.x, impulseVector.y, impulseVector.z);

					// apply angular impulse
					glm::vec3 AngularImpulse = IinvA *
						(glm::cross(glm::vec3(r1.x, r1.y, r1.z), impulseVectorGLM));
					rbAngularVelocity_CPU[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);
					rbAngularVelocity_CPU[index].w = 0;

					//std::cout << "Rigid body state after collision:" << std::endl;
					//std::cout << "Position: (" << CMs_CPU[index].x << " " << CMs_CPU[index].y << " "
					//	<< CMs_CPU[index].z << " " << CMs_CPU[index].w << ") " << std::endl;;
					//std::cout << "Velocity: (" << vel_CPU[index].x << " " << vel_CPU[index].y << " "
					//	<< vel_CPU[index].z << " " << vel_CPU[index].w << ") " << std::endl;;
					//std::cout << "Angular velocity: (" << rbAngularVelocity_CPU[index].x << " " << rbAngularVelocity_CPU[index].y << " "
					//	<< rbAngularVelocity_CPU[index].z << " " << rbAngularVelocity_CPU[index].w << ") " << std::endl;;
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
	delete position_CPU;
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
	if (mortonCodes)
		cudaFree(mortonCodes);
	if (sortedMortonCodes)
		cudaFree(sortedMortonCodes);
	if (indices)
		cudaFree(indices);
	if (sortedIndices)
		cudaFree(sortedIndices);
	if (parentIndices)
		cudaFree(parentIndices);
	if (leftIndices)
		cudaFree(leftIndices);
	if (rightIndices)
		cudaFree(rightIndices);
	if (radii)
		cudaFree(radii);
	if (minRange)
		cudaFree(minRange);
	if (maxRange)
		cudaFree(maxRange);
	if (bounds)
		cudaFree(bounds);
	if (isLeaf)
		cudaFree(isLeaf);
	if (CMs)
		cudaFree(CMs);
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
	m_params.spring = 0.5f;
	m_params.damping = 0.02f;//0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;
	m_params.gravity = make_float3(0.0f, 0.0f, 0.0f);// make_float3(0.0f, -0.0003f, 0.0f);
	m_params.globalDamping = 1.0f;
	if (m_numParticles)
	{
		if (collisionMethod == M_UNIFORM_GRID)
        {
			//updateGrid(deltaTime);
			updateGridExperimental(deltaTime);
			//flushAndPrintRigidBodyParameters();
		/*	if (simulateAR)
            {
				m_params.spring = 0.12f;
				m_params.damping = 0.02f;
				m_params.shear = 0.f;
				m_params.attraction = 0.0f;
				m_params.boundaryDamping = -0.5f;
				m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
				m_params.globalDamping = 1.f;
				updateStaticParticles(deltaTime);
			}*/
		}
		else if (collisionMethod == M_BVH)
		{
			updateRigidBodies(deltaTime);
			if (simulateAR)
			{
				m_params.spring = 0.1f;
				m_params.damping = 0.02f;
				m_params.shear = 0.02f;
				m_params.attraction = 0.0f;
				m_params.boundaryDamping = -0.5f;
				m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
				m_params.globalDamping = 1.f;
                staticUpdateRigidBodies(deltaTime);
			}
		}
	}
    if (!pauseFrame)
    {
    	updateFrame();

    }
}

void ParticleSystem::Integrate_RB_System(float deltaTime)
{
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
}
