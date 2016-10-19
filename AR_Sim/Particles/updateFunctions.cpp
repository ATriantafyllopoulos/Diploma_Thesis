#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
#include "BVHcreation.h"
#define USE_RUNGE_KUTTA

#define USE_QUATERNIONS


float* evaluate(float u [], float dt, int n, float derivative [], glm::mat3 Iinv, float massInv)
{
	float *state = new float[n];

	//x = x + xdot * dt
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;

	//q = q + qdot * dt?

	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;

	//P = P + F * dt
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;

	//L = L + tau * dt
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;
	*state++ = (*u++) + (*derivative++) * dt;

	//reset pointers to origin
	state -= n;
	derivative -=n;
	u -= n;
	//compute auxiliary variables for current iteration

	//v = P / m
	glm::vec3 v(state[7] * massInv, state[8] * massInv, state[9] * massInv);
	//w = Iinv * L
	glm::vec3 w = Iinv * glm::vec3(state[10], state[11], state[12]);
	glm::quat q(state[3], state[4], state[5], state[6]);

	float *x_dot = new float[n];
	//x_dot = u
	*x_dot++ = v.x;
	*x_dot++ = v.y;
	*x_dot++ = v.z;
	//q_dot = 1 / 2 * w * q;
	glm::quat w_hat(0, w.x, w.y, w.z);
	glm::quat q_dot = (w_hat * q) / 2.f;
	*x_dot++ = q_dot.x;
	*x_dot++ = q_dot.y;
	*x_dot++ = q_dot.z;
	*x_dot++ = q_dot.w;
	//P_dot = f - no force for now
	*x_dot++ = 0;
	*x_dot++ = 0;
	*x_dot++ = 0;
	//L_dot = tau - no torque for now
	*x_dot++ = 0;
	*x_dot++ = 0;
	*x_dot++ = 0;

	x_dot -= n;
	return x_dot;
}

float* integrate(int n, float u0 [], float step, float f [], glm::mat3 Iinv, float massInv)
{
    int i;
    float *k1, *k2, *k3;
    float *u1, *u2, *u3;
    float *u = new float[n];

    u1 = new float[n];
    u2 = new float[n];
    u3 = new float[n];

    k1 = evaluate(u0, step / 2.0f, n, f, Iinv, massInv);
    k2 = evaluate(u0, step / 2.0f, n, k1, Iinv, massInv);
    k3 = evaluate(u0, step, n, k2, Iinv, massInv);

    //
    //  Combine them to estimate the solution.
    //
    for (i = 0; i < n; i++)
    {
        u[i] = u0[i] + step * (f[i] + 2.0f * k1[i] + 2.0f * k2[i] + k3[i]) / 6.0f;
    }


    delete k1;
    delete k2;
    delete k3;
    delete u1;
    delete u2;
    delete u3;
    return u;
}

void ParticleSystem::integrateRigidBodyCPU_RK(float deltaTime) //simulation parameters
{
	int STATES = 13;
	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);

	static float totalTime = 0;
	std::cout << "Integrating rigid bodies on the CPU" << std::endl;
	float4 *CMs_CPU = new float4[numRigidBodies]; //rigid body center of mass
	float4 *vel_CPU = new float4[numRigidBodies];  //velocity of rigid body
	float4 *force_CPU = new float4[numRigidBodies];  //force applied to rigid body due to previous collisions
	float4 *rbAngularVelocity_CPU = new float4[numRigidBodies];  //contains angular velocities for each rigid body
	glm::quat *rbQuaternion_CPU = new glm::quat[numRigidBodies]; //contains current quaternion for each rigid body
	float4 *rbTorque_CPU = new float4[numRigidBodies];  //torque applied to rigid body due to previous collisions
	float4 *rbAngularMomentum_CPU = new float4[numRigidBodies];  //cumulative angular momentum of the rigid body
	float4 *rbLinearMomentum_CPU = new float4[numRigidBodies];  //cumulative linear momentum of the rigid body
	glm::mat3 *rbInertia_CPU = new glm::mat3[numRigidBodies];  //original moment of inertia for each rigid body - 9 values per RB
	glm::mat3 *rbCurrentInertia_CPU = new glm::mat3[numRigidBodies];  //current moment of inertia for each rigid body - 9 values per RB
	glm::vec3 *rbAngularAcceleration_CPU = new glm::vec3[numRigidBodies];  //current angular acceleration due to misaligned angular momentum and velocity
	float *rbRadii_CPU = new float[numRigidBodies];  //radius chosen for each rigid body sphere
	float *rbMass_CPU = new float[numRigidBodies];  //inverse of total mass of rigid body



	checkCudaErrors(cudaMemcpy(CMs_CPU, dPos, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vel_CPU, m_dVel, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(force_CPU, rbForces, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity_CPU, rbAngularVelocity, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbQuaternion_CPU, rbQuaternion, numRigidBodies * sizeof(glm::quat), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbTorque_CPU, rbTorque, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularMomentum_CPU, rbAngularMomentum, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbLinearMomentum_CPU, rbLinearMomentum, numRigidBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbInertia_CPU, rbInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia_CPU, rbCurrentInertia, numRigidBodies * sizeof(glm::mat3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbAngularAcceleration_CPU, rbAngularAcceleration, numRigidBodies * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbRadii_CPU, rbRadii, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbMass_CPU, rbMass, numRigidBodies * sizeof(float), cudaMemcpyDeviceToHost));

	for (int index = 0; index < numRigidBodies; index++)
	{
		float4 locLinearMomentum = rbLinearMomentum_CPU[index];

		maxPos.x = maxPos.x + 0.1;
		maxPos.y = maxPos.y + 0.1;
		maxPos.z = maxPos.z + 0.1;

		minPos.x = minPos.x - 0.1;
		minPos.y = minPos.y - 0.1;
		minPos.z = minPos.z - 1;

		float4 locPos = CMs_CPU[index];
		float sphereRadius = rbRadii_CPU[index];
		//handle wall collisions
		if (locPos.x > maxPos.x - sphereRadius)
		{
			locPos.x = maxPos.x - sphereRadius;
			locLinearMomentum.x *= m_params.boundaryDamping;
		}

		if (locPos.x < minPos.x + sphereRadius)
		{
			locPos.x = minPos.x + sphereRadius;
			locLinearMomentum.x *= m_params.boundaryDamping;
		}

		if (locPos.y > maxPos.y - sphereRadius && locLinearMomentum.y > 0)
		{
			locPos.y = maxPos.y - 2 * sphereRadius;
			locLinearMomentum.y *= m_params.boundaryDamping;
		}

		if (locPos.y < minPos.y + sphereRadius)
		{
			locPos.y = minPos.y + sphereRadius;
			locLinearMomentum.y *= m_params.boundaryDamping;
		}

		if (locPos.z > maxPos.z - sphereRadius)
		{
			locPos.z = maxPos.z - sphereRadius;
			locLinearMomentum.z *= m_params.boundaryDamping;
		}

		if (locPos.z < minPos.z + sphereRadius)
		{
			locPos.z = minPos.z + sphereRadius;
			locLinearMomentum.z *= m_params.boundaryDamping;
		}

		locLinearMomentum *= m_params.globalDamping;
		rbLinearMomentum_CPU[index] = locLinearMomentum;
		locPos.w = 0.f;
		CMs_CPU[index] = locPos;

		//xdot
		float* x_dot = new float[STATES];
		//x_dot = u
		*x_dot++ = vel_CPU[index].x;
		*x_dot++ = vel_CPU[index].y;
		*x_dot++ = vel_CPU[index].z;
		//q_dot = 1 / 2 * w * q;
		glm::vec3 w = rbCurrentInertia_CPU[index] * glm::vec3(rbAngularMomentum_CPU[index].x, rbAngularMomentum_CPU[index].y, rbAngularMomentum_CPU[index].z);
		glm::quat w_hat(0, w.x, w.y, w.z);
		glm::quat q_dot = (w_hat * rbQuaternion_CPU[index]) / 2.f;
		*x_dot++ = q_dot.x;
		*x_dot++ = q_dot.y;
		*x_dot++ = q_dot.z;
		*x_dot++ = q_dot.w;
		//P_dot = f
		*x_dot++ = force_CPU[index].x;
		*x_dot++ = force_CPU[index].y;
		*x_dot++ = force_CPU[index].z;
		//L_dot = tau
		*x_dot++ = rbTorque_CPU[index].x;
		*x_dot++ = rbTorque_CPU[index].y;
		*x_dot++ = rbTorque_CPU[index].z;
		x_dot -= STATES;

		//get state
	    float* state = new float[STATES];
	    *state++ = CMs_CPU[index].x;
	    *state++ = CMs_CPU[index].y;
	    *state++ = CMs_CPU[index].z;
	    *state++ = rbQuaternion_CPU[index].x;
	    *state++ = rbQuaternion_CPU[index].y;
	    *state++ = rbQuaternion_CPU[index].z;
	    *state++ = rbQuaternion_CPU[index].w;
	    *state++ = rbLinearMomentum_CPU[index].x;
	    *state++ = rbLinearMomentum_CPU[index].y;
	    *state++ = rbLinearMomentum_CPU[index].z;
	    *state++ = rbAngularMomentum_CPU[index].x;
	    *state++ = rbAngularMomentum_CPU[index].y;
	    *state++ = rbAngularMomentum_CPU[index].z;
	    state -= STATES;

	    float* newState = new float[STATES];
	    newState = integrate(STATES, state, deltaTime, x_dot, rbCurrentInertia_CPU[index], rbMass_CPU[index]);

		//set state
	    CMs_CPU[index].x = *newState++;
	    CMs_CPU[index].y = *newState++;
	    CMs_CPU[index].z = *newState++;
	    rbQuaternion_CPU[index].x = (*newState++);
	    rbQuaternion_CPU[index].y = (*newState++);
	    rbQuaternion_CPU[index].z = (*newState++);
	    rbQuaternion_CPU[index].w = (*newState++);
	    rbLinearMomentum_CPU[index].x = *newState++;
	    rbLinearMomentum_CPU[index].y = *newState++;
	    rbLinearMomentum_CPU[index].z = *newState++;
	    rbAngularMomentum_CPU[index].x = *newState++;
	    rbAngularMomentum_CPU[index].y = *newState++;
	    rbAngularMomentum_CPU[index].z = *newState++;
	    newState -= STATES;

	    //momentum
	    vel_CPU[index] = rbLinearMomentum_CPU[index] * rbMass_CPU[index];
	    rbQuaternion_CPU[index] = glm::normalize(rbQuaternion_CPU[index]);
	    glm::mat3 R = glm::mat3_cast(rbQuaternion_CPU[index]);
	    rbCurrentInertia_CPU[index] = R * rbCurrentInertia_CPU[index] * glm::transpose(R);
	    //angular momentum
	    glm::vec3 newVel = rbCurrentInertia_CPU[index] * glm::vec3(rbAngularMomentum_CPU[index].x,
	    		rbAngularMomentum_CPU[index].y, rbAngularMomentum_CPU[index].z);
	    rbAngularVelocity_CPU[index] = make_float4(newVel.x, newVel.y, newVel.z, 0);
	    rbTorque_CPU[index] = make_float4(0);
	    force_CPU[index] = make_float4(0);


	    if (newState)
	    delete newState;
	    if (x_dot)
	    delete x_dot;
	    if (state)
	    delete state;


	}

	checkCudaErrors(cudaMemcpy(dPos, CMs_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_dVel, vel_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbForces, force_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularVelocity, rbAngularVelocity_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbQuaternion, rbQuaternion_CPU, numRigidBodies * sizeof(glm::quat), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbTorque, rbTorque_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularMomentum, rbAngularMomentum_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbLinearMomentum, rbLinearMomentum_CPU, numRigidBodies * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbInertia, rbInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbCurrentInertia, rbCurrentInertia_CPU, numRigidBodies * sizeof(glm::mat3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbAngularAcceleration, rbAngularAcceleration_CPU, numRigidBodies * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbRadii, rbRadii_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbMass, rbMass_CPU, numRigidBodies * sizeof(float), cudaMemcpyHostToDevice));

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

	totalTime += deltaTime;
	unmapGLBufferObject(m_cuda_posvbo_resource);
}

glm::vec3 getWdot(glm::vec3 w, glm::mat3 I_inv)
{
	return I_inv * glm::cross(transpose(I_inv) * w, w);
}
//		static int iterations = 0;
//		static std::ofstream myfile ("velocity.txt");
//		static std::ofstream myfileQuat ("myQuaternion.txt");
//		if (myfile.is_open())
//		{
//			myfile << newVelocity.x << " " << newVelocity.y << " " << newVelocity.z << '\n';
//		}

//		glm::vec3 undistortedVelocity = glm::vec3(rbAngularVelocity_CPU[index].x, rbAngularVelocity_CPU[index].y, rbAngularVelocity_CPU[index].z);
//		glm::vec3 angularAcceleration = currentInertia * glm::cross(currentMomentum, undistortedVelocity);
////		glm::vec3 newVelocity = currentInertia * currentMomentum - angularAcceleration * deltaTime;
//		glm::vec3 w;
//		w = currentInertia * L;
//		glm::vec3 wdot = currentInertia * (Ldot - glm::cross(w, Ldot));
//		glm::vec3 wXwdot = glm::cross(w, wdot);
//		glm::vec3 wdotXL = glm::cross(wdot, L);
//		glm::vec3 wXLdot = glm::cross(w, Ldot);
//		glm::vec3 wXL = glm::cross(w, L);
//		glm::vec3 wXwXL = glm::cross(w, wXL);
//		glm::vec3 wddot = wXwdot + currentInertia * ( -wdotXL -  wXLdot * 2.f + glm::cross(w, wXL) );
//		glm::vec3 superVec = -glm::cross(wdot, Ldot) * 3.f - glm::cross(wddot, L) + glm::cross(wdot, wXL) +
//				glm::cross(w, wdotXL) * 2.f + glm::cross(w, wdotXL) - glm::cross( w, wXwXL );
//		glm::vec3 wdddot = glm::cross(w, wddot) * 2.f - glm::cross(w, wXwdot) + currentInertia * superVec;
//		float h = deltaTime;
//		glm::vec3 newVelocity = w + wdot * (h / 2.f) + wddot * (h * h / 6.f) + glm::cross(wdot, w) * (h * h / 12.f) +
//				wdddot * (h * h * h / 24.f) + glm::cross(wddot, w) * (h * h * h / 24.f);
//		currentMomentum *= 0.9f;
//		newVelocity *= 0.9f;
//		glm::vec3 k1 = getWdot(newVelocity, currentInertia);
//		glm::vec3 k2 = getWdot(newVelocity + k1 * 0.5f , currentInertia);
//		glm::vec3 k3 = getWdot(newVelocity + k2 * 0.5f, currentInertia);
//		glm::vec3 k4 = getWdot(newVelocity + k3, currentInertia);
//		newVelocity += (k1 + k2 * 2.f + k3 * 3.f + k4) * (deltaTime / 6.f);
		//	correct angular drift
//		glm::vec3 currentTorque(torque.x, torque.y, torque.z);
//		glm::vec3 angularAcceleration = currentInertia * glm::cross(currentMomentum, newVelocity);
//		std::cout << "Velocity before correction: " << newVelocity.x << " " <<
//				newVelocity.y << " " << newVelocity.z  << ")" << std::endl;
//		newVelocity -= angularAcceleration * deltaTime;
//
//		angularAcceleration = currentInertia * glm::cross(currentMomentum, newVelocity);
//		newVelocity -= angularAcceleration * deltaTime;
//		std::cout << "Velocity after correction: " << newVelocity.x << " " <<
//				newVelocity.y << " " << newVelocity.z  << ")" << std::endl;
//		std::cout << std::endl;
	//	newVelocity = glm::vec3(0.001, 0.004, 0.001);
//		glm::quat qdot = glm::quat(0, newVelocity.x, newVelocity.y, newVelocity.z) * quaternion;
//		qdot /= 2.f;
//		quaternion += qdot * deltaTime;



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
			updateGrid(deltaTime);
			//flushAndPrintRigidBodyParameters();
			if (simulateAR)
            {
				m_params.spring = 0.12f;
				m_params.damping = 0.02f;
				m_params.shear = 0.f;
				m_params.attraction = 0.0f;
				m_params.boundaryDamping = -0.5f;
				m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
				m_params.globalDamping = 1.f;
				updateStaticParticles(deltaTime);
			}
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
