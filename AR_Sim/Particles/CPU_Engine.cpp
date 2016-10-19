#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
void ParticleSystem::updateCPU(const float &dt)
{
	//	std::cout << "Minpos: (" << minPos.x << ", " << minPos.y << ", " << minPos.z << ")" << std::endl;
	//	std::cout << "Maxpos: (" << maxPos.x << ", " << maxPos.y << ", " << maxPos.z << ")" << std::endl;
	integrateCPU(dt);
	collisionsCPU();
	float *dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
	checkCudaErrors(cudaMemcpy(dPos, POS_CPU, sizeof(float) * 4 * m_numParticles, cudaMemcpyHostToDevice));
	unmapGLBufferObject(m_cuda_posvbo_resource);

	float *dCol = (float *)mapGLBufferObject(&m_cuda_colorvbo_resource);
	checkCudaErrors(cudaMemcpy(dCol, COL_CPU, sizeof(float) * 4 * m_numParticles, cudaMemcpyHostToDevice));
	unmapGLBufferObject(m_cuda_colorvbo_resource);
}

void ParticleSystem::WallCollision(
	const float &radius, // particle radius
	const float &m, // rigid body mass
	const float4 &disp, // particle displacement
	const glm::mat3 &Iinv, // rigid body inverse inertia matrix
	const float4 &p, // old particle position
	float4 &n, // wall normal
	float3 &wallPos, // wall plane
	float4 &CM, // rigid body center of mass
	float4 &force, // total force acting on rigid body
	float4 &v, // rigid body linear velocity
	float4 &w)  // rigid body angular velocity
{
	float4 radiusAuxil = radius * n;
	float4 wallPosAuxil = n * dot(wallPos, make_float3(n));
	float4 particlePosAuxil = n * dot(p, n);

	CM += wallPosAuxil - radiusAuxil - particlePosAuxil; // move center of mass in the direction of the normal
	float4 newP = CM + disp;
	float4 cn, cp;
	findExactContactPoint(newP, newP + n * radius, radius, 0, cp, cn);
	float4 r = CM - cp;
	float4 localforce = cn * computeImpulseMagnitude(v, w, r, Iinv, m, cn);
	force += localforce;
	v += localforce / m;
	glm::vec3 torque = Iinv * (glm::cross(glm::vec3(r.x, r.y, r.z), glm::vec3(localforce.x, localforce.y, localforce.z)));
	w += make_float4(torque.x, torque.y, torque.z, 0);
}

bool ParticleSystem::HandleWallCollisions(
	const float &radius,
	const float &m,
	const float4 &p,
	const float4 &disp,
	float4 &v,
	float4 &w,
	const glm::mat3 &Iinv,
	float4 &CM,
	float4 &force)
{
	if (p.x > maxPos.x - radius)
	{
		float4 n = make_float4(1, 0, 0, 0);
		WallCollision(
			radius, // particle radius
			m, // rigid body mass
			disp, // particle displacement
			Iinv, // rigid body inverse inertia matrix
			p, // old particle position
			n, // wall normal
			maxPos, // wall plane
			CM, // rigid body center of mass
			force, // total force acting on rigid body
			v, // rigid body linear velocity
			w);  // rigid body angular velocity
		return true;
	}
	else if (p.y > maxPos.y - radius)
	{
		float4 n = make_float4(0, 1, 0, 0);
		WallCollision(
			radius, // particle radius
			m, // rigid body mass
			disp, // particle displacement
			Iinv, // rigid body inverse inertia matrix
			p, // old particle position
			n, // wall normal
			maxPos, // wall plane
			CM, // rigid body center of mass
			force, // total force acting on rigid body
			v, // rigid body linear velocity
			w);  // rigid body angular velocity
		return true;
	}
	else if (p.z > maxPos.z - radius)
	{
		float4 n = make_float4(0, 0, 1, 0);
		WallCollision(
			radius, // particle radius
			m, // rigid body mass
			disp, // particle displacement
			Iinv, // rigid body inverse inertia matrix
			p, // old particle position
			n, // wall normal
			maxPos, // wall plane
			CM, // rigid body center of mass
			force, // total force acting on rigid body
			v, // rigid body linear velocity
			w);  // rigid body angular velocity
		return true;
	}
	else if (p.x < minPos.x + radius)
	{
		float4 n = make_float4(-1, 0, 0, 0);
		const float4 negP = -1 * p;
		// in case of negative n, p and wallPos must also be negative
		// so that they will be used with the correct sign inside the 
		// function CM += wallPos * n - radius * n - p * n
		// if n is negative then these variables should have the opposite
		// sign, excluding radius which must be reversed
		WallCollision(
			radius, // particle radius
			m, // rigid body mass
			disp, // particle displacement
			Iinv, // rigid body inverse inertia matrix
			negP, // old particle position
			n, // wall normal
			-minPos, // wall plane
			CM, // rigid body center of mass
			force, // total force acting on rigid body
			v, // rigid body linear velocity
			w);  // rigid body angular velocity
		return true;
	}

	else if (p.y < minPos.y + radius)
	{
		float4 n = make_float4(0, -1, 0, 0);
		const float4 negP = -1 * p;
		// in case of negative n, p and wallPos must also be negative
		// so that they will be used with the correct sign inside the 
		// function CM += wallPos * n - radius * n - p * n
		// if n is negative then these variables should have the opposite
		// sign, excluding radius which must be reversed
		WallCollision(
			radius, // particle radius
			m, // rigid body mass
			disp, // particle displacement
			Iinv, // rigid body inverse inertia matrix
			negP, // old particle position
			n, // wall normal
			-minPos, // wall plane
			CM, // rigid body center of mass
			force, // total force acting on rigid body
			v, // rigid body linear velocity
			w);  // rigid body angular velocity
		return true;
	}
	else if (p.z < minPos.z + radius)
	{
		float4 n = make_float4(0, 0, -1, 0);
		const float4 negP = -1 * p;
		// in case of negative n, p and wallPos must also be negative
		// so that they will be used with the correct sign inside the 
		// function CM += wallPos * n - radius * n - p * n
		// if n is negative then these variables should have the opposite
		// sign, excluding radius which must be reversed
		WallCollision(
			radius, // particle radius
			m, // rigid body mass
			disp, // particle displacement
			Iinv, // rigid body inverse inertia matrix
			negP, // old particle position
			n, // wall normal
			-minPos, // wall plane
			CM, // rigid body center of mass
			force, // total force acting on rigid body
			v, // rigid body linear velocity
			w);  // rigid body angular velocity
		return true;
	}
	return false;
}

bool ParticleSystem::CheckWallCollisions(
	const float &radius,
	const float4 &CM)
{
	if (CM.x > maxPos.x - radius)return true;
	if (CM.x < minPos.x + radius)return true;
	if (CM.y > maxPos.y - radius)return true;
	if (CM.y < minPos.y + radius)return true;
	if (CM.z > maxPos.z - radius)return true;
	if (CM.z < minPos.z + radius)return true;
	return false;

}

void ParticleSystem::findWallCollisions()
{
	int totalParticlesProcessed = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		const float4 CM = CM_CPU[index];
		const float radius = bunnyRadius;
		//		if (CheckWallCollisions(radius, CM))
		if (1)
		{
			const glm::mat3 Iinv = Iinv_CPU[index];
			const float m = bunnyMass;
			for (int i = totalParticlesProcessed; i < totalParticlesProcessed + particlesPerObjectThrown[index]; i++)
			{
				if (HandleWallCollisions(m_params.particleRadius, m,
					POS_CPU[i], displacement[i],
					V_CPU[index], W_CPU[index],
					Iinv, CM_CPU[index], F_CPU[index]))
				{
					COL_CPU[i] = make_float4(1, 1, 1, 0);
					return;
				}
			}
		}
		totalParticlesProcessed += particlesPerObjectThrown[index];
	}
}

void ParticleSystem::findParticleCollisions()
{
	int totalParticlesProcessed = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int i = totalParticlesProcessed; i < totalParticlesProcessed + particlesPerObjectThrown[index]; i++)
		{
			for (int j = totalParticlesProcessed + particlesPerObjectThrown[index]; j < m_numParticles; j++)
			{
				int rigidBodyIndex = indexRB[j];
				if (testParticleCollision(CM_CPU[index] + displacement[i],
					CM_CPU[rigidBodyIndex] + displacement[j],
					m_params.particleRadius,
					m_params.particleRadius,
					CM_CPU[index]))
				{
					float4 cp, cn;
					findExactContactPoint(CM_CPU[index] + displacement[i],
						CM_CPU[rigidBodyIndex] + displacement[j],
						m_params.particleRadius,
						m_params.particleRadius,
						cp, cn);
					float4 r1 = cp - CM_CPU[index];
					float4 r2 = cp - CM_CPU[rigidBodyIndex];
					float impulse = computeImpulseMagnitude(
						V_CPU[index], V_CPU[rigidBodyIndex],
						W_CPU[index], W_CPU[rigidBodyIndex],
						r1, r2,
						Iinv_CPU[index], Iinv_CPU[rigidBodyIndex],
						bunnyMass, bunnyMass,
						cn);

					float4 impulseVector = cn * impulse;
					glm::vec3 rA(r1.x, r1.y, r1.z);
					glm::vec3 rB(r2.x, r2.y, r2.z);
					glm::vec3 impulseVectorGLM(impulseVector.x, impulseVector.y, impulseVector.z);

					F_CPU[index] += impulseVector;
					glm::vec3 T1 = Iinv_CPU[index] * glm::cross(rA, impulseVectorGLM);
					float4 angularImpulse = make_float4(T1.x, T1.y, T1.z, 0);
					T_CPU[index] += angularImpulse;
					V_CPU[index] += impulseVector / bunnyMass;
					W_CPU[index] += angularImpulse;

					F_CPU[rigidBodyIndex] -= impulseVector;
					glm::vec3 T2 = Iinv_CPU[rigidBodyIndex] * glm::cross(rB, impulseVectorGLM * (-1.f));
					angularImpulse = make_float4(T2.x, T2.y, T2.z, 0);
					T_CPU[rigidBodyIndex] += angularImpulse;
					V_CPU[rigidBodyIndex] -= impulseVector / bunnyMass;
					W_CPU[rigidBodyIndex] += angularImpulse;
					
					COL_CPU[i] = make_float4(1, 0, 0, 0);
					COL_CPU[j] = make_float4(1, 0, 0, 0);
				}
			}
		}
		totalParticlesProcessed += particlesPerObjectThrown[index];
	}
}

void ParticleSystem::integrateRigidBody(
	const float &dt,
	const float &r,
	const float &m,
	float4 &linearMomentum,
	float4 &linearVelocity,
	float4 &angularMomentum,
	float4 &angularVelocity,
	float4 &massCenter,
	float4 &force,
	float4 &torque,
	glm::mat3 &IinvCurrent,
	glm::mat3 &IinvOriginal,
	glm::quat &quaternion)
{
	massCenter += linearVelocity * dt;
	glm::vec3 w(angularVelocity.x, angularVelocity.y, angularVelocity.z);
	glm::vec3 normalizedW = normalize(w);
	float theta = glm::length(w);
	if (theta > 0.00001)
	{
		quaternion.w = cos(theta / 2.f);
		quaternion.x = sin(theta / 2.f) * normalizedW.x;
		quaternion.y = sin(theta / 2.f) * normalizedW.y;
		quaternion.z = sin(theta / 2.f) * normalizedW.z;
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
	IinvCurrent = rot * IinvOriginal * transpose(rot);
	force = make_float4(0, 0, 0, 0);
	torque = make_float4(0, 0, 0, 0);
}

void ParticleSystem::integrateCPU(const float &dt)
{
	int totalParticlesProcessed = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		integrateRigidBody(
			dt,
			bunnyRadius,
			bunnyMass,
			P_CPU[index],
			V_CPU[index],
			L_CPU[index],
			W_CPU[index],
			CM_CPU[index],
			F_CPU[index],
			T_CPU[index],
			Iinv_CPU[index],
			bunnyInertia,
			Q_CPU[index]);

		// update particle displacement
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			glm::quat quaternion = Q_CPU[index];
			float4 tempPos = displacement[particle + totalParticlesProcessed];
			glm::vec4 pos = glm::vec4(tempPos.x, tempPos.y, tempPos.z, tempPos.w);
			glm::mat4 rot = mat4_cast(quaternion);
			pos = rot * pos;
			tempPos = make_float4(pos.x, pos.y, pos.z, pos.w);
			displacement[particle + totalParticlesProcessed] = tempPos;
		}

		// update particle positions
		for (int particle = 0; particle < particlesPerObjectThrown[index]; particle++)
		{
			POS_CPU[particle + totalParticlesProcessed] = CM_CPU[index] + displacement[particle + totalParticlesProcessed];
			COL_CPU[particle + totalParticlesProcessed] = make_float4(0, 0, 1, 0);
		}
		totalParticlesProcessed += particlesPerObjectThrown[index];
	}
}

void ParticleSystem::collisionsCPU()
{
	findWallCollisions(); //check for wall collisions first
	//update particle positions
	for (int particle = 0; particle < m_numParticles; particle++)
	{
		POS_CPU[particle] = CM_CPU[indexRB[particle]] + displacement[particle];
	}

	findParticleCollisions();

	for (int particle = 0; particle < m_numParticles; particle++)
	{
		POS_CPU[particle] = CM_CPU[indexRB[particle]] + displacement[particle];
	}

}

float ParticleSystem::computeImpulseMagnitude(
	const float4 &v1,
	const float4 &w1,
	const float4 &r1,
	const glm::mat3 &IinvA,
	const float &mA,
	const float4 &n)
{
	glm::vec3 vA(v1.x, v1.y, v1.z);

	glm::vec3 wA(w1.x, w1.y, w1.z);

	glm::vec3 rA(r1.x, r1.y, r1.z);

	glm::vec3 norm(n.x, n.y, n.z);

	glm::vec3 velA = vA + glm::cross(wA, rA);
	float epsilon = 1;
	float numerator = -(1 + epsilon) * (glm::dot(velA, norm));
	float a = 1.f / mA;
	float b = glm::dot(glm::cross(IinvA * glm::cross(rA, norm), rA), norm);
	float denominator = a + b;
	float j = numerator / denominator;

	return j;
}

float ParticleSystem::computeImpulseMagnitude(
	const float4 &v1, const float4 &v2,
	const float4 &w1, const float4 &w2,
	const float4 &r1, const float4 &r2,
	const glm::mat3 &Iinv1, const glm::mat3 &Iinv2,
	const float &m1, const float &m2,
	const float4 &n)
{
	glm::vec3 vA(v1.x, v1.y, v1.z);
	glm::vec3 vB(v2.x, v2.y, v2.z);

	glm::vec3 wA(w1.x, w1.y, w1.z);
	glm::vec3 wB(w2.x, w2.y, w2.z);

	glm::vec3 rA(r1.x, r1.y, r1.z);
	glm::vec3 rB(r2.x, r2.y, r2.z);

	glm::vec3 norm(n.x, n.y, n.z);

	glm::vec3 velA = vA + glm::cross(wA, rA);
	glm::vec3 velB = vB + glm::cross(wB, rB);

	float numerator = -1.9 * (glm::dot(velA, norm) - glm::dot(velB, norm));
	float a = 1.f / m1;
	float b = 1.f / m2;
	float c = glm::dot(glm::cross(Iinv1 * glm::cross(rA, norm), rA), norm);
	float d = glm::dot(glm::cross(Iinv2 * glm::cross(rB, norm), rB), norm);
	float denominator = a + b + c + d;
	float j = numerator / denominator;

	return j;

}

void ParticleSystem::findExactContactPoint(
	const float4 &p1,
	const float4 &p2,
	const float &r1,
	const float &r2,
	float4 &cp,
	float4 &cn)
{
	float t = r1 / (r1 + r2);
	cp = p1 + (p2 - p1) * t;
	cn = normalize(p2 - p1) * (-1.f);
}

bool ParticleSystem::testParticleCollision(const float4 &p1, const float4 &p2, const float &r1, const float &r2, float4 &CM1)
{
	float4 displacementVector = p2 - p1;
	float displacementDistance = length(displacementVector);
	if (displacementDistance < r1 + r2)
	{
		float dr = (r1 + r2 - displacementDistance) / 2;
		displacementVector = normalize(displacementVector);
		CM1 -= displacementVector * dr;
		return true;
	}
	return false;
}