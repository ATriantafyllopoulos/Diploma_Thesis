#include "BVHAuxiliary.cuh"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

/*
* Change collision handling to use the exact contact point
*/
__device__
void collideSpheresBaraffExact(float3 posA, float3 posB,
float3 velA, float3 velB, //these are now the rigid body linear velocities
float3 wA, float3 wB, //these are the rigid body angular velocities
float3 CMa, float3 CMb, //rigid body center of mass
float radiusA, float radiusB,
glm::mat3 inertiaA, glm::mat3 inertiaB,
float massA, float massB,
int *numCollisions,
SimParams params,
float3* outForce,
float3* outTorque)
{
	// calculate relative position
	float3 relPos = posA - posB;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);
	glm::vec3 torque(0, 0, 0);
	if (dist < collideDist)
	{
		if (numCollisions)(*numCollisions)++; //increase number of collisions - used to change particle's color

		//calculate exact contact point
		float t = radiusA / (radiusA + radiusB);
		float3 cp = posA + t * (posB - posA);
		float3 norm = relPos / dist;

		float e = params.ARrestitution; //restitution
		glm::vec3 rA(cp.x - CMa.x, cp.y - CMa.y, cp.z - CMa.z);
		glm::vec3 rB(cp.x - CMb.x, cp.y - CMb.y, cp.z - CMb.z);
		glm::vec3 n(norm.x, norm.y, norm.z);
		glm::vec3 vA(velA.x, velA.y, velA.z), vB(velB.x, velB.y, velB.z);
		vA += glm::cross(glm::vec3(wA.x, wA.y, wA.z), rA);
		vB += glm::cross(glm::vec3(wB.x, wB.y, wB.z), rB);
		float relVel = glm::dot((vA - vB), n);
		float denom = glm::dot(glm::cross(inertiaA * glm::cross(rA, n), rA), n) +
			glm::dot(glm::cross(inertiaB * glm::cross(rB, n), rB), n) +
			1.f / massA +
			1.f / massB;
		float j = -(1 + e) * relVel / denom;
		force = norm * j;
		torque = glm::cross(rA, n * j);
	}

	*outForce = force;
	*outTorque = make_float3(torque.x, torque.y, torque.z);
}

/*
* Collide two spheres using Baraff's rigid body method
* ISSUE: calculate exact collision point if necessary
* For now assume that exact collision point for each
* rigid body is current particle
*/

__device__
float3 collideSpheresBaraff(float3 posA, float3 posB,
float3 velA, float3 velB,
float radiusA, float radiusB,
glm::mat3 inertiaA, glm::mat3 inertiaB,
float3 relA, float3 relB,
float massA, float massB,
int *numCollisions,
SimParams params)
{
	// calculate relative position
	float3 relPos = posA - posB;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		if (numCollisions)(*numCollisions)++; //increase number of collisions - used to change particle's color

		float3 norm = relPos / dist;
		float relVel = dot((velA - velB), norm);
		float e = params.ARrestitution; //restitution
		glm::vec3 rA(relA.x, relA.y, relA.z), rB(relB.x, relB.y, relB.z), n(norm.x, norm.y, norm.z);
		float denom = glm::dot(glm::cross(inertiaA * glm::cross(rA, n), rA), n) +
			glm::dot(glm::cross(inertiaB * glm::cross(rB, n), rB), n) +
			1.f / massA +
			1.f / massB;
		float j = -(1 + e) * relVel / denom;
		force = norm * j;
	}

	return force;
}

// calculate position in uniform grid
__device__ int3 calcGridPosAuxil(float3 p, SimParams params)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHashAuxil(int3 gridPos, SimParams params)
{
	gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.gridSize.y - 1);
	gridPos.z = gridPos.z & (params.gridSize.z - 1);
	return gridPos.z * params.gridSize.y * params.gridSize.x + gridPos.y * params.gridSize.x + gridPos.x;
}

__device__
void collideCellRigidBody(
int3 gridPos,
uint    index,
float3  pos,
float3  vel,
float4 *oldPos,
float4 *oldVel,
uint   *cellStart,
uint   *cellEnd,
int *rbIndices, //index of the rigid body each particle belongs to
uint *gridParticleIndex,//sorted particle indices
int rigidBodyIndex, //rigid body index corresponding to current particle
glm::mat3 *rbCurrentInertia,
glm::mat3 currentInertiaMatrix,
float4 *relativePos,
float3 currentRelativePos,
float *rbMass,
float currentMass,
float4 *rbPositions,
float3 currentCM,
float4 *rbAngularVelocity,
float3 currentAngularVelocity,
float4 *rbLinearVelocity,
float3 currentLinearVelocity,
int *numCollisions,
SimParams params,
float3 *outForce,
float3 *outTorque)
{
	uint gridHash = calcGridHashAuxil(gridPos, params);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);
	float3 torque = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			int rigidBodyIndex2 = rbIndices[gridParticleIndex[j]];
			if (j != index && (rigidBodyIndex != rigidBodyIndex2 || rigidBodyIndex == -1))// check not colliding with self and not of the same rigid body
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				float3 vel2 = make_float3(FETCH(oldVel, j));
				float3 linear2 = make_float3(rbLinearVelocity[rigidBodyIndex2]);
				glm::mat3 inertia2 = rbCurrentInertia[rigidBodyIndex2];
				float mass2 = rbMass[rigidBodyIndex2];
				float3 relPos2 = make_float3(relativePos[gridParticleIndex[j]]);
				float3 CM2 = make_float3(rbPositions[rigidBodyIndex2]);
				float3 angular2 = make_float3(rbAngularVelocity[rigidBodyIndex2]);
				// collide two spheres
				float3 localForce, localTorque;
				collideSpheresBaraffExact(
					pos, pos2,
					currentLinearVelocity, vel2,
					currentAngularVelocity, angular2,
					currentCM, CM2,
					params.particleRadius,
					params.particleRadius,
					currentInertiaMatrix,
					inertia2,
					currentMass,
					mass2,
					numCollisions,
					params,
					&localForce,
					&localTorque);
				force += localForce;
				torque += localTorque;
				//				force += collideSpheresBaraff(
				//						pos,
				//						pos2,
				//						vel,
				//						vel2,
				//						params.particleRadius,
				//						params.particleRadius,
				//						currentInertiaMatrix,
				//						inertia2,
				//						currentRelativePos,
				//						relPos2,
				//						currentMass,
				//						mass2,
				//						numCollisions,
				//						params);
			}
		}
	}

	*outForce = force;
	*outTorque = torque;
}

/*
* Kernel function to perform rigid body collision detection and handling
* TODO: sort auxiliary variables according to hash to maximize speed
*/
__global__
void collideUniformGridRigidBodiesKernel(
float4 *pForce, //total force applied to rigid body - per particle
int *rbIndices, //index of the rigid body each particle belongs to
float4 *relativePos, //particle's relative position
float4 *rbPositions, //rigid body center of mass
float4 *rbAngularVelocity, //rigid body angular velocity
float4 *rbVelocities, //rigid body linear velocity
float4 *pTorque,  //rigid body angular momentum - per particle
glm::mat3 *rbCurrentInertia, //current moment of inertia of rigid body
float *rbMass, //mass of rigid body
float4 *color,
float4 *newVel,               // output: new velocity
float4 *oldPos,               // input: sorted positions
float4 *oldVel,               // input: sorted velocities
uint   *gridParticleIndex,    // input: sorted particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(oldVel, index));

	// get address in grid
	int3 gridPos = calcGridPosAuxil(pos, params);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);
	float3 torque = make_float3(0.0f);
	int numCollisions = 0;
	uint originalIndex = gridParticleIndex[index];
	int rigidBodyIndex = rbIndices[originalIndex];
	glm::mat3 currentInertia = rbCurrentInertia[rigidBodyIndex];
	float currentMass = rbMass[rigidBodyIndex];
	float3 currentRelativePos = make_float3(relativePos[originalIndex]);
	float3 currentAngularVelocity = make_float3(rbAngularVelocity[rigidBodyIndex]);
	float3 currentCM = make_float3(rbPositions[rigidBodyIndex]);
	float3 currentVel = make_float3(rbVelocities[rigidBodyIndex]);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				float3 localForce, localTorque;
				collideCellRigidBody(
					neighbourPos,
					index,
					pos,
					vel,
					oldPos,
					oldVel,
					cellStart,
					cellEnd,
					rbIndices,
					gridParticleIndex,
					rigidBodyIndex,
					rbCurrentInertia,
					currentInertia,
					relativePos,
					currentRelativePos,
					rbMass,
					currentMass,
					rbPositions,
					currentCM,
					rbAngularVelocity,
					currentAngularVelocity,
					rbVelocities,
					currentVel,
					&numCollisions,
					params,
					&localForce,
					&localTorque);
				force += localForce;
				torque += localTorque;
			}
		}
	}

	// write new velocity back to original unsorted location


	if (rigidBodyIndex == -1)
		newVel[originalIndex] = make_float4(vel + force, 0.0f);
	else
	{
		//		rbForces[rigidBodyIndex] += make_float4(force, 0.0f);
		//		rbTorque[rigidBodyIndex] += make_float4(torque, 0);
		pForce[originalIndex] = make_float4(force, 0.0f);
		pTorque[originalIndex] = make_float4(torque, 0);
	}

	if (numCollisions)
		color[originalIndex] = make_float4(1, 0, 0, 0);
	else
		color[originalIndex] = make_float4(0, 0, 1, 0);
}

void collideUniformGridRigidBodiesWrapper(float4 *pForce, //total force applied to rigid body - per particle
	int *rbIndices, //index of the rigid body each particle belongs to
	float4 *relativePos, //particle's relative position
	float4 *rbPositions, //rigid body center of mass
	float4 *rbAngularVelocity, //rigid body angular velocity
	float4 *rbVelocities, //rigid body linear velocity
	float4 *pTorque,  //rigid body angular momentum - per particle
	glm::mat3 *rbCurrentInertia, //current moment of inertia of rigid body
	float *rbMass, //mass of rigid body
	float4 *color,
	float4 *newVel,               // output: new velocity
	float4 *oldPos,               // input: sorted positions
	float4 *oldVel,               // input: sorted velocities
	uint   *gridParticleIndex,    // input: sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	SimParams params,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	collideUniformGridRigidBodiesKernel << < gridDim, blockDim >> >(
		pForce, //total force applied to rigid body - per particle
		rbIndices, //index of the rigid body each particle belongs to
		relativePos, //particle's relative position
		rbPositions, //rigid body center of mass
		rbAngularVelocity, //rigid body angular velocity
		rbVelocities, //rigid body linear velocity
		pTorque,  //rigid body angular momentum - per particle
		rbCurrentInertia, //current moment of inertia of rigid body
		rbMass, //mass of rigid body
		color,
		newVel,               // output: new velocity
		oldPos,               // input: sorted positions
		oldVel,               // input: sorted velocities
		gridParticleIndex,    // input: sorted particle indices
		cellStart,
		cellEnd,
		numParticles,
		params);
}

__device__
float3 collideARSpheresDEM(float3 posA, float3 posB,
float3 velA,
float radiusA, float radiusB,
SimParams params)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = -velA;

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		// attraction
		force += params.attraction*relPos;
	}

	return force;
	//return - 2 * velA;
}

__device__
void collideARSpheresBaraffExact(float3 posA, float3 posB,
float3 cn, //normal pre-computed based on point's 8 neighborhood
float3 velA,
float radiusA, float radiusB,
glm::mat3 inertiaA,
float3 relA,
float massA,
int *numCollisions,
SimParams params,
float3 *localForce, float3 *localTorque)
{

	// calculate relative position
	float3 relPos = posA - posB;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);
	if (dist < collideDist)
	{
		if (numCollisions)(*numCollisions)++; //increase number of collisions - used to change particle's color
		//calculate exact contact point
		float t = radiusA / (radiusA + radiusB);
		float3 cp = posA + t * (posB - posA);

		float3 norm = relPos / dist;
		//		cnorm = cn;
		float relVel = dot(velA, norm);
		float3 CMa = posA - relA;
		float e = params.ARrestitution; //restitution
		glm::vec3 rA(cp.x - CMa.x, cp.y - CMa.y, cp.z - CMa.z);
		glm::vec3 n(norm.x, norm.y, norm.z);
		float denom = glm::dot(glm::cross(inertiaA * glm::cross(rA, n), rA), n) +
			1.f / massA;
		float j = -(1 + e) * relVel / denom;
		force = norm * j;
		*localForce += force;
		*localTorque += cross(make_float3(rA.x, rA.y, rA.z), force);
	}

}


/*
* Collide one virtual particle with an AR particle using Baraff's rigid body method
* ISSUE: calculate exact collision point if necessary
* For now assume that exact collision point for each
* rigid body is current particle
*/

__device__
void collideARSpheresBaraff(float3 posA, float3 posB,
float3 cn, //normal pre-computed based on point's 8 neighborhood
float3 velA,
float radiusA, float radiusB,
glm::mat3 inertiaA,
float3 relA,
float massA,
int *numCollisions,
SimParams params,
float3 *localForce, float3 *localTorque)
{
	// calculate relative position
	float3 relPos = posA - posB;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);
	if (dist < collideDist)
	{
		if (numCollisions)(*numCollisions)++; //increase number of collisions - used to change particle's color


		float3 norm = relPos / dist;
		//		cnorm = cn;
		float relVel = dot(velA, norm);
		float e = params.ARrestitution; //restitution
		glm::vec3 rA(relA.x, relA.y, relA.y), n(norm.x, norm.y, norm.z);
		float denom = glm::dot(glm::cross(inertiaA * glm::cross(rA, n), rA), n) +
			1.f / massA;
		float j = -(1 + e) * relVel / denom;
		force = norm * j;
		*localForce += force;
		*localTorque += cross(relA, force);
	}

	//	return force;
}

__device__
void ARCellRigidBody(
int3 gridPos,
uint index,
float3  pos,
float3  vel,
uint   *cellStart,
uint   *cellEnd,
uint *ARgridParticleIndex,//sorted AR particle indices
glm::mat3 currentInertiaMatrix,
float3 currentRelativePos,
float currentMass,
float4 *staticPos,
float4 *staticNorm, //normals associated with each AR particle
float *r_radii, //radii of all scene particles
int *numCollisions,
SimParams params,
float3 *localForce, float3 *localTorque)
{
	uint gridHash = calcGridHashAuxil(gridPos, params);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	//	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			float3 pos2 = make_float3(FETCH(staticPos, j));
			float3 vel2 = make_float3(0);
			glm::mat3 inertia2(0, 0, 0, 0, 0, 0, 0, 0, 0);
			float mass2 = 0;
			float3 relPos2 = make_float3(0);
			// collide two spheres
			//				collideARSpheresBaraff(
			//						pos,
			//						pos2,
			//						make_float3(staticNorm[ARgridParticleIndex[j]]),
			//						vel,
			//						params.particleRadius,
			//						r_radii[ARgridParticleIndex[j]],
			//						currentInertiaMatrix,
			//						currentRelativePos,
			//						currentMass,
			//						numCollisions,
			//						params,
			//						localForce, localTorque);
			collideARSpheresBaraffExact(
				pos,
				pos2,
				make_float3(staticNorm[ARgridParticleIndex[j]]),
				vel,
				params.particleRadius,
				r_radii[ARgridParticleIndex[j]],
				currentInertiaMatrix,
				currentRelativePos,
				currentMass,
				numCollisions,
				params,
				localForce, localTorque);
			//				force += collideARSpheresDEM(
			//						pos,
			//						pos2,
			//						vel,
			//						params.particleRadius,
			//						r_radii[ARgridParticleIndex[j]],
			//						params);
		}
	}

	//	return force;
}


__global__
void ARcollisionsUniformGridKernel(
int *pCountARCollions, //count AR collisions per particle
float4 *pForce, //total force applied to rigid body - per particle
int *rbIndices, //index of the rigid body each particle belongs to
float4 *relativePos, //particle's relative position
float4 *pTorque,  //rigid body angular momentum - per particle
glm::mat3 *rbCurrentInertia, //current moment of inertia of rigid body
float *rbMass, //mass of rigid body
float4 *color,
float *r_radii, //radii of all scene particles
float4 *newVel,               // output: new velocity
float4 *oldPos,               // input: sorted positions
float4 *oldVel,               // input: sorted velocities
float4 *staticPos, //positions of AR particles
float4 *staticNorm, //normals associated with each AR particle
uint   *gridParticleIndex,    // input: sorted particle indices
uint *ARgridParticleIndex,//sorted AR particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(oldVel, index));

	// get address in grid
	int3 gridPos = calcGridPosAuxil(pos, params);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);
	float3 torque = make_float3(0.0f);
	uint originalIndex = gridParticleIndex[index];
	int rigidBodyIndex = rbIndices[originalIndex];
	int numCollisions = 0;
	glm::mat3 currentInertia = rbCurrentInertia[rigidBodyIndex];
	float currentMass = rbMass[rigidBodyIndex];
	float3 currentRelativePos = make_float3(relativePos[originalIndex]);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				float3 localForce = make_float3(0, 0, 0);
				float3 localTorque = make_float3(0, 0, 0);
				ARCellRigidBody(
					gridPos,
					index,
					pos,
					vel,
					cellStart,
					cellEnd,
					ARgridParticleIndex,//sorted particle indices
					currentInertia,
					currentRelativePos,
					currentMass,
					staticPos,
					staticNorm, //normals associated with each AR particle
					r_radii, //radii of all scene particles
					&numCollisions,
					params,
					&localForce,
					&localTorque);
				force += localForce;
				torque += localTorque;
				//				torque += cross(make_float3(relativePos[originalIndex]), localForce);
			}
		}
	}
	if (numCollisions)
		color[originalIndex] = make_float4(0, 1, 0, 0);
	if (rigidBodyIndex == -1)
		newVel[originalIndex] += make_float4(force, 0.0f);
	else
	{
		pForce[originalIndex] = make_float4(force, 0.0f);
		pTorque[originalIndex] = make_float4(torque, 0);
		pCountARCollions[originalIndex] = numCollisions;
	}

}

void ARcollisionsUniformGridWrapper(
	int *pCountARCollions, //count AR collisions per particle
	float4 *pForce, //total force applied to rigid body - per particle
	int *rbIndices, //index of the rigid body each particle belongs to
	float4 *relativePos, //particle's relative position
	float4 *pTorque,  //rigid body angular momentum - per particle
	glm::mat3 *rbCurrentInertia, //current moment of inertia of rigid body
	float *rbMass, //mass of rigid body
	float4 *color,
	float *r_radii, //radii of all scene particles
	float4 *newVel,               // output: new velocity
	float4 *oldPos,               // input: sorted positions
	float4 *oldVel,               // input: sorted velocities
	float4 *staticPos, //positions of AR particles
	float4 *staticNorm, //normals associated with each AR particle
	uint   *gridParticleIndex,    // input: sorted particle indices
	uint *ARgridParticleIndex,//sorted AR particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	SimParams params,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	ARcollisionsUniformGridKernel << < gridDim, blockDim >> >(
		pCountARCollions, //count AR collisions per particle
		pForce, //total force applied to rigid body - per particle
		rbIndices, //index of the rigid body each particle belongs to
		relativePos, //particle's relative position
		pTorque,  //rigid body angular momentum - per particle
		rbCurrentInertia, //current moment of inertia of rigid body
		rbMass, //mass of rigid body
		color,
		r_radii, //radii of all scene particles
		newVel,               // output: new velocity
		oldPos,               // input: sorted positions
		oldVel,               // input: sorted velocities
		staticPos, //positions of AR particles
		staticNorm, //normals associated with each AR particle
		gridParticleIndex,    // input: sorted particle indices
		ARgridParticleIndex,
		cellStart,
		cellEnd,
		numParticles,
		params);
}


__device__
void FindRigidBodyCollisionsUniformGridCell(
int3 gridPos,
uint index,
uint originalIndex,
float3 pos,
float4 *oldPos,
uint *cellStart,
uint *cellEnd,
uint *gridParticleIndex,//sorted particle indices
int rigidBodyIndex, //rigid body index corresponding to current particle
int *rbIndices, //index of the rigid body each particle belongs to
int *collidingRigidBodyIndex, // index of rigid body of contact
int *collidingParticleIndex, // index of particle of contact
float *contactDistance, // penetration distance
int *numCollisions,
SimParams params)
{
	uint gridHash = calcGridHashAuxil(gridPos, params);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);
	// contact distance is unsorted
	float maxDistance = contactDistance[originalIndex]; // maximum contact distance up until now

	float collisionThreshold = 2 * params.particleRadius; // assuming all particles have the same radius

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			int originalIndex_2 = gridParticleIndex[j];
			int rigidBodyIndex_2 = rbIndices[originalIndex_2];
			if (j != index && (rigidBodyIndex != rigidBodyIndex_2))// check not colliding with self and not of the same rigid body
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				float dist = length(pos - pos2); // distance between two particles
				if (collisionThreshold - dist > maxDistance)
				{
					maxDistance = collisionThreshold - dist;
					contactDistance[originalIndex] = maxDistance;
					collidingParticleIndex[originalIndex] = originalIndex_2;
					collidingRigidBodyIndex[originalIndex] = rigidBodyIndex_2;
					(*numCollisions)++;
				}
			}
		}
	}
}

/*
* Kernel function to perform rigid body collision detection
*/
__global__
void FindRigidBodyCollisionsUniformGridKernel(
int *rbIndices, // index of the rigid body each particle belongs to
int *collidingRigidBodyIndex, // index of rigid body of contact
int *collidingParticleIndex, // index of particle of contact
float *contactDistance, // penetration distance
float4 *color, // particle color
float4 *oldPos,  // sorted positions
uint   *gridParticleIndex, // sorted particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));

	// get address in grid
	int3 gridPos = calcGridPosAuxil(pos, params);

	// examine neighbouring cells
	int numCollisions = 0;
	uint originalIndex = gridParticleIndex[index];
	int rigidBodyIndex = rbIndices[originalIndex];
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				FindRigidBodyCollisionsUniformGridCell(
					neighbourPos,
					index,
					originalIndex,
					pos,
					oldPos,
					cellStart,
					cellEnd,
					gridParticleIndex,//sorted particle indices
					rigidBodyIndex, //rigid body index corresponding to current particle
					rbIndices, //index of the rigid body each particle belongs to
					collidingRigidBodyIndex, // index of rigid body of contact
					collidingParticleIndex, // index of particle of contact
					contactDistance, // penetration distance
					&numCollisions, // counting number of collisions
					params);
			}
		}
	}

	if (numCollisions)
		color[originalIndex] = make_float4(1, 0, 0, 0);
	/*else
		color[originalIndex] = make_float4(0, 0, 1, 0);*/
}

void FindRigidBodyCollisionsUniformGridWrapper(
	int *rbIndices, // index of the rigid body each particle belongs to
	int *collidingRigidBodyIndex, // index of rigid body of contact
	int *collidingParticleIndex, // index of particle of contact
	float *contactDistance, // penetration distance
	float4 *color, // particle color
	float4 *oldPos,  // sorted positions
	uint   *gridParticleIndex, // sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	SimParams params,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	FindRigidBodyCollisionsUniformGridKernel << < gridDim, blockDim >> >(
		rbIndices, // index of the rigid body each particle belongs to
		collidingRigidBodyIndex, // index of rigid body of contact
		collidingParticleIndex, // index of particle of contact
		contactDistance, // penetration distance
		color, // particle color
		oldPos, // sorted positions
		gridParticleIndex, // sorted particle indices
		cellStart,
		cellEnd,
		numParticles,
		params);
}


__device__
void FindAugmentedRealityCollisionsUniformGridCell(
int3 gridPos, // cell to check
uint originalIndex, // unsorted index of current particle
uint index, // sorted index of current particle
float3 pos, // current particle position
float4 *ARPos, // sorted scene particle positions
uint *gridParticleIndexAR, // sorted scene particle indices
uint *cellStart, // scene cell start
uint *cellEnd, // scene cell end
int *collidingParticleIndex, // index of scene particle of contact
float *contactDistance, // penetration distance
SimParams params,
int *collisionCounter)
{
	uint gridHash = calcGridHashAuxil(gridPos, params);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);
	// contact distance is unsorted
	float maxDistance = contactDistance[originalIndex]; // maximum contact distance up until now

	float collisionThreshold = 2 * params.particleRadius; // assuming all particles have the same radius

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j < endIndex; j++)
		{
			//int originalIndexAR = gridParticleIndexAR[j];
			float3 pos2 = make_float3(FETCH(ARPos, j));
			float dist = length(pos - pos2); // distance between two particles
			if (collisionThreshold - dist > maxDistance)
			{
				maxDistance = collisionThreshold - dist;
				contactDistance[originalIndex] = maxDistance;
				collidingParticleIndex[originalIndex] = gridParticleIndexAR[j];
				(*collisionCounter)++;
			}
		}
	}
}

/*
* Kernel function to perform rigid body vs real scene collision detection
*/
__global__
void FindAugmentedRealityCollisionsUniformGridKernel(
int *collidingParticleIndex, // index of particle of contact
float *contactDistance, // penetration distance
float4 *contactNormal, // contact normal
float4 *color,  // particle color
float4 *oldPos,  // unsorted positions
float4 *ARPos,  // sorted augmented reality positions
uint   *gridParticleIndex, // sorted particle indices
uint   *gridParticleIndexAR, // sorted scene particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles,
SimParams params)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));

	// get address in grid
	int3 gridPos = calcGridPosAuxil(pos, params);
	uint originalIndex = gridParticleIndex[index];
	// examine neighbouring cells
	int collisionCounter = 0;
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				FindAugmentedRealityCollisionsUniformGridCell(
					neighbourPos, // cell to check
					originalIndex, // unsorted index of current particle
					index, // sorted index of current particle
					pos, // current particle position
					ARPos, // sorted scene particle positions
					gridParticleIndexAR, // sorted scene particle indices
					cellStart, // scene cell start
					cellEnd, // scene cell end
					collidingParticleIndex, // index of particle of contact
					contactDistance, // penetration distance
					params,
					&collisionCounter);
			}
		}
	}
	// manually override collision detection
	// collide with virtual plane at y = 0.5
	/*if (pos.y < 0.2)
	{
		contactDistance[originalIndex] = 0.2 - pos.y;
		contactNormal[originalIndex] = make_float4(0, 1, 0, 0);
		collisionCounter++;
	}*/
	/*else if (pos.x < -0.5)
	{
		contactDistance[originalIndex] = -1 - pos.x;
		contactNormal[originalIndex] = make_float4(1, 0, 0, 0);
		collisionCounter++;
	}
	else if (pos.x > 0.5)
	{
		contactDistance[originalIndex] = pos.x - 1;
		contactNormal[originalIndex] = make_float4(-1, 0, 0, 0);
		collisionCounter++;
	}
	else if (pos.z < -0.5)
	{
		contactDistance[originalIndex] = -0.5 - pos.z;
		contactNormal[originalIndex] = make_float4(0, 0, 1, 0);
		collisionCounter++;
	}
	else if (pos.z > 0.5)
	{
		contactDistance[originalIndex] = pos.z - 0.5;
		contactNormal[originalIndex] = make_float4(0, 0, -1, 0);
		collisionCounter++;
	}*/

	/*float3 n = make_float3(0, sqrt(2.f) / 2, sqrt(2.f) / 2);
	if (dot(n, pos) < 0)
	{
		contactDistance[originalIndex] = -dot(n, pos);
		collisionCounter++;
	}*/
	// collide with virtual plane at 45 degrees
	if (collisionCounter)
		color[originalIndex] = make_float4(0, 1, 0, 0);
	else
		color[originalIndex] = make_float4(0, 0, 1, 0);
}


void FindAugmentedRealityCollisionsUniformGridWrapper(
	int *collidingParticleIndex, // index of particle of contact
	float *contactDistance, // penetration distance
	float4 *contactNormal, // contact normal
	float4 *color,  // particle color
	float4 *oldPos,  // unsorted positions
	float4 *ARPos,  // sorted augmented reality positions
	uint   *gridParticleIndex, // sorted particle indices
	uint   *gridParticleIndexAR, // sorted scene particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	uint	numberOfRangeData,
	SimParams params,
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	FindAugmentedRealityCollisionsUniformGridKernel << < gridDim, blockDim >> >(
		collidingParticleIndex, // index of particle of contact
		contactDistance, // penetration distance
		contactNormal, // contact normal
		color,  // particle color
		oldPos, // unsorted positions
		ARPos,  // sorted augmented reality positions
		gridParticleIndex, // sorted particle indices
		gridParticleIndexAR, // sorted scene particle indices
		cellStart,
		cellEnd,
		numParticles,
		params);

	//// copy particle variables to CPU
	//float4 *particlePosition_CPU = new float4[numParticles];
	//checkCudaErrors(cudaMemcpy(particlePosition_CPU, oldPos, numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
	//float4 *position_CPU = new float4[numberOfRangeData];
	//checkCudaErrors(cudaMemcpy(position_CPU, ARPos, numberOfRangeData * sizeof(float4), cudaMemcpyDeviceToHost));

	//// copy contact info to CPU - one contact per particle
	//float *contactDistance_CPU = new float[numParticles];
	//int *collidingParticleIndex_CPU = new int[numParticles];

	//checkCudaErrors(cudaMemcpy(contactDistance_CPU, contactDistance, numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(collidingParticleIndex_CPU, collidingParticleIndex, numParticles * sizeof(int), cudaMemcpyDeviceToHost));

	//bool foundCollision = false;
	//int collisionIndex = -1;
	//for (int i = 0; i < numParticles; i++)
	//	if (contactDistance_CPU[i] > 0)
	//	{
	//		foundCollision = true;
	//		collisionIndex = i;
	//		std::cout << "GPU Collision found @ " << collidingParticleIndex_CPU[i] << std::endl;
	//		std::cout << "GPU distance is " << contactDistance_CPU[i] << std::endl;
	//		float collisionThreshold = 2 * params.particleRadius;
	//		std::cout << "CPU Collision threshold is " << collisionThreshold << std::endl;
	//		float4 sceneParticle = position_CPU[collidingParticleIndex_CPU[i]];
	//		float4 rigidBodyParticle = particlePosition_CPU[collisionIndex];
	//		float distance = length(make_float3(sceneParticle) - make_float3(rigidBodyParticle));
	//		std::cout << "Distance between particles is " << distance << std::endl;
	//		std::cout << "CPU distance is " << collisionThreshold - distance << std::endl;
	//		break;
	//	}
	//		

	//if (foundCollision)
	//{
	//	float4 rigidBodyParticle = particlePosition_CPU[collisionIndex];
	//	for (int i = 0; i < numberOfRangeData; i++)
	//	{ 
	//		float4 sceneParticle = position_CPU[i];
	//		float distance = 2 * params.particleRadius - length(make_float3(sceneParticle) - make_float3(rigidBodyParticle));
	//		if (distance > 0)
	//		{
	//			std::cout << "CPU Collision found @ " << i << " and distance is " << distance << std::endl;
	//		}
	//	}
	//}
	//delete particlePosition_CPU;
	//delete position_CPU;
	//delete contactDistance_CPU;
	//delete collidingParticleIndex_CPU;

}
