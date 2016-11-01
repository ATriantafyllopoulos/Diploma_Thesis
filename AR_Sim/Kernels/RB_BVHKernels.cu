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

inline __device__ float4 maxOpF4(const float4 &a, const float4 &b)
{
	float4 res;
	res.x = (b.x > a.x) ? b.x : a.x;
	res.y = (b.y > a.y) ? b.y : a.y;
	res.z = (b.z > a.z) ? b.z : a.z;
	res.w = (b.w > a.w) ? b.w : a.w;
	return res;
}


inline __device__
float3 collideStaticSpheresBVH(float4 posA, float4 posB, float4 *correctedPos, float4 *CM,
float3 velA, float3 velB,
float radiusA, float radiusB,
SimParams params)
{
	// calculate relative position
	float3 relPos = make_float3(posB - posA);

	float dist = length(relPos); //distance between two centers
	float collideDist = radiusA + radiusB; //sum of radii

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		float3 norm = relPos / dist; //norm points from A(queryParticle) to B(leafParticle)

		// relative velocity
		float3 relVel = velB - velA;

		//if collision velocity is negative then they are moving towards the direction of the norm
		float collisionVelocity = dot(relVel, norm);

		// relative tangential velocity
		float3 tanVel = relVel - (collisionVelocity * norm);

		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		//		force = -300.f*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping*relVel;
		//		force += 0.5f*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		//		force += 0.5*tanVel;
		// attraction
		force += params.attraction*relPos;

		//correct positions
		//the normal is pointing towards one another so we must move them to the other direction
		//this is why the offset must be negative
		float offset = -(collideDist - dist) / 2.f;
		if (length(make_float3(*CM - posB)) < length(make_float3(*CM - posA)))
			offset = -offset;
		*correctedPos = make_float4(norm * offset, 0);

		//		if (collisionVelocity > 0 && (dist > radiusA || dist > radiusB)) //if they are moving towards one another
		//			offset -= collisionVelocity * 0.01 / 2.f;
		//		//they are already halfway in so we need to push them in the opposite direction
		//		//now the normal points away from one another so the offset must be positive
		//		if(dist < radiusA && dist < radiusB)
		//			offset = -offset;
		//		if(collisionVelocity < 0 && dist < radiusA && dist < radiusB)
		//			offset += collisionVelocity * 0.01 / 2.f;
		//		*correctedPos = make_float4(norm * offset, 0);

		//		float offset;
		//		if (dot(norm, *CMnorm) < 0) //move towards the norm direction
		//		{
		//			offset = (collideDist - dist) / 2.f;
		//			if (collisionVelocity < 0) //velocity is pointing towards the other direction
		//				offset += collisionVelocity * 0.01 / 2.f;
		//		}
		//		else
		//		{
		//			offset = -(collideDist - dist) / 2.f;
		//			if (collisionVelocity < 0) //velocity is pointing towards the other direction
		//				offset -= collisionVelocity * 0.01 / 2.f;
		//		}
		//		*correctedPos = make_float4(norm * offset, 0);
	}

	return force;
}

__global__
void collideBVHSoARigidBody(float4 *color, //particle's color, only used for testing purposes
float4 *pForce, //total force applied to rigid body
int *rigidBodyIndex, //index of the rigid body each particle belongs to
float4 *pPositions, //rigid body center of mass
float4 *positions, //unsorted particle positions
float4 *relativePos, //particle's relative position
float4 *pTorque,  //rigid body angular momentum
float4 *vel, //particles original velocity, updated after all collisions are handled
bool *isLeaf, //array containing a flag to indicate whether node is leaf
int *leftIndices, //array containing indices of the left children of each node
int *rightIndices, //array containing indices of the right children of each node
int *minRange, //array containing minimum (sorted) leaf covered by each node
int *maxRange, //array containing maximum (sorted) leaf covered by each node
float4 *CMs, //array containing centers of mass for each leaf
AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
int *sortedIndices, //array containing corresponding unsorted indices for each leaf
float *radii, //radii of all nodes - currently the same for all particles
int numParticles, //number of virtual particles
SimParams params) //simulation parameters
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //handling particle #(index)
	if (index >= numParticles) return;

	float3 force = make_float3(0.0f);


	int queryIndex = sortedIndices[index]; //load original particle index once
	float3 queryVel = make_float3(FETCH(vel, queryIndex)); //load particle original velocity once
	int stack[64]; //using stack of indices
	int* stackPtr = stack;
	*stackPtr++ = -1; //push -1 at beginning so that thread will return when stack is empty 
	int numCollisions = 0; //count number of collisions
	//Traverse nodes starting from the root.
	//load leaf positions once - better to use this array as successive threads will access successive memory positions
	float4 queryPos = CMs[index];
	float queryRad = radii[index]; //load particle radius once - currently all particles have the same radius
	int SoAindex = numParticles; //start at root
	int queryRigidBody = rigidBodyIndex[queryIndex]; //rigid body index corresponding to query particle
	//if (queryRigidBody == -1) return; //if this is an independent virtual particle
	float3 torque = make_float3(0, 0, 0);
	AABB queryBound = bounds[index]; //load particle's bounding volume once
	float4 correctedPos = make_float4(0, 0, 0, 0);

	float4 rbCM = queryPos - relativePos[queryIndex];
	do
	{
		//Check each child node for overlap.
		int leftChildIndex = leftIndices[SoAindex]; //load left child index once
		int rightChildIndex = rightIndices[SoAindex]; //load right child index once

		bool overlapL = checkOverlap(queryBound, bounds[leftChildIndex]); //check overlap with left child
		bool overlapR = checkOverlap(queryBound, bounds[rightChildIndex]); //check overlap with right child

		//indices are unique for each node, internal or leaf, as they are stored in the same array
		if (leftChildIndex == index) //left child is current leaf
			overlapL = false;
		if (rightChildIndex == index) //right child is current leaf
			overlapR = false;
		bool isLeftLeaf = isLeaf[leftChildIndex]; //load left child's leaf flag once
		bool isRightLeaf = isLeaf[rightChildIndex]; //load right child's leaf flag once

		//Query overlaps a leaf node => report collision
		if (overlapL && isLeftLeaf && (queryRigidBody != rigidBodyIndex[sortedIndices[leftChildIndex]] || queryRigidBody == -1))
		{
			float4 localCorrection = make_float4(0.f);

			float3 localForce = collideStaticSpheresBVH(queryPos, CMs[leftChildIndex], &localCorrection, &rbCM,
				queryVel, make_float3(FETCH(vel, sortedIndices[leftChildIndex])),
				queryRad, radii[leftChildIndex],
				params);
			//correctedPos = maxOpF4(localCorrection, correctedPos);
			correctedPos += localCorrection;
			force += localForce;
			torque += cross(make_float3(relativePos[queryIndex]), localForce);
			if (length(make_float3(CMs[leftChildIndex] - queryPos)) < queryRad + radii[leftChildIndex])
			{
				numCollisions++;
				color[queryIndex] = make_float4(1, 0, 0, 0);
			}
		}


		if (overlapR && isRightLeaf && (queryRigidBody != rigidBodyIndex[sortedIndices[rightChildIndex]] || queryRigidBody == -1))
		{
			float4 localCorrection = make_float4(0.f);

			float3 localForce = collideStaticSpheresBVH(queryPos, CMs[rightChildIndex], &localCorrection, &rbCM,
				queryVel, make_float3(FETCH(vel, sortedIndices[rightChildIndex])),
				queryRad, radii[rightChildIndex],
				params);
			correctedPos = maxOpF4(localCorrection, correctedPos);
			correctedPos += localCorrection;
			force += localForce;
			torque += cross(make_float3(relativePos[queryIndex]), localForce);
			if (length(make_float3(CMs[rightChildIndex] - queryPos)) < queryRad + radii[rightChildIndex])
			{
				numCollisions++;
				color[queryIndex] = make_float4(1, 0, 0, 0);
			}
		}

		//Query overlaps an internal node => traverse
		bool traverseL = (overlapL && !isLeftLeaf);
		bool traverseR = (overlapR && !isRightLeaf);

		if (!traverseL && !traverseR)
			SoAindex = *--stackPtr; //pop
		else
		{
			SoAindex = (traverseL) ? leftChildIndex : rightChildIndex;
			if (traverseL && traverseR)
				*stackPtr++ = rightChildIndex; // push
		}
	} while (SoAindex != -1);

	if (!numCollisions)
		color[queryIndex] = make_float4(1, 1, 1, 0);
	// collide with cursor sphere
	//	float4 colPos = make_float4(params.colliderPos.x, params.colliderPos.y, params.colliderPos.z, 1);
	//	if (length(queryPos - colPos) <= queryRad + params.colliderRadius)
	//		force += collideSpheresBVH(queryPos,
	//		colPos,
	//		queryVel,
	//		make_float3(0.0f, 0.0f, 0.0f),
	//		queryRad,
	//		params.colliderRadius,
	//		params);

	if (queryRigidBody == -1)
	{
		//positions[queryIndex] = queryPos + correctedPos;
		//ISSUE: particles's velocity changes before the other particles have processed collision with it
		//unlikely, but it may happen
		//there should be a second, potentially sorted, array of velocities
		//this is not an issue for rigid bodies so its priority is low
		vel[queryIndex] = make_float4(queryVel + force, 0);
	}
	else
	{
		pForce[queryIndex] = make_float4(force, 0);
		pTorque[queryIndex] = make_float4(torque, 0);
		pPositions[queryIndex] = correctedPos;
	}
}

void collideBVHSoARigidBodyWrapper(float4 *color, //particle's color, only used for testing purposes
	float4 *pForce, //total force applied to rigid body - uniquely tied to each particle
	int *rigidBodyIndex, //index of the rigid body each particle belongs to
	float4 *pPositions, //rigid body center of mass - uniquely tied to each particle
	float4 *positions, //unsorted particle positions
	float4 *relativePos, //particle's relative position
	float4 *pTorque,  //rigid body angular momentum - uniquely tied to each particle
	float4 *vel, //particles original velocity, updated after all collisions are handled
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	float4 *CMs, //array containing centers of mass for each leaf
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, //array containing corresponding unsorted indices for each leaf
	float *radii, //radii of all nodes - currently the same for all particles
	int numParticles, //number of virtual particles
	int numThreads, //number of threads
	SimParams params)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	collideBVHSoARigidBody << < gridDim, blockDim >> >(color, //particle's color, only used for testing purposes
		pForce, //total force applied to rigid body
		rigidBodyIndex, //index of the rigid body each particle belongs to
		pPositions, //rigid body center of mass
		positions, //unsorted particle positions
		relativePos, //particle's relative position
		pTorque,  //rigid body angular momentum
		vel, //particles original velocity, updated after all collisions are handled
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		numParticles, //number of virtual particles
		params); //simulation parameters
}

/*
* Function to compute particle-scene collisions, assuming they belong to a rigid body.
*/
__global__
void staticCollideBVHSoARigidBody(float4 *dCol, //virtual particle colors
float4 *positions, //virtual particle positions
float4 *relativePos, //particle's relative position
float4 *rbTorque,  //rigid body angular momentum
float4 *rigidBodyForce, //total force applied to rigid body
float *rbMass, //total mass of rigid body
int *rigidBodyIndex, //index of the rigid body each particle belongs to
float4 *rbPositions, //rigid body center of mass
float4 *vel, //particles original velocity, updated after all collisions are handled
float4 *normals, //normals computed for each real particle using its 8-neighborhood
bool *isLeaf, //array containing a flag to indicate whether node is leaf
int *leftIndices, //array containing indices of the left children of each node
int *rightIndices, //array containing indices of the right children of each node
int *minRange, //array containing minimum (sorted) leaf covered by each node
int *maxRange, //array containing maximum (sorted) leaf covered by each node
float4 *CMs, //array containing centers of mass for each leaf
AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
int *sortedIndices, //array containing corresponding unsorted indices for each leaf
float *radii, //radii of all nodes - currently the same for all particles
int numParticles, //number of virtual particles
int numRangeData, //number of static data
SimParams params) //simulation parameters

{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //handling particle #(index)
	if (index >= numParticles) return;

	float3 force = make_float3(0.0f);

	int queryIndex = index; //testing for virtual particles
	float3 queryVel = make_float3(FETCH(vel, queryIndex)); //load particle original velocity once

	int stack[64]; //using stack of indices
	int* stackPtr = stack;
	*stackPtr++ = -1; //push -1 at beginning so that thread will return when stack is empty 
	//Traverse nodes starting from the root.
	//load leaf positions once
	float4 queryPos = positions[index];
	float queryRad = params.particleRadius; //load particle radius once - currently all particles have the same radius
	int SoAindex = numRangeData; //start at root
	int numCollisions = 0; //count number of collisions
	int queryRigidBody = rigidBodyIndex[queryIndex];
	float queryMass;
	if (queryRigidBody == -1)
		queryMass = 1; //every particle has a mass of 1
	else
		queryMass = rbMass[queryRigidBody];
	float3 torque = make_float3(0, 0, 0);
	float4 correctedPos = make_float4(0, 0, 0, 0);

	//float4 rbCM = queryPos - relativePos[queryIndex];
	do
	{
		//Check each child node for overlap.
		int leftChildIndex = leftIndices[SoAindex]; //load left child index once
		int rightChildIndex = rightIndices[SoAindex]; //load right child index once

		bool overlapL = checkOverlap(queryPos, bounds[leftChildIndex], queryRad); //check overlap with left child
		bool overlapR = checkOverlap(queryPos, bounds[rightChildIndex], queryRad); //check overlap with right child

		bool isLeftLeaf = isLeaf[leftChildIndex]; //load left child's leaf flag once
		bool isRightLeaf = isLeaf[rightChildIndex]; //load right child's leaf flag once

		//Query overlaps a leaf node => report collision
		if (overlapL && isLeftLeaf)
		{
			//force += reflect(queryVel, make_float3(normals[sortedIndices[leftChildIndex]])) * params.boundaryDamping / 10;
			/*float3 norm = make_float3(normals[sortedIndices[leftChildIndex]]);
			queryVel = reflect(queryVel, make_float3(normals[sortedIndices[leftChildIndex]]));
			force += -2.0f * norm * dot(norm, queryVel);*/
			float3 localForce = collideSpheresBVH(queryPos, CMs[leftChildIndex],
				queryVel, make_float3(0, 0, 0),
				queryRad, radii[leftChildIndex],
				params);
			//			if (queryRigidBody == -1)
			//			{
			//				queryPos = correctedPos;
			//			}
			//			float3 localNormal = make_float3(normals[sortedIndices[leftChildIndex]]);
			//			float3 localForce = -2 * dot(queryVel, localNormal) * localNormal * queryMass * params.boundaryDamping;
			force += localForce;
			torque += cross(make_float3(relativePos[queryIndex]), localForce);

			//			if (queryRigidBody == -1)
			//			{
			//				queryPos = correctedPos;
			//			}
			if (length(make_float3(CMs[leftChildIndex] - queryPos)) < queryRad + radii[leftChildIndex])
			{
				numCollisions++;
				dCol[index] = make_float4(0, 1, 0, 0);
			}
		}
		if (overlapR && isRightLeaf)
		{
			//force += reflect(queryVel, make_float3(normals[sortedIndices[rightChildIndex]])) * params.boundaryDamping / 10;
			/*float3 norm = make_float3(normals[sortedIndices[rightChildIndex]]);
			queryVel = reflect(queryVel, make_float3(normals[sortedIndices[rightChildIndex]]));
			force += -2.0f * norm * dot(norm, queryVel);*/
			float3 localForce = collideSpheresBVH(queryPos, CMs[rightChildIndex],
				queryVel, make_float3(0, 0, 0),
				queryRad, radii[rightChildIndex],
				params);
			//			if (queryRigidBody == -1)
			//			{
			//				queryPos = correctedPos;
			//			}
			//			float3 localNormal = make_float3(normals[sortedIndices[rightChildIndex]]);
			//			float3 localForce = -2 * dot(queryVel, localNormal) * localNormal * queryMass * params.boundaryDamping;
			force += localForce;
			torque += cross(make_float3(relativePos[queryIndex]), localForce);
			if (length(make_float3(CMs[rightChildIndex] - queryPos)) < queryRad + radii[rightChildIndex])
			{
				numCollisions++;
				dCol[index] = make_float4(0, 1, 0, 0);
			}
		}

		//Query overlaps an internal node => traverse
		bool traverseL = (overlapL && !isLeftLeaf);
		bool traverseR = (overlapR && !isRightLeaf);

		if (!traverseL && !traverseR)
			SoAindex = *--stackPtr; //pop
		else
		{
			SoAindex = (traverseL) ? leftChildIndex : rightChildIndex;
			if (traverseL && traverseR)
				*stackPtr++ = rightChildIndex; // push
		}
	} while (SoAindex != -1);



	if (queryRigidBody >= 0)
	{
		rigidBodyForce[queryRigidBody] += make_float4(force, 0);
		rbTorque[queryRigidBody] += make_float4(torque, 0);
		//rbPositions[queryRigidBody] += correctedPos;
		//rigidBodyForce[queryRigidBody] += make_float4(queryVel, 0);
	}
	else
	{
		//positions[index] = queryPos;// + correctedPos;
		vel[queryIndex] = make_float4(queryVel + force, 0);
	}
	//vel[queryIndex] = make_float4(queryVel, 0);

}

void staticCollideBVHSoARigidBodyWrapper(float4 *dCol, //virtual particle colors
	float4 *positions, //virtual particle positions
	float4 *relativePos, //particle's relative position
	float4 *rbTorque,  //rigid body angular momentum
	float4 *rigidBodyForce, //total force applied to rigid body
	float *rbMass, //total mass of rigid body
	int *rigidBodyIndex, //index of the rigid body each particle belongs to
	float4 *rbPositions, //rigid body center of mass
	float4 *vel, //particles original velocity, updated after all collisions are handled
	float4 *normals, //normals computed for each real particle using its 8-neighborhood
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	float4 *CMs, //array containing centers of mass for each leaf
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, //array containing corresponding unsorted indices for each leaf
	float *radii, //radii of all nodes - currently the same for all particles
	int numParticles, //number of virtual particles
	int numRangeData, //number of static data
	int numThreads, //number of threads
	SimParams params) //simulation parameters
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	staticCollideBVHSoARigidBody << < gridDim, blockDim >> >(dCol, //virtual particle colors
		positions, //virtual particle positions
		relativePos, //particle's relative position
		rbTorque,  //rigid body angular momentum
		rigidBodyForce, //total force applied to rigid body
		rbMass, //total mass of rigid body
		rigidBodyIndex, //index of the rigid body each particle belongs to
		rbPositions, //rigid body center of mass
		vel, //particles original velocity, updated after all collisions are handled
		normals, //normals computed for each real particle using its 8-neighborhood
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		numParticles, //number of virtual particles
		numRangeData, //number of static data
		params); //simulation parameters
}


/*
* Function to compute particle-particle collisions, assuming they belong to a rigid body.
*/
__device__ void collideRigidBodies(
	float4 relative_A, //center of mass of first rigid body
	float4 relative_B, //center of mass of second rigid body
	float3 vel_A, //velocity of first rigid body @ point of contact
	float3 vel_B, //velocity of second rigid body @ point of contact
	float4 contact_A, //point of contact on first rigid body
	float4 contact_B, //point of contact on second rigid body
	float radius_A, //particle radius of first rigid body
	float radius_B, //particle radius of second rigid body
	glm::mat3 currentInertia_A, //current inertia matrix of first rigid body
	glm::mat3 currentInertia_B, //current inertia matrix of second rigid body
	float mass_A, //inverse mass of first rigid body
	float mass_B, //inverse mass of second rigid body
	float3 *localForce,
	float3 *localTorque,
	float4 *localCorrection)
{
	// calculate relative position
	float3 relPos = make_float3(contact_B - contact_A);

	float dist = length(relPos); //distance between two centers
	float collideDist = radius_A + radius_B; //sum of radii

	if (dist < collideDist) //then this is a colliding contact
	{
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
		*localForce = j;
		*localTorque = cross(make_float3(relative_A), j);
		float offset = collideDist - dist;
		//		float3 relCM = CM_B - CM_A;

		*localCorrection = make_float4(0.f);
	}

}

/*
* Function to compute rigid body to rigid body collisions. We assume that there are no independent particles.
* Particles belonging to rigid bodies are used only for contact point detection.
*/
__global__
void collideBVHSoARigidBodyOnly(float4 *color, //particle's color, only used for testing purposes
float *rbMass, //inverse mass of each rigid body
glm::mat3 *rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
float4 *rbPositions, //rigid body center of mass
float4 *pForce, //per particle force applied to rigid body
int *rigidBodyIndex, //index of the rigid body each particle belongs to
float4 *pPositions, //per particle changes to rigid body center of mass
float4 *positions, //unsorted particle positions
float4 *relativePos, //particle's relative position
float4 *pTorque,  //per particle torque applied to rigid body
float4 *vel, //particle's original velocity
bool *isLeaf, //array containing a flag to indicate whether node is leaf
int *leftIndices, //array containing indices of the left children of each node
int *rightIndices, //array containing indices of the right children of each node
int *minRange, //array containing minimum (sorted) leaf covered by each node
int *maxRange, //array containing maximum (sorted) leaf covered by each node
float4 *CMs, //array containing centers of mass for each leaf
AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
int *sortedIndices, //array containing corresponding unsorted indices for each leaf
float *radii, //radii of all nodes - currently the same for all particles
int numParticles, //number of virtual particles
SimParams params) //simulation parameters
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //handling particle #(index)
	if (index >= numParticles) return;

	float3 force = make_float3(0.0f);

	int queryIndex = sortedIndices[index]; //load original particle index once
	float3 queryVel = make_float3(FETCH(vel, queryIndex)); //load particle original velocity once
	int stack[64]; //using stack of indices
	int* stackPtr = stack;
	*stackPtr++ = -1; //push -1 at beginning so that thread will return when stack is empty
	int numCollisions = 0; //count number of collisions
	//Traverse nodes starting from the root.
	//load leaf positions once - better to use this array as successive threads will access successive memory positions
	float4 queryPos = CMs[index];
	float queryRad = radii[queryIndex]; //load particle radius once - currently all particles have the same radius
	int SoAindex = numParticles; //start at root
	int queryRigidBody = rigidBodyIndex[queryIndex]; //rigid body index corresponding to query particle
	//if (queryRigidBody == -1) return; //if this is an independent virtual particle
	float3 torque = make_float3(0, 0, 0);
	AABB queryBound = bounds[index]; //load particle's bounding volume once
	float4 correctedPos = make_float4(0, 0, 0, 0);

	float4 queryRelative = relativePos[queryIndex];
	do
	{
		//Check each child node for overlap.
		int leftChildIndex = leftIndices[SoAindex]; //load left child index once
		int rightChildIndex = rightIndices[SoAindex]; //load right child index once

		bool overlapL = checkOverlap(queryBound, bounds[leftChildIndex]); //check overlap with left child
		bool overlapR = checkOverlap(queryBound, bounds[rightChildIndex]); //check overlap with right child

		//indices are unique for each node, internal or leaf, as they are stored in the same array
		if (leftChildIndex == index) //left child is current leaf
			overlapL = false;
		if (rightChildIndex == index) //right child is current leaf
			overlapR = false;
		bool isLeftLeaf = isLeaf[leftChildIndex]; //load left child's leaf flag once
		bool isRightLeaf = isLeaf[rightChildIndex]; //load right child's leaf flag once

		int leftSortedIndex = sortedIndices[leftChildIndex];
		int rightSortedIndex = sortedIndices[rightChildIndex];
		//Query overlaps a leaf node => report collision
		if (overlapL && isLeftLeaf && queryRigidBody != rigidBodyIndex[leftSortedIndex])
		{
			float4 localCorrection = make_float4(0.f);
			float3 localForce;
			float3 localTorque;
			collideRigidBodies(
				queryRelative, //center of mass of first rigid body
				relativePos[leftSortedIndex], //center of mass of second rigid body
				queryVel, //velocity of first rigid body @ point of contact
				make_float3(vel[leftSortedIndex]), //velocity of second rigid body @ point of contact
				queryPos, //point of contact on first rigid body
				CMs[leftChildIndex], //point of contact on second rigid body
				queryRad, //particle radius of first rigid body
				radii[leftSortedIndex], //particle radius of second rigid body
				rbCurrentInertia[queryRigidBody], //current inertia matrix of first rigid body
				rbCurrentInertia[rigidBodyIndex[leftSortedIndex]], //current inertia matrix of second rigid body
				rbMass[queryRigidBody], //inverse mass of first rigid body
				rbMass[rigidBodyIndex[leftSortedIndex]], //inverse mass of second rigid body
				&localForce,
				&localTorque,
				&localCorrection);
			//correctedPos = maxOpF4(localCorrection, correctedPos);
			correctedPos += localCorrection;
			force += localForce;
			torque += localTorque;
			if (length(make_float3(CMs[leftChildIndex] - queryPos)) < queryRad + radii[leftChildIndex])
			{
				numCollisions++;
				color[queryIndex] = make_float4(1, 0, 0, 0);
			}
		}
		if (overlapR && isRightLeaf && queryRigidBody != rigidBodyIndex[rightSortedIndex])
		{
			float4 localCorrection = make_float4(0.f);
			float3 localForce;
			float3 localTorque;
			collideRigidBodies(
				queryRelative, //center of mass of first rigid body
				relativePos[rightSortedIndex], //center of mass of second rigid body
				queryVel, //velocity of first rigid body @ point of contact
				make_float3(vel[rightSortedIndex]), //velocity of second rigid body @ point of contact
				queryPos, //point of contact on first rigid body
				CMs[rightChildIndex], //point of contact on second rigid body
				queryRad, //particle radius of first rigid body
				radii[rightSortedIndex], //particle radius of second rigid body
				rbCurrentInertia[queryRigidBody], //current inertia matrix of first rigid body
				rbCurrentInertia[rigidBodyIndex[rightSortedIndex]], //current inertia matrix of second rigid body
				rbMass[queryRigidBody], //inverse mass of first rigid body
				rbMass[rigidBodyIndex[rightSortedIndex]], //inverse mass of second rigid body
				&localForce,
				&localTorque,
				&localCorrection);
			//correctedPos = maxOpF4(localCorrection, correctedPos);
			correctedPos += localCorrection;
			force += localForce;
			torque += localTorque;
			if (length(make_float3(CMs[rightChildIndex] - queryPos)) < queryRad + radii[rightChildIndex])
			{
				numCollisions++;
				color[queryIndex] = make_float4(1, 0, 0, 0);
			}
		}

		//Query overlaps an internal node => traverse
		bool traverseL = (overlapL && !isLeftLeaf);
		bool traverseR = (overlapR && !isRightLeaf);

		if (!traverseL && !traverseR)
			SoAindex = *--stackPtr; //pop
		else
		{
			SoAindex = (traverseL) ? leftChildIndex : rightChildIndex;
			if (traverseL && traverseR)
				*stackPtr++ = rightChildIndex; // push
		}
	} while (SoAindex != -1);

	if (!numCollisions)
		color[queryIndex] = make_float4(1, 1, 1, 0);

	pForce[queryIndex] = make_float4(force, 0);
	pTorque[queryIndex] = make_float4(torque, 0);
	pPositions[queryIndex] = correctedPos;
}

void collideBVHSoARigidBodyOnlyWrapper(float4 *color, //particle's color, only used for testing purposes
	float *rbMass, //inverse mass of each rigid body
	glm::mat3 *rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
	float4 *rbPositions, //rigid body center of mass
	float4 *pForce, //per particle force applied to rigid body
	int *rigidBodyIndex, //index of the rigid body each particle belongs to
	float4 *pPositions, //per particle changes to rigid body center of mass
	float4 *positions, //unsorted particle positions
	float4 *relativePos, //particle's relative position
	float4 *pTorque,  //per particle torque applied to rigid body
	float4 *vel, //particle's original velocity
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	float4 *CMs, //array containing centers of mass for each leaf
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, //array containing corresponding unsorted indices for each leaf
	float *radii, //radii of all nodes - currently the same for all particles
	int numParticles, //number of virtual particles
	SimParams params, //simulation parameters
	int numThreads)
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	collideBVHSoARigidBodyOnly << < gridDim, blockDim >> >(color, //particle's color, only used for testing purposes
		rbMass, //inverse mass of each rigid body
		rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
		rbPositions, //rigid body center of mass
		pForce, //per particle force applied to rigid body
		rigidBodyIndex, //index of the rigid body each particle belongs to
		pPositions, //per particle changes to rigid body center of mass
		positions, //unsorted particle positions
		relativePos, //particle's relative position
		pTorque,  //per particle torque applied to rigid body
		vel, //particle's original velocity
		isLeaf, //array containing a flag to indicate whether node is leaf
		leftIndices, //array containing indices of the left children of each node
		rightIndices, //array containing indices of the right children of each node
		minRange, //array containing minimum (sorted) leaf covered by each node
		maxRange, //array containing maximum (sorted) leaf covered by each node
		CMs, //array containing centers of mass for each leaf
		bounds, //array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, //array containing corresponding unsorted indices for each leaf
		radii, //radii of all nodes - currently the same for all particles
		numParticles, //number of virtual particles
		params); //simulation parameters
}


__global__ void FindRigidBodyCollisionsBVHKernel(
	float4 *color, // Input: particle's color, only used for testing purposes
	int *rigidBodyIndex, // Input: index of the rigid body each particle belongs to
	bool *isLeaf, // Input: array containing a flag to indicate whether node is leaf
	int *leftIndices, // Input:  array containing indices of the left children of each node
	int *rightIndices, // Input: array containing indices of the right children of each node
	int *minRange, // Input: array containing minimum (sorted) leaf covered by each node
	int *maxRange, // Input: array containing maximum (sorted) leaf covered by each node
	float4 *CMs, // Input: array containing centers of mass for each leaf
	AABB *bounds, // Input: array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, // Input: array containing corresponding unsorted indices for each leaf
	float *radii, // Input: radii of all nodes - currently the same for all particles
	int numParticles, // Input: number of virtual particles
	SimParams params, // Input: simulation parameters
	float *contactDistance, // Output: distance between particles presenting largest penetration
	int *collidingParticleIndex, // Output: particle of most important contact
	int *collidingRigidBodyIndex) // Output: rigid body of most important contact
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //handling particle #(index)
	if (index >= numParticles)
		return;

	// initialize stack
	int stack[64]; // using stack of indices
	int* stackPtr = stack;
	*stackPtr++ = -1; // push -1 at beginning so that thread will return when stack is empty
	int SoAindex = numParticles; // start at root
	// Traverse nodes starting from the root.
	// load leaf positions once - better to use this array as successive threads will access successive memory positions

	// load sorted variables
	int queryIndex = sortedIndices[index]; //load original particle index once
	float4 queryPos = CMs[index];
	AABB queryBound = bounds[index]; // load particle's bounding volume once
	
	// load unsorted variables
	float queryRad = radii[queryIndex]; // load particle radius once - currently all particles have the same radius
	int queryRigidBody = rigidBodyIndex[queryIndex]; // rigid body index corresponding to query particle	
	float maxDistance = contactDistance[queryIndex];

	int numCollisions = 0; // count number of collisions
	do
	{
		//Check each child node for overlap
		int leftChildIndex = leftIndices[SoAindex]; //load left child index once
		int rightChildIndex = rightIndices[SoAindex]; //load right child index once

		bool overlapL = checkOverlap(queryBound, bounds[leftChildIndex]); //check overlap with left child
		bool overlapR = checkOverlap(queryBound, bounds[rightChildIndex]); //check overlap with right child

		//indices are unique for each node, internal or leaf, as they are stored in the same array
		if (leftChildIndex == index || maxRange[leftChildIndex] < index) //left child is current leaf
			overlapL = false;
		if (rightChildIndex == index || maxRange[rightChildIndex] < index) //right child is current leaf
			overlapR = false;
		bool isLeftLeaf = isLeaf[leftChildIndex]; //load left child's leaf flag once
		bool isRightLeaf = isLeaf[rightChildIndex]; //load right child's leaf flag once

		// broad phase collision detection test
		if (overlapL && isLeftLeaf)
		{
			int leftSortedIndex = sortedIndices[leftChildIndex];
			int leftRigidBody = rigidBodyIndex[leftSortedIndex];
			if (queryRigidBody != leftRigidBody)
			{
				float dist = length(CMs[leftChildIndex] - queryPos); //distance between two centers
				float collisionThreshold = queryRad + radii[leftSortedIndex]; //sum of radii
				if (collisionThreshold - dist > maxDistance)
				{
					maxDistance = collisionThreshold - dist;
					// narrow phase collision detection test
					numCollisions++;
					color[queryIndex] = make_float4(1, 0, 0, 0);
					// report collision
					contactDistance[queryIndex] = maxDistance;
					collidingParticleIndex[queryIndex] = leftSortedIndex;
					collidingRigidBodyIndex[queryIndex] = leftRigidBody;
				}
			}
		}
		// broad phase collision detection test
		if (overlapR && isRightLeaf)
		{

			int rightSortedIndex = sortedIndices[rightChildIndex];
			int rightRigidBody = rigidBodyIndex[rightSortedIndex];
			if (queryRigidBody != rightRigidBody)
			{
				float dist = length(CMs[rightChildIndex] - queryPos); // distance between two centers
				float collisionThreshold = queryRad + radii[rightSortedIndex]; // sum of radii
				if (collisionThreshold - dist > maxDistance)
				{
					maxDistance = collisionThreshold - dist;
					// narrow phase collision detection test
					numCollisions++;
					color[queryIndex] = make_float4(1, 0, 0, 0);
					// report collision
					contactDistance[queryIndex] = maxDistance;
					collidingParticleIndex[queryIndex] = rightSortedIndex;
					collidingRigidBodyIndex[queryIndex] = rightRigidBody;
				}
			}
		}

		// Query overlaps an internal node => traverse
		bool traverseL = (overlapL && !isLeftLeaf);
		bool traverseR = (overlapR && !isRightLeaf);

		if (!traverseL && !traverseR)
			SoAindex = *--stackPtr; // pop
		else
		{
			SoAindex = (traverseL) ? leftChildIndex : rightChildIndex;
			if (traverseL && traverseR)
				*stackPtr++ = rightChildIndex; // push
		}
	} while (SoAindex != -1);

	if (!numCollisions)
		color[queryIndex] = make_float4(1, 1, 1, 0);
}

void FindRigidBodyCollisionsBVHWrapper(
		float4 *color, // Input: particle's color, only used for testing purposes
		int *rigidBodyIndex, // Input: index of the rigid body each particle belongs to
		bool *isLeaf, // Input: array containing a flag to indicate whether node is leaf
		int *leftIndices, // Input:  array containing indices of the left children of each node
		int *rightIndices, // Input: array containing indices of the right children of each node
		int *minRange, // Input: array containing minimum (sorted) leaf covered by each node
		int *maxRange, // Input: array containing maximum (sorted) leaf covered by each node
		float4 *CMs, // Input: array containing centers of mass for each leaf
		AABB *bounds, // Input: array containing bounding volume for each node - currently templated Array of Structures
		int *sortedIndices, // Input: array containing corresponding unsorted indices for each leaf
		float *radii, // Input: radii of all nodes - currently the same for all particles
		int numThreads, // Input: number of threads to use
		int numParticles, // Input: number of virtual particles
		SimParams params, // Input: simulation parameters
		float *contactDistance, // Output: distance between particles presenting largest penetration
		int *collidingParticleIndex, // Output: particle of most important contact
		int *collidingRigidBodyIndex) // Output: rigid body of most important contact
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	FindRigidBodyCollisionsBVHKernel << < gridDim, blockDim >> >(
		color, // Input: particle's color, only used for testing purposes
		rigidBodyIndex, // Input: index of the rigid body each particle belongs to
		isLeaf, // Input: array containing a flag to indicate whether node is leaf
		leftIndices, // Input:  array containing indices of the left children of each node
		rightIndices, // Input: array containing indices of the right children of each node
		minRange, // Input: array containing minimum (sorted) leaf covered by each node
		maxRange, // Input: array containing maximum (sorted) leaf covered by each node
		CMs, // Input: array containing centers of mass for each leaf
		bounds, // Input: array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, // Input: array containing corresponding unsorted indices for each leaf
		radii, // Input: radii of all nodes - currently the same for all particles
		numParticles, // Input: number of virtual particles
		params, // Input: simulation parameters
		contactDistance, // Output: distance between particles presenting largest penetration
		collidingParticleIndex, // Output: particle of most important contact
		collidingRigidBodyIndex); // Output: rigid body of most important contact


}

__global__ void FindAugmentedRealityCollisionsBVHKernel(
	float4 *color, // Input: particle's color, only used for testing purposes
	float4 *position, // Input: virutal particle positions
	bool *isLeaf, // Input: array containing a flag to indicate whether node is leaf
	int *leftIndices, // Input:  array containing indices of the left children of each node
	int *rightIndices, // Input: array containing indices of the right children of each node
	int *minRange, // Input: array containing minimum (sorted) leaf covered by each node
	int *maxRange, // Input: array containing maximum (sorted) leaf covered by each node
	float4 *CMs, // Input: array containing centers of mass for each leaf
	AABB *bounds, // Input: array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, // Input: array containing corresponding unsorted indices for each leaf
	float *radii, // Input: radii of all nodes - currently the same for all particles
	int numParticles, // Input: number of virtual particles
	int numRangeData, // Input: number of augmented reality particles
	SimParams params, // Input: simulation parameters
	float *contactDistance, // Output: distance between particles presenting largest penetration
	int *collidingParticleIndex) // Output: particle of most important contact
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //handling particle #(index)
	if (index >= numParticles)
		return;

	// initialize stack
	int stack[64]; // using stack of indices
	int* stackPtr = stack;
	*stackPtr++ = -1; // push -1 at beginning so that thread will return when stack is empty
	int SoAindex = numRangeData; // start at root
	// Traverse nodes starting from the root.
	// load leaf positions once - better to use this array as successive threads will access successive memory positions

	// load sorted variables
	float4 queryPos = position[index];

	// load unsorted variables
	float queryRad = radii[index]; // load particle radius once - currently all particles have the same radius
	float maxDistance = contactDistance[index];

	do
	{
		//Check each child node for overlap
		int leftChildIndex = leftIndices[SoAindex]; //load left child index once
		int rightChildIndex = rightIndices[SoAindex]; //load right child index once

		bool overlapL = checkOverlap(queryPos, bounds[leftChildIndex], queryRad); //check overlap with left child
		bool overlapR = checkOverlap(queryPos, bounds[rightChildIndex], queryRad); //check overlap with right child

		bool isLeftLeaf = isLeaf[leftChildIndex]; //load left child's leaf flag once
		bool isRightLeaf = isLeaf[rightChildIndex]; //load right child's leaf flag once

		// broad phase collision detection test
		if (overlapL && isLeftLeaf)
		{
			int leftSortedIndex = sortedIndices[leftChildIndex];
			float dist = length(CMs[leftChildIndex] - queryPos); //distance between two centers
			float collisionThreshold = queryRad + radii[leftSortedIndex]; //sum of radii
			if (collisionThreshold - dist > maxDistance)
			{
				maxDistance = collisionThreshold - dist;
				// narrow phase collision detection test
				color[index] = make_float4(0, 1, 0, 0);
				// report collision
				contactDistance[index] = maxDistance;
				collidingParticleIndex[index] = leftSortedIndex;
			}
		}
		// broad phase collision detection test
		if (overlapR && isRightLeaf)
		{

			int rightSortedIndex = sortedIndices[rightChildIndex];

			float dist = length(CMs[rightChildIndex] - queryPos); // distance between two centers
			float collisionThreshold = queryRad + radii[rightSortedIndex]; // sum of radii
			if (collisionThreshold - dist > maxDistance)
			{
				maxDistance = collisionThreshold - dist;
				// narrow phase collision detection test
				color[index] = make_float4(0, 1, 0, 0);
				// report collision
				contactDistance[index] = maxDistance;
				collidingParticleIndex[index] = rightSortedIndex;
			}
		}

		// Query overlaps an internal node => traverse
		bool traverseL = (overlapL && !isLeftLeaf);
		bool traverseR = (overlapR && !isRightLeaf);

		if (!traverseL && !traverseR)
			SoAindex = *--stackPtr; // pop
		else
		{
			SoAindex = (traverseL) ? leftChildIndex : rightChildIndex;
			if (traverseL && traverseR)
				*stackPtr++ = rightChildIndex; // push
		}
	} while (SoAindex != -1);

}

void FindAugmentedRealityCollisionsBVHWrapper(
	float4 *color, // Input: particle's color, only used for testing purposes
	float4 *position, // Input: virutal particle positions
	bool *isLeaf, // Input: array containing a flag to indicate whether node is leaf
	int *leftIndices, // Input:  array containing indices of the left children of each node
	int *rightIndices, // Input: array containing indices of the right children of each node
	int *minRange, // Input: array containing minimum (sorted) leaf covered by each node
	int *maxRange, // Input: array containing maximum (sorted) leaf covered by each node
	float4 *CMs, // Input: array containing centers of mass for each leaf
	AABB *bounds, // Input: array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, // Input: array containing corresponding unsorted indices for each leaf
	float *radii, // Input: radii of all nodes - currently the same for all particles
	int numThreads, // Input: number of threads to use
	int numParticles, // Input: number of virtual particles
	int numRangeData, // Input: number of augmented reality particles
	SimParams params, // Input: simulation parameters
	float *contactDistance, // Output: distance between particles presenting largest penetration
	int *collidingParticleIndex)// Output: particle of most important contact
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	FindAugmentedRealityCollisionsBVHKernel << < gridDim, blockDim >> >(
		color, // Input: particle's color, only used for testing purposes
		position, // Input: virutal particle positions
		isLeaf, // Input: array containing a flag to indicate whether node is leaf
		leftIndices, // Input:  array containing indices of the left children of each node
		rightIndices, // Input: array containing indices of the right children of each node
		minRange, // Input: array containing minimum (sorted) leaf covered by each node
		maxRange, // Input: array containing maximum (sorted) leaf covered by each node
		CMs, // Input: array containing centers of mass for each leaf
		bounds, // Input: array containing bounding volume for each node - currently templated Array of Structures
		sortedIndices, // Input: array containing corresponding unsorted indices for each leaf
		radii, // Input: radii of all nodes - currently the same for all particles
		numParticles, // Input: number of virtual particles
		numRangeData, // Input: number of augmented reality particles
		params, // Input: simulation parameters
		contactDistance, // Output: distance between particles presenting largest penetration
		collidingParticleIndex); // Output: particle of most important contact
}