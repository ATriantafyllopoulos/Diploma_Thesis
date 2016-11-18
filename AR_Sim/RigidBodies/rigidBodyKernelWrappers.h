#include "particleSystem.cuh"
#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include "BVHcreation.h"
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#ifndef RIGIDBODYKERNELWRAPPERS_H
#define RIGIDBODYKERNELWRAPPERS_H
void integrateSystemRigidBodies(float4 *CMs, //rigid body center of mass
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
	float *rbMass, //total mass of rigid body
	float3 minPos, //smallest coordinate of scene's bounding box
	float3 maxPos, //largest coordinate of scene's bounding box
	int numBodies, //number of rigid bodies
	SimParams params, //simulation parameters
	int numThreads); //number of threads;

void computeGlobalAttributesWrapper(float4 *CMs, //rigid body's center of mass
	float4 *rigidVel, //rigid body's velocity
	float4 *relativePos, //particle's relative position
	float4 *globalPos, //particle's global position
	float4 *globalVel, //particle's world velocity
	glm::quat *rbQuaternion, //contains current quaternion for each rigid body
	float4 *rbAngularVelocity, //contains angular velocities for each rigid body
	int *rigidBodyIndex, //index of associated rigid body
	int numParticles, //number of particles
	int numThreads);

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
	SimParams params);

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
	SimParams params); //simulation parameters

void PreloadRigidBodyVariablesWrapper(
		float4 *rbForce, //Input: rigid body forces - one element per rigid body
		float4 *rbTorque, //Input: rigid body torques - one element per rigid body
		float4 *rbPositions, //Input: rigid body positions - one element per rigid body
		float4 *pForce, //Output: rigid body forces - one element per particle
		float4 *pTorque, //Output: rigid body torques - one element per particle
		float4 *pPositions, //Output: rigid body positions - one element per particle
		int *rbIndices, //Auxil.: indices of corresponding rigid bodies - one element per particle
		int numParticles, //Auxil.: number of particles
		int numThreads); //number of threads to use

void ReduceRigidBodyVariables(
		float4 *rbForce, //Output: rigid body forces - one element per rigid body
		float4 *rbTorque, //Output: rigid body torques - one element per rigid body
		float4 *rbPositions, //Output: rigid body positions - one element per rigid body
		float4 *pForce, //Input: rigid body forces - one element per particle
		float4 *pTorque, //Input: rigid body torques - one element per particle
		float4 *pPositions, //Input: rigid body positions - one element per particle
		int *particlesPerObjectThrown, //Auxil.: number of particles for each rigid body - one element per thrown objects
		bool *isRigidBody, //Auxil.: flag indicating whether thrown object is a rigid body
		int objectsThrown, //Auxil.: number of objects thrown - rigid bodies AND point sprites
		int numRigidBodies, //Auxil.: number of rigid bodies
		int numThreads, //number of threads to use
		int numParticles, //total number of virtual particles; //number of threads to use
		bool *toExit);

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
int numThreads);

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
int numThreads);

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
		int numThreads);
void ReduceRigidBodyARVariables(
		float4 *rbForce, //Output: rigid body forces - one element per rigid body
		float4 *rbTorque, //Output: rigid body torques - one element per rigid body
		float4 *rbPositions, //Output: rigid body positions - one element per rigid body
		float4 *pForce, //Input: rigid body forces - one element per particle
		float4 *pTorque, //Input: rigid body torques - one element per particle
		float4 *pPositions, //Input: rigid body positions - one element per particle
		int *pCountARCollions, //Input: AR collisions - one element per particle
		int *particlesPerObjectThrown, //Auxil.: number of particles for each rigid body - one element per thrown objects
		bool *isRigidBody, //Auxil.: flag indicating whether thrown object is a rigid body
		int objectsThrown, //Auxil.: number of objects thrown - rigid bodies AND point sprites
		int numRigidBodies, //Auxil.: number of rigid bodies
		int numThreads, //number of threads to use
		int numParticles, //total number of virtual particles
		bool *toExit);

void WallCollisionWrapper(
	float4 *particlePos, // particle positions
	float4 *rbPos, // rigid body center of mass
	float3 minPos, // scene AABB's smallest coordinates
	float3 maxPos, // scene AABB's largest coordinates
	float4 *rbVel, // rigid body linear velocity
	float4 *rbAng, // rigid body angular velocity
	float4 *rbLinMom, // rigid body linear momentum
	float4 *rbAngMom, // rigid body angular momentum
	glm::mat3 *Iinv, // current rigid body inverse inertia tensor
	float *rbMass, // rigid body mass
	int *rbIndices, // index showing where each particle belongs
	int *particlesPerRB, // number of particles per rigid body
	int numRigidBodies, // total number of scene's rigid bodies
	int numParticles, // number of particles to test
	int numThreads, // number of threads to use
	SimParams params);

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
	int numThreads);

void HandleRigidBodyCollisionWrapper(
	float4 *particlePos, // particle positions
	float4 *rbPos, // rigid body center of mass
	float4 *rbVel, // rigid body linear velocity
	float4 *rbAng, // rigid body angular velocity
	float4 *rbLinMom, // rigid body linear momentum
	float4 *rbAngMom, // rigid body angular momentum
	glm::mat3 *Iinv, // current rigid body inverse inertia tensor
	float *rbMass, // rigid body mass
	int *rbIndices, // index showing where each particle belongs
	int *particlesPerRB, // number of particles per rigid body
	int *collidingRigidBodyIndex, // index of rigid body of contact
	int *collidingParticleIndex, // index of particle of contact
	float *contactDistance, // penetration distance
	int numRigidBodies, // total number of scene's rigid bodies
	int numParticles, // number of particles to test
	int numThreads, // number of threads to use
	SimParams params); // simulation parameters

void FindAugmentedRealityCollisionsUniformGridWrapper(
	int *collidingParticleIndex, // index of particle of contact
	float *contactDistance, // penetration distance
	float4 *color,  // particle color
	float4 *oldPos,  // sorted positions
	float4 *ARPos,  // sorted augmented reality positions
	uint   *gridParticleIndex, // sorted particle indices
	uint   *gridParticleIndexAR, // sorted scene particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	uint	numberOfRangeData,
	SimParams params,
	int numThreads);

void HandleAugmentedRealityCollisionWrapper(
	float4 *particlePos, // particle positions
	float4 *scenePos, // scene particle positions
	float4 *rbPos, // rigid body center of mass
	float4 *rbVel, // rigid body linear velocity
	float4 *rbAng, // rigid body angular velocity
	float4 *rbLinMom, // rigid body linear momentum
	float4 *rbAngMom, // rigid body angular momentum
	glm::mat3 *Iinv, // current rigid body inverse inertia tensor
	float *rbMass, // rigid body mass
	int *rbIndices, // index showing where each particle belongs
	int *particlesPerRB, // number of particles per rigid body
	int *collidingParticleIndex, // index of particle of contact
	float *contactDistance, // penetration distance
	int numRigidBodies, // total number of scene's rigid bodies
	int numParticles, // number of particles to test
	int numThreads, // number of threads to use
	SimParams params); // simulation parameters

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
	int *collidingRigidBodyIndex); // Output: rigid body of most important contact

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
	int *collidingParticleIndex); // Output: particle of most important contact

void DebugComputeGlobalAttributes(float4 *CMs, //rigid body's center of mass
	float4 *rigidVel, //rigid body's velocity
	float4 *relativePos, //particle's relative position
	float4 *globalPos, //particle's global position
	float4 *globalVel, //particle's world velocity
	glm::quat *rbQuaternion, //contains current quaternion for each rigid body
	float4 *rbAngularVelocity, //contains angular velocities for each rigid body
	int *rigidBodyIndex, //index of associated rigid body
	int startPos, //starting position of rigid body to test
	int numParticles, //number of particles of rigid body to test
	int numThreads); //number of threads

void resetQuaternionWrapper(glm::quat *rbQuaternion, //contains current quaternion for each rigid body
	int numRigidBodies, //number of rigid bodies
	int numThreads);

void ResetParticleImpulseWrapper(
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	int numParticles, //number of rigid bodies
	int numThreads);

void ReducePerParticleImpulses(
	int *rbIndices, // index of the rigid body each particle belongs to
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	float4 *rbVel, // rigid body linear velocity 
	float4 *rbAng, // rigid body angular velocity 
	float *rbMass, // rigid body mass
	glm::mat3 *rbInertia, // rigid body inverse inertia matrix
	int *particlesPerObject, // number of particles per object
	int numRigidBodies, // total number of rigid bodies
	int numParticles, // total number of particles
	int numThreads);

void FindAndHandleRigidBodyCollisionsUniformGridWrapper(
	int *rbIndices, // index of the rigid body each particle belongs to
	float4 *pLinearImpulse, // total linear impulse acting on current particle
	float4 *pAngularImpulse, // total angular impulse acting on current particle
	float4 *color, // particle color
	float4 *sortedPos,  // sorted particle positions
	float4 *sortedVel,  // sorted particle velocities
	float4 *relativePos, // unsorted relative positions
	uint   *gridParticleIndex, // sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	SimParams params,
	int numThreads);

#endif
