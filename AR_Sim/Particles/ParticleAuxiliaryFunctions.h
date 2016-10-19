#include "particleSystem.cuh"

void cudaInit(int argc, char **argv);
void allocateArray(void **devPtr, size_t size);
void freeArray(void *devPtr);
void threadSync();
void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
void copyArrayToDevice(void *device, const void *host, int offset, int size);
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void setParameters(SimParams *hostParams);
void integrateSystem(float *pos,
	float *vel,
	float deltaTime,
	float3 minPos,
	float3 maxPos,
	int *rigidBodyIndices,
	uint numParticles);
void calcHash(uint  *gridParticleHash,
	uint  *gridParticleIndex,
	float *pos,
	int    numParticles);
void reorderDataAndFindCellStart(int *rbIndices, //index of the rigid body each particle belongs to
	uint  *cellStart,
	uint  *cellEnd,
	float *sortedPos,
	float *sortedVel,
	uint  *gridParticleHash,
	uint  *gridParticleIndex,
	float *oldPos,
	float *oldVel,
	uint   numParticles,
	uint   numCells);
void collide(float4 *pForce, //total force applied to rigid body
	int *rbIndices, //index of the rigid body each particle belongs to
	float4 *relativePos, //particle's relative position
	float4 *pTorque,  //rigid body angular momentum
	float *color,
	float *newVel,
	float *sortedPos,
	float *sortedVel,
	uint  *gridParticleIndex,
	uint  *cellStart,
	uint  *cellEnd,
	uint   numParticles,
	uint   numCells);
void sortParticles(uint **dGridParticleHash, uint **dGridParticleIndex, uint numParticles);
void staticCalcHash(uint  *gridParticleHash,
	uint  *gridParticleIndex,
	float *pos,
	int    numParticles,
	int imageWidth,
	int imageHeight);
void staticReorderDataAndFindCellStart(uint  *cellStart,
	uint  *cellEnd,
	float3 *sortedPos,
	uint  *gridParticleHash,
	uint  *gridParticleIndex,
	float *oldPos,
	uint   numParticles,
	uint   numCells);
void staticCollide(float4 *dCol,
	float4 *rbForces, //total force applied to rigid body
	int *rbIndices, //index of the rigid body each particle belongs to
	float4 *relativePos, //particle's relative position
	float4 *rbTorque,  //rigid body angular momentum
	float *r_radii, //radii of all scene particles
	float *newVel,
	float *sortedPos,
	float *sortedVel,
	float *staticSortedPos,
	uint  *gridParticleIndex,
	uint  *cellStart,
	uint  *cellEnd,
	uint   numParticles,
	uint   numCells);

void mapRelativePositionIndependentParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads);

void mapActualPositionIndependentParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads);

void mapRelativePositionRigidBodyParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads);

void mapActualPositionRigidBodyParticlesWrapper(
	float4 *positions, //particle positions
	float4 *relativePositions, //relative particle positions
	float4 *rbPositions, //rigid body center of mass
	int *rbIndices, //rigid body indices
	int numParticles,
	int numThreads);

void initializeRadiiWrapper(float *radii,
	float particleRadius,
	int numParticles,
	int numThreads);
