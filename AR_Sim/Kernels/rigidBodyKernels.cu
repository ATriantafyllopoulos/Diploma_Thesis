/*
* Particle-based dynamic simulations for rigid bodies.
* Instead of having autonomous particles, we now associate each of them with a rigid body.
* For each particle we have a relative position to the center of mass of its associated rigid body.
* System integration occures over each rigid body.
*/
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

#define CUB_STDERR
//define _CubLog to avoid encountering error: "undefined reference"
#if !defined(_CubLog)
#if (CUB_PTX_ARCH == 0)
#define _CubLog(format, ...) printf(format,__VA_ARGS__);
#elif (CUB_PTX_ARCH >= 200)
#define _CubLog(format, ...) printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, __VA_ARGS__);
#endif
#endif
#define inf 0x7f800000
//cub headers
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

template<typename valueType, typename keyType>
void Cub_Sort_By_Key_Wrapper(valueType **value, const keyType *const *const key, const int &arraySize);

/*
* Integrates each rigid body. Moves center of mass only.
*/
__global__ void integrateRigidBody(float4 *CMs, //rigid body center of mass
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
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numBodies)
		return;
	float4 locLinearMomentum = rbLinearMomentum[index];
	locLinearMomentum += force[index] * deltaTime;
	locLinearMomentum += make_float4(params.gravity, 0) * deltaTime;

	maxPos.x = maxPos.x + 0.1;
	maxPos.y = maxPos.y + 0.1;
	maxPos.z = maxPos.z + 0.1;

	minPos.x = minPos.x - 0.1;
	minPos.y = minPos.y - 0.1;
	minPos.z = minPos.z - 1;

	float4 locPos = CMs[index];
	float locMass = rbMass[index];
	float sphereRadius = rbRadii[index];
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
		locPos.y = maxPos.y - 2 * sphereRadius;
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
	float4 locVel = locLinearMomentum * locMass;
	rbLinearMomentum[index] = locLinearMomentum;
	//locVel += make_float4(params.gravity, 0) * locMass * deltaTime;
	//locVel *= params.globalDamping;

	locPos += locVel * deltaTime;

	//add a 1cm offset to prevent false collisions
	locPos.w = 0.f;
	locVel.w = 0.f;
	CMs[index] = locPos;
	vel[index] = locVel;
	force[index] = make_float4(0, 0, 0, 0); //reset force to zero
	
	//now consider rotational motion
	glm::mat3 inertia = rbInertia[index]; //each inertia matrix has 9 elements

	glm::quat quaternion = rbQuaternion[index];
	glm::mat3 rot = mat3_cast(quaternion);

	glm::mat3 currentInertia = rot * inertia * transpose(rot);
	float4 angularMomentum = rbAngularMomentum[index];
	float4 torque = rbTorque[index];
	angularMomentum += torque * deltaTime;
	//angularMomentum *= params.globalDamping;
	glm::vec3 currentMomentum = glm::vec3(angularMomentum.x, angularMomentum.y, angularMomentum.z);
	glm::vec3 newVelocity = currentInertia * currentMomentum;
	//	correct angular drift
	glm::vec3 currentTorque(torque.x, torque.y, torque.z);
	glm::vec3 angularAcceleration = currentInertia * glm::cross(currentMomentum, newVelocity);

//	newVelocity -= angularAcceleration * deltaTime;
//	newVelocity = glm::vec3(0.001, 0.004, 0.001);
	glm::quat qdot = glm::quat(0, newVelocity.x, newVelocity.y, newVelocity.z) * quaternion;
	qdot /= 2.f;
	quaternion += qdot * deltaTime;
//	float angular_speed = glm::length(newVelocity);
//	float rotation_angle = angular_speed*deltaTime;
//	glm::vec3 rotationAxis = normalize(newVelocity);
//	glm::quat dq(cos(rotation_angle / 2), sin(rotation_angle / 2) * rotationAxis.x, sin(rotation_angle / 2) * rotationAxis.y, sin(rotation_angle / 2) * rotationAxis.z);
//	quaternion = glm::cross(dq, quaternion);
	quaternion = normalize(quaternion);

	newVelocity -= angularAcceleration * deltaTime;

	rbAngularAcceleration[index] = angularAcceleration;
	rbCurrentInertia[index] = currentInertia;
	rbAngularMomentum[index] = angularMomentum;
	rbQuaternion[index] = quaternion;
	rbAngularVelocity[index] = make_float4(newVelocity.x, newVelocity.y, newVelocity.z, 0);
	rbTorque[index] = make_float4(0, 0, 0, 0); //reset torque to zero
}

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
	int numThreads) //number of threads
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numBodies + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
		gridDim.x = 1;
	integrateRigidBody << < gridDim, blockDim >> >(CMs, //rigid body center of mass
		vel, //velocity of rigid body
		force, //force applied to rigid body due to previous collisions
		rbAngularVelocity, //contains angular velocities for each rigid body
		rbQuaternion, //contains current quaternion for each rigid body
		rbTorque, //torque applied to rigid body due to previous collisions
		rbAngularMomentum, //cumulative angular momentum of the rigid body
		rbLinearMomentum, //cumulative linear momentum of the rigid body
		rbInertia, //original moment of inertia for each rigid body - 9 values per RB
		rbCurrentInertia, //current moment of inertia for each rigid body - 9 values per RB
		rbAngularAcceleration, //current angular acceleration due to misaligned angular momentum and velocity
		deltaTime, //dt
		rbRadii, //radius chosen for each rigid body sphere
		rbMass, //total mass of rigid body
		minPos, //smallest coordinate of scene's bounding box
		maxPos, //largest coordinate of scene's bounding box
		numBodies, //number of rigid bodies
		params); //simulation parameters
}

/*
* Function to calculate the global position of each particle given its relative position to its
* associated rigid body's center of mass.
*/
__global__ void computeGlobalAttributes(float4 *CMs, //rigid body's center of mass
	float4 *rigidVel, //rigid body's velocity
	float4 *relativePos, //particle's relative position
	float4 *globalPos, //particle's global position
	float4 *globalVel, //particle's world velocity
	glm::quat *rbQuaternion, //contains current quaternion for each rigid body
	float4 *rbAngularVelocity, //contains angular velocities for each rigid body
	int *rigidBodyIndex, //index of associated rigid body
	int numParticles) //number of particles
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	int correspondingRigidBody = rigidBodyIndex[index];
	if (correspondingRigidBody == -1) return; //if this is an independent virtual particle
	//float4 tempQuat = rbQuaternion[correspondingRigidBody];
	//glm::quat quaternion(tempQuat.w, tempQuat.x, tempQuat.y, tempQuat.z);
	glm::quat quaternion = rbQuaternion[correspondingRigidBody];
	float4 tempPos = relativePos[index];
	glm::vec4 pos = glm::vec4(tempPos.x, tempPos.y, tempPos.z, tempPos.w);
	glm::mat4 rot = mat4_cast(quaternion);
	pos = rot * pos;
	//pos = quaternion * pos * conjugate(quaternion);
	tempPos = make_float4(pos.x, pos.y, pos.z, pos.w);
	relativePos[index] = tempPos;
	globalPos[index] = tempPos + CMs[correspondingRigidBody];
	//particle's velocity is the same as its associated rigid body's
	//for the moment we ignore angular velocity
	globalVel[index] = rigidVel[correspondingRigidBody] + make_float4(cross(make_float3(rbAngularVelocity[correspondingRigidBody]), make_float3(tempPos)), 0);
}

void computeGlobalAttributesWrapper(float4 *CMs, //rigid body's center of mass
float4 *rigidVel, //rigid body's velocity
float4 *relativePos, //particle's relative position
float4 *globalPos, //particle's global position
float4 *globalVel, //particle's world velocity
glm::quat *rbQuaternion, //contains current quaternion for each rigid body
float4 *rbAngularVelocity, //contains angular velocities for each rigid body
int *rigidBodyIndex, //index of associated rigid body
int numParticles, //number of particles
int numThreads) //number of threads
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	computeGlobalAttributes << < gridDim, blockDim >> >(CMs, //rigid body's center of mass
		rigidVel, //rigid body's velocity
		relativePos, //particle's relative position
		globalPos, //particle's global position
		globalVel, //particle's world velocity
		rbQuaternion, //contains current quaternion for each rigid body
		rbAngularVelocity, //contains angular velocities for each rigid body
		rigidBodyIndex, //index of associated rigid body
		numParticles); //number of particles
}

template<typename valueType, typename keyType>
void Cub_Sort_By_Key_Wrapper(valueType **value, const keyType *const *const key, const int &arraySize)
{
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// create unsorted auxiliaries
	valueType *d_value;
	keyType *d_key;

	checkCudaErrors(cudaMalloc((void**)&d_key, arraySize * sizeof(keyType)));
	checkCudaErrors(cudaMemcpy(d_key, *key, arraySize * sizeof(keyType), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMalloc((void**)&d_value, arraySize * sizeof(valueType)));
	checkCudaErrors(cudaMemcpy(d_value, *value, arraySize * sizeof(valueType), cudaMemcpyDeviceToDevice));

	// create sorted auxiliaries
	valueType *d_value_sorted;
	keyType *d_key_sorted;

	checkCudaErrors(cudaMalloc((void**)&d_key_sorted, arraySize * sizeof(keyType)));
	checkCudaErrors(cudaMalloc((void**)&d_value_sorted, arraySize * sizeof(valueType)));


	cub::CachingDeviceAllocator  g_allocator(true);
	cub::DoubleBuffer<keyType> keyBuffer;
	cub::DoubleBuffer<valueType> valueBuffer;
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;

	// Allocate temporary storage
	keyBuffer.d_buffers[0] = d_key;
	keyBuffer.d_buffers[1] = d_key_sorted;

	valueBuffer.d_buffers[0] = d_value;
	valueBuffer.d_buffers[1] = d_value_sorted;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keyBuffer, valueBuffer, arraySize));
	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keyBuffer, valueBuffer, arraySize));

	d_value_sorted = valueBuffer.Current();
	d_value = valueBuffer.Alternate();

	d_key_sorted = keyBuffer.Current();
	d_key = keyBuffer.Alternate();

	checkCudaErrors(cudaMemcpy(*value, d_value_sorted, arraySize * sizeof(valueType), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (d_key)
		checkCudaErrors(cudaFree(d_key));
	if (d_key_sorted)
		checkCudaErrors(cudaFree(d_key_sorted));
	if (d_value)
		checkCudaErrors(cudaFree(d_value));
	if (d_value_sorted)
		checkCudaErrors(cudaFree(d_value_sorted));
	if (d_temp_storage)
		checkCudaErrors(cudaFree(d_temp_storage));
}

// explicitly instansiate all function derivations to be used
template void Cub_Sort_By_Key_Wrapper<int, float>(int **value, const float *const *const key, const int &arraySize);
template void Cub_Sort_By_Key_Wrapper<int, int>(int **value, const int *const *const key, const int &arraySize);

void Cub_Double_Sort(float *float_key, int *int_key, int **value, const int& arraySize)
{

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// create unsorted auxiliaries
	float *d_float_key;
	int *d_int_key, *d_value;
	checkCudaErrors(cudaMalloc((void**)&d_float_key, arraySize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_float_key, float_key, arraySize * sizeof(float), cudaMemcpyDeviceToDevice));
	
	checkCudaErrors(cudaMalloc((void**)&d_int_key, arraySize * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_int_key, int_key, arraySize * sizeof(int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMalloc((void**)&d_value, arraySize * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_value, *value, arraySize * sizeof(int), cudaMemcpyDeviceToDevice));

	// create sorted auxiliaries
	float *d_float_key_sorted;
	int *d_int_key_sorted, *d_value_sorted;
	checkCudaErrors(cudaMalloc((void**)&d_float_key_sorted, arraySize * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_int_key_sorted, arraySize * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&d_value_sorted, arraySize * sizeof(int)));


	cub::CachingDeviceAllocator  g_allocator(true);
	cub::DoubleBuffer<float> sortIntValuesByFloatKey_key;
	cub::DoubleBuffer<int> sortIntValuesByFloatKey_value;
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;

	// Allocate temporary storage
	sortIntValuesByFloatKey_key.d_buffers[0] = d_float_key;
	sortIntValuesByFloatKey_key.d_buffers[1] = d_float_key_sorted;

	sortIntValuesByFloatKey_value.d_buffers[0] = d_value;
	sortIntValuesByFloatKey_value.d_buffers[1] = d_value_sorted;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntValuesByFloatKey_key,
		sortIntValuesByFloatKey_value, arraySize));
	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntValuesByFloatKey_key,
		sortIntValuesByFloatKey_value, arraySize));

	d_value_sorted = sortIntValuesByFloatKey_value.Current();
	d_value = sortIntValuesByFloatKey_value.Alternate();

	d_float_key_sorted = sortIntValuesByFloatKey_key.Current();
	d_float_key = sortIntValuesByFloatKey_key.Alternate();

	if (d_temp_storage)
		cudaFree(d_temp_storage);
	temp_storage_bytes = 0;
	d_temp_storage = NULL;

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	// Allocate temporary storage
	cub::DoubleBuffer<float> sortIntKeysByFloatKey_key;
	cub::DoubleBuffer<int> sortIntKeysByFloatKey_value;
	sortIntKeysByFloatKey_key.d_buffers[0] = d_float_key;
	sortIntKeysByFloatKey_key.d_buffers[1] = d_float_key_sorted;

	sortIntKeysByFloatKey_value.d_buffers[0] = d_int_key;
	sortIntKeysByFloatKey_value.d_buffers[1] = d_int_key_sorted;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntKeysByFloatKey_key,
		sortIntKeysByFloatKey_value, arraySize));
	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntKeysByFloatKey_key,
		sortIntKeysByFloatKey_value, arraySize));

	d_int_key_sorted = sortIntKeysByFloatKey_value.Current();
	d_int_key = sortIntKeysByFloatKey_value.Alternate();

	d_float_key_sorted = sortIntKeysByFloatKey_key.Current();
	d_float_key = sortIntKeysByFloatKey_key.Alternate();

	if (d_temp_storage)
		cudaFree(d_temp_storage);
	temp_storage_bytes = 0;
	d_temp_storage = NULL;

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	cub::DoubleBuffer<int> sortIntValuesByIntKey_key;
	cub::DoubleBuffer<int> sortIntValuesByIntKey_value;
	sortIntValuesByIntKey_key.d_buffers[0] = d_int_key_sorted;
	sortIntValuesByIntKey_key.d_buffers[1] = d_int_key;

	sortIntValuesByIntKey_value.d_buffers[0] = d_value_sorted;
	sortIntValuesByIntKey_value.d_buffers[1] = d_value;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntValuesByIntKey_key,
		sortIntValuesByIntKey_value, arraySize));
	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntValuesByIntKey_key,
		sortIntValuesByIntKey_value, arraySize));

	d_value = sortIntValuesByIntKey_value.Current();
	d_value_sorted = sortIntValuesByIntKey_value.Alternate();

	d_int_key = sortIntValuesByIntKey_key.Current();
	d_int_key_sorted = sortIntValuesByIntKey_key.Alternate();

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(*value, sortIntValuesByIntKey_value.Current(), arraySize * sizeof(int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	if (d_float_key)
		checkCudaErrors(cudaFree(d_float_key));
	if (d_int_key)
		checkCudaErrors(cudaFree(d_int_key));
	if (d_float_key_sorted)
		checkCudaErrors(cudaFree(d_float_key_sorted));
	if (d_int_key_sorted)
		checkCudaErrors(cudaFree(d_int_key_sorted));
	if (d_value_sorted)
		checkCudaErrors(cudaFree(d_value_sorted));
	if (d_temp_storage)
		checkCudaErrors(cudaFree(d_temp_storage));
}

/*
 * Kernel function used to pre-load necessary rigid body variables used by different particles during
 * collision detection (see wrapper function for more details).
 */
__global__ void PreloadRigidBodyVariablesKernel(
		float4 *rbForce, //Input: rigid body forces - one element per rigid body
		float4 *rbTorque, //Input: rigid body torques - one element per rigid body
		float4 *rbPositions, //Input: rigid body center of mass - one element per rigid body
		float4 *pForce, //Output: rigid body forces - one element per particle
		float4 *pTorque, //Output: rigid body torques - one element per particle
		float4 *pPositions, //Output: rigid body center of mass - one element per particle
		int *rbIndices, //Auxil.: indices of corresponding rigid bodies - one element per particle
		int numParticles) //Auxil.: number of particles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	int rigidBodyIndex = rbIndices[index];
	if (rigidBodyIndex == - 1) //if this is an independent particle
		return; //no values are stored - check explicitly @ collision detection

	pForce[index] = rbForce[rigidBodyIndex]; //load rigid body force
	pTorque[index] = rbTorque[rigidBodyIndex]; //load rigid body torque
	pPositions[index] = rbPositions[rigidBodyIndex]; //load rigid body positions

}

/*
 * Since some rigid body variables are both accessed and, most importantly, updated by all of their respective
 * particles, it is necessary to synchronize these activities. Since it is impossible to specifically match ea-
 * ch block to a particular rigid body, so as to utilize shared memory, we must explicitly pre-load these vari-
 * ables. This is the purpose of this function. Its arguments are /float4 rbForce, /float4 rbTorque and /float4
 * rbPositions. This list must be updated if more variables are added. It has an equal number of outputs, but
 * each output array has a size equal to the total number of particles. Specific care must be given to the bogus
 * rigid body values corresponding to independent virtual particles. They should be explicitly checked during col-
 * lision detectin. After processing collisions, the results must be gathered (i.e. reduced) to these variables again.
 * Possibly DEPRECATED.
 */
void PreloadRigidBodyVariablesWrapper(
		float4 *rbForce, //Input: rigid body forces - one element per rigid body
		float4 *rbTorque, //Input: rigid body torques - one element per rigid body
		float4 *rbPositions, //Input: rigid body positions - one element per rigid body
		float4 *pForce, //Output: rigid body forces - one element per particle
		float4 *pTorque, //Output: rigid body torques - one element per particle
		float4 *pPositions, //Output: rigid body positions - one element per particle
		int *rbIndices, //Auxil.: indices of corresponding rigid bodies - one element per particle
		int numParticles, //Auxil.: number of particles
		int numThreads) //number of threads to use
{
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);
	PreloadRigidBodyVariablesKernel << < gridDim, blockDim >> >(
			rbForce, //Input: rigid body forces - one element per rigid body
			rbTorque, //Input: rigid body torques - one element per rigid body
			rbPositions, //Input: rigid body positions - one element per rigid body
			pForce, //Output: rigid body forces - one element per particle
			pTorque, //Output: rigid body torques - one element per particle
			pPositions, //Output: rigid body positions - one element per particle
			rbIndices, //Auxil.: indices of corresponding rigid bodies - one element per particle
			numParticles); //Auxil.: number of particles)
}


struct CustomAdd
{
	template <typename T>
	__device__ __forceinline__
	T operator()(const T &a, const T &b) { return a + b; }
};

/*
 * Kernel used to reset particle force, torque and correct position values for current iteration to zero.
 */
__global__ void resetBlockVariables(
		float4 *pForce, //Input: rigid body forces - one element per particle
		float4 *pTorque, //Input: rigid body torques - one element per particle
		float4 *pPositions, //Input: rigid body positions - one element per particle
		int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	pForce[index] = make_float4(0.f);
	pTorque[index] = make_float4(0.f);
	pPositions[index] = make_float4(0.f);
}

/*
 * Kernel used to add reduced results to rigid body variables.
 */
__global__ void combineIntermediateResults(
		float4 *rbForce, //Output: rigid body forces - one element per rigid body
		float4 *rbTorque, //Output: rigid body torques - one element per rigid body
		float4 *rbPositions, //Output: rigid body positions - one element per rigid body
		float4 *intermediateForce, //Input: reduced rigid body forces - one element per rigid body
		float4 *intermediateTorque, //Input: reduced rigid body torques - one element per rigid body
		float4 *intermediatePositions, //Input: reduced rigid body positions - one element per rigid body
		int numRigidBodies) //Auxil.: number of rigid bodies
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numRigidBodies)
		return;

	//simply add corresponding intermediate results
	rbForce[index] += intermediateForce[index];
	rbTorque[index] += intermediateTorque[index];
	//rbPositions[index] += intermediatePositions[index];
}

struct CustomMax4RB
{
	template <typename T>
	__device__ __forceinline__
	T operator()(const T &a, const T &b) const {
		T res;
		res.x =  (b.x > a.x) ? b.x : a.x;
		res.y =  (b.y > a.y) ? b.y : a.y;
		res.z =  (b.z > a.z) ? b.z : a.z;
		res.w =  (b.w > a.w) ? b.w : a.w;
		return res;
	}
};

/*
 * After computing collision detection results for each particles it is necessary to combine (i.e. reduce)
 * all the intermediate results for each rigid body. This is done using cub for reduction. To do this it is
 * necessary to know how many particles belong to each rigid body, and the start and end of each rigid body's
 * particle sequence. To this end, we introduce two new variables, /int* particlesPerObjectThrown and /int objectsThrown
 * to keep count of how many objects are thrown, and how many particles belong to each of them. Note: objectsThrown is
 * different than numRigidBodies because it also takes into account point sprites.
 */
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
		int numParticles, //total number of virtual particles
		bool *toExit)
{
	//auxiliary variables used to store intermediate reduce results
	float4 *intermediateForce;
	float4 *intermediateTorque;
	float4 *intermediatePositions;

	cudaMalloc((void**)&intermediateForce, sizeof(float4) * numRigidBodies);
	cudaMalloc((void**)&intermediateTorque, sizeof(float4) * numRigidBodies);
	cudaMalloc((void**)&intermediatePositions, sizeof(float4) * numRigidBodies);
//	std::cout << "Started new iteration..." << std::endl;
//	std::cout << "Number of objects thrown: " << objectsThrown << std::endl;
//	std::cout << "Number of rigid bodies: " << numRigidBodies << std::endl;
//	std::cout << "Number of particles: " << numParticles << std::endl;
	int rbCounter = 0;
	int sumParticles = 0;
	//bool toExit = false;
	for (int num = 0; num < objectsThrown; num++)
	{
		int currentNumParticles = particlesPerObjectThrown[num];
		if(isRigidBody[num]) //if this is NOT a point sprite - point sprites are not rigid bodies
		{
//			std::cout << "Current rigid body: " << rbCounter + 1 << std::endl;
//			std::cout << "Particles of current rigid body: " << currentNumParticles << std::endl;
//			std::cout << "Total particles processed so far: " << sumParticles << std::endl;
//			float4 *cpuForce = new float4[currentNumParticles];
//			checkCudaErrors(cudaMemcpy(cpuForce, &pForce[sumParticles], sizeof(float4) * currentNumParticles, cudaMemcpyDeviceToHost));
//			for (int i = 0; i < currentNumParticles; i++)
//				if (cpuForce[i].x != 0 || cpuForce[i].y != 0 ||cpuForce[i].z != 0)
//					std::cout << "Particle force @" << i + sumParticles << ": (" << cpuForce[i].x << " " << cpuForce[i].y << " " << cpuForce[i].z << ")" << std::endl;
//			delete cpuForce;
//			CustomAdd addOp;
//			float4 init = make_float4(0, 0, 0, 0);
			float4 *d_out;
			//rigid body force reduction
			checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float4) * currentNumParticles));
			void     *d_temp_storage = NULL;
			size_t   temp_storage_bytes = 0;
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pForce[sumParticles], d_out, currentNumParticles));
			checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pForce[sumParticles], d_out, currentNumParticles));

			//copy result to intermediate variable
			checkCudaErrors(cudaMemcpy(&intermediateForce[rbCounter], &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));
			float4 cpuTest;
			checkCudaErrors(cudaMemcpy(&cpuTest, &intermediateForce[rbCounter], sizeof(float4), cudaMemcpyDeviceToHost));
			if (cpuTest.x != 0 || cpuTest.y != 0 ||cpuTest.z != 0)
			{
//				std::cerr << "Reduced force: (" << cpuTest.x << " " << cpuTest.y << " " << cpuTest.z << ")" << std::endl;
				if (cpuTest.x != cpuTest.x || cpuTest.y != cpuTest.y || cpuTest.z != cpuTest.z)
					*toExit = true;
			}
			cudaMemset(d_out, 0, particlesPerObjectThrown[num]);

			//rigid body torque reduction
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pTorque[sumParticles], d_out, currentNumParticles));
			checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pTorque[sumParticles], d_out, currentNumParticles));

			//copy result to intermediate variable
			checkCudaErrors(cudaMemcpy(&intermediateTorque[rbCounter], &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(&cpuTest, &intermediateTorque[rbCounter], sizeof(float4), cudaMemcpyDeviceToHost));
			if (cpuTest.x != 0 || cpuTest.y != 0 ||cpuTest.z != 0)
			{
//				std::cerr << "Applied torque: (" << cpuTest.x << " " << cpuTest.y << " " << cpuTest.z << ")" << std::endl;
				if (cpuTest.x != cpuTest.x || cpuTest.y != cpuTest.y || cpuTest.z != cpuTest.z)
					*toExit = true;
			}
			cudaMemset(d_out, 0, particlesPerObjectThrown[num]);

			CustomMax4RB    max_op;
			float4 init = make_float4(-inf, -inf, -inf, -inf);

			//rigid body positions reduction
			checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage,
					temp_storage_bytes,
					&pPositions[sumParticles],
					d_out,
					currentNumParticles,
					max_op,
					init));
			checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage,
					temp_storage_bytes,
					&pPositions[sumParticles],
					d_out,
					currentNumParticles,
					max_op,
					init));


			//copy result to intermediate variable
			checkCudaErrors(cudaMemcpy(&intermediatePositions[rbCounter], &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));
//			checkCudaErrors(cudaMemcpy(&cpuTest, &intermediatePositions[rbCounter], sizeof(float4), cudaMemcpyDeviceToHost));
//			std::cout << "Applied correction: (" << cpuTest.x << " " << cpuTest.y << " " << cpuTest.z << ")" << std::endl;
			checkCudaErrors(cudaFree(d_out));
			checkCudaErrors(cudaFree(d_temp_storage));
			rbCounter++; //increase counter to keep track of rigid bodies processed


		}
		sumParticles += currentNumParticles;
	}

	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numRigidBodies + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
			gridDim.x = 1;
	if(numRigidBodies)
	{

		combineIntermediateResults<< < gridDim, blockDim >> >(
				rbForce, //Output: rigid body forces - one element per rigid body
				rbTorque, //Output: rigid body torques - one element per rigid body
				rbPositions, //Output: rigid body positions - one element per rigid body
				intermediateForce, //Input: reduced rigid body forces - one element per rigid body
				intermediateTorque, //Input: reduced rigid body torques - one element per rigid body
				intermediatePositions, //Input: reduced rigid body positions - one element per rigid body
				numRigidBodies); //Auxil.: number of rigid bodies
//		float *totalTorque = new float[4 * numRigidBodies];
//		float *currentTorque = new float[4 * numRigidBodies];
//		cudaMemcpy(totalTorque, rbTorque, sizeof(float) * 4 *numRigidBodies, cudaMemcpyDeviceToHost);
//		cudaMemcpy(currentTorque, intermediateTorque, sizeof(float) * 4 *numRigidBodies, cudaMemcpyDeviceToHost);
//		for (int i = 0; i < numRigidBodies; i++)
//		{
//			bool closeAll = false;
//			if(totalTorque[4 * i] != totalTorque[4 * i] ||
//					totalTorque[4 * i + 1] != totalTorque[4 * i + 1] ||
//					totalTorque[4 * i + 2] != totalTorque[4 * i + 2])
//			{
//				std::cerr << "Total torque is wrong." << std::endl;
//				closeAll = true;
//			}
//			if(currentTorque[4 * i] != currentTorque[4 * i] ||
//					currentTorque[4 * i + 1] != currentTorque[4 * i + 1] ||
//					currentTorque[4 * i + 2] != currentTorque[4 * i + 2])
//			{
//				std::cerr << "Current torque is wrong." << std::endl;
//				closeAll = true;
//			}
//
//			if (closeAll)
//			{
//				exit(1);
//			}
//		}
//		delete totalTorque;
//		delete currentTorque;
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	blockDim = dim3(numThreads, 1);
	gridDim = dim3((numParticles + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
		gridDim.x = 1;
	if(numParticles)
	{
		resetBlockVariables<< < gridDim, blockDim >> >(
				pForce, //Input: rigid body forces - one element per particle
				pTorque, //Input: rigid body torques - one element per particle
				pPositions, //Input: rigid body positions - one element per particle
				numParticles);
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	cudaFree(intermediateForce);
	cudaFree(intermediateTorque);
	cudaFree(intermediatePositions);
}

void testCubReduce(int elements)
{
	float4 *cpuData = new float4[elements];
	for (int i = 0; i < elements; i++)
	{
		cpuData[i].x = 1.0;
		cpuData[i].y = 2.0;
		cpuData[i].z = 3.0;
		cpuData[i].w = -1.0;
	}
	float4 *gpuData;
	cudaMalloc((void**)&gpuData, sizeof(float4) * elements);
	cudaMemcpy(gpuData, cpuData, sizeof(float4) * elements, cudaMemcpyHostToDevice);

	float4 *d_out;
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float4) * elements));
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpuData, d_out, elements));
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpuData, d_out, elements));

	float4 gpuResult;
	//copy result CPU
	checkCudaErrors(cudaMemcpy(&gpuResult, &d_out[0], sizeof(float4), cudaMemcpyDeviceToHost));
	std::cout << "Result of cub sum reduction is: (" << gpuResult.x << ", " << gpuResult.y << ", " << gpuResult.z <<
			", " << gpuResult.w << ")" << std::endl;

	cudaFree(d_out);
	cudaFree(d_temp_storage);
	cudaFree(gpuData);
	delete cpuData;
}

/*
 * Kernel used to reset particle force, torque and correct position values for current iteration to zero.
 */
__global__ void resetBlockVariablesAR(
		float4 *pForce, //Input: rigid body forces - one element per particle
		float4 *pTorque, //Input: rigid body torques - one element per particle
		float4 *pPositions, //Input: rigid body positions - one element per particle
		int *pCountARCollions, //Input: AR collisions - one element per particle
		int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	pForce[index] = make_float4(0.f);
	pTorque[index] = make_float4(0.f);
	pPositions[index] = make_float4(0.f);
	pCountARCollions[index] = 0;
}

/*
 * Kernel used to add reduced results to rigid body variables.
 */
__global__ void combineIntermediateResultsAR(
		float4 *rbForce, //Output: rigid body forces - one element per rigid body
		float4 *rbTorque, //Output: rigid body torques - one element per rigid body
		float4 *rbPositions, //Output: rigid body positions - one element per rigid body
		float4 *intermediateForce, //Input: reduced rigid body forces - one element per rigid body
		float4 *intermediateTorque, //Input: reduced rigid body torques - one element per rigid body
		float4 *intermediatePositions, //Input: reduced rigid body positions - one element per rigid body
		int *ARCollisionsRigidBody, //Input: reduced number of collisions
		int numRigidBodies) //Auxil.: number of rigid bodies
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numRigidBodies)
		return;
	float numCollisions = (float)ARCollisionsRigidBody[index];
	if (numCollisions < 1) numCollisions = 1;
	//simply add corresponding intermediate results
	rbForce[index] += intermediateForce[index] / numCollisions;
	rbTorque[index] += intermediateTorque[index] / numCollisions;// * 4.f;
	//rbPositions[index] += intermediatePositions[index];
}
/*
 * This function is different to the original in that it now averages computed force/torque over
 * the number of collisions per rigid body to normalize the total force/torque.
 * TODO: integrate the common parts of these two function in one auxiliary function to avoid
 * unnecessary debugging.
 */
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
		bool *toExit)
{
	//auxiliary variables used to store intermediate reduce results
	float4 *intermediateForce;
	float4 *intermediateTorque;
	float4 *intermediatePositions;

	cudaMalloc((void**)&intermediateForce, sizeof(float4) * numRigidBodies);
	cudaMalloc((void**)&intermediateTorque, sizeof(float4) * numRigidBodies);
	cudaMalloc((void**)&intermediatePositions, sizeof(float4) * numRigidBodies);

	int *ARCollisionsRigidBody;
	cudaMalloc((void**)&ARCollisionsRigidBody, sizeof(int) * numRigidBodies);

	int rbCounter = 0;
	int sumParticles = 0;
	for (int num = 0; num < objectsThrown; num++)
	{
		int currentNumParticles = particlesPerObjectThrown[num];
		if(isRigidBody[num]) //if this is NOT a point sprite - point sprites are not rigid bodies
		{
			float4 *d_out;
			checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float4) * currentNumParticles));
			void     *d_temp_storage = NULL;
			size_t   temp_storage_bytes = 0;
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pForce[sumParticles], d_out, currentNumParticles));
			checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pForce[sumParticles], d_out, currentNumParticles));

			//copy result to intermediate variable
			checkCudaErrors(cudaMemcpy(&intermediateForce[rbCounter], &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));
			float4 cpuTest;
			checkCudaErrors(cudaMemcpy(&cpuTest, &intermediateForce[rbCounter], sizeof(float4), cudaMemcpyDeviceToHost));
			if (cpuTest.x != 0 || cpuTest.y != 0 ||cpuTest.z != 0)
			{
//				std::cerr << "Reduced force: (" << cpuTest.x << " " << cpuTest.y << " " << cpuTest.z << ")" << std::endl;
				if (cpuTest.x != cpuTest.x || cpuTest.y != cpuTest.y || cpuTest.z != cpuTest.z)
					*toExit = true;
			}
			cudaMemset(d_out, 0, particlesPerObjectThrown[num]);

			//rigid body torque reduction
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pTorque[sumParticles], d_out, currentNumParticles));
			checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pTorque[sumParticles], d_out, currentNumParticles));

			//copy result to intermediate variable
			checkCudaErrors(cudaMemcpy(&intermediateTorque[rbCounter], &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(&cpuTest, &intermediateTorque[rbCounter], sizeof(float4), cudaMemcpyDeviceToHost));
			if (cpuTest.x != 0 || cpuTest.y != 0 ||cpuTest.z != 0)
			{
//				std::cerr << "Applied torque: (" << cpuTest.x << " " << cpuTest.y << " " << cpuTest.z << ")" << std::endl;
				if (cpuTest.x != cpuTest.x || cpuTest.y != cpuTest.y || cpuTest.z != cpuTest.z)
					*toExit = true;
			}
			cudaMemset(d_out, 0, particlesPerObjectThrown[num]);

			CustomMax4RB    max_op;
			float4 init = make_float4(-inf, -inf, -inf, -inf);

			//rigid body positions reduction
			checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage,
					temp_storage_bytes,
					&pPositions[sumParticles],
					d_out,
					currentNumParticles,
					max_op,
					init));
			checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			checkCudaErrors(cub::DeviceReduce::Reduce(d_temp_storage,
					temp_storage_bytes,
					&pPositions[sumParticles],
					d_out,
					currentNumParticles,
					max_op,
					init));

			//copy result to intermediate variable
			checkCudaErrors(cudaMemcpy(&intermediatePositions[rbCounter], &d_out[0], sizeof(float4), cudaMemcpyDeviceToDevice));


			int *collisionCounter;
			checkCudaErrors(cudaMalloc((void**)&collisionCounter, sizeof(int) * currentNumParticles));
			//rigid body torque reduction
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pCountARCollions[sumParticles], collisionCounter, currentNumParticles));
			checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			checkCudaErrors(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &pCountARCollions[sumParticles], collisionCounter, currentNumParticles));

			//copy result to intermediate variable
			checkCudaErrors(cudaMemcpy(&ARCollisionsRigidBody[rbCounter], &collisionCounter[0], sizeof(int), cudaMemcpyDeviceToDevice));


			checkCudaErrors(cudaFree(collisionCounter));
			checkCudaErrors(cudaFree(d_out));
			checkCudaErrors(cudaFree(d_temp_storage));
			rbCounter++; //increase counter to keep track of rigid bodies processed


		}
		sumParticles += currentNumParticles;
	}

	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numRigidBodies + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
			gridDim.x = 1;
	if(numRigidBodies)
	{

		combineIntermediateResultsAR<< < gridDim, blockDim >> >(
				rbForce, //Output: rigid body forces - one element per rigid body
				rbTorque, //Output: rigid body torques - one element per rigid body
				rbPositions, //Output: rigid body positions - one element per rigid body
				intermediateForce, //Input: reduced rigid body forces - one element per rigid body
				intermediateTorque, //Input: reduced rigid body torques - one element per rigid body
				intermediatePositions, //Input: reduced rigid body positions - one element per rigid body
				ARCollisionsRigidBody, //Input: reduced number of collisions
				numRigidBodies); //Auxil.: number of rigid bodies
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	blockDim = dim3(numThreads, 1);
	gridDim = dim3((numParticles + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
		gridDim.x = 1;
	if(numParticles)
	{
		resetBlockVariablesAR<< < gridDim, blockDim >> >(
				pForce, //Input: rigid body forces - one element per particle
				pTorque, //Input: rigid body torques - one element per particle
				pPositions, //Input: rigid body positions - one element per particle
				pCountARCollions, //Input: AR collisions - one element per particle
				numParticles);
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	cudaFree(ARCollisionsRigidBody);
	cudaFree(intermediateForce);
	cudaFree(intermediateTorque);
	cudaFree(intermediatePositions);
}



// headers for the functions we will use during collision handling
__global__ void CreateIndexKernel(int *indices, int elements);
__device__ void FindExactContactPointKernelAuxil(
	float4 point1, float4 point2,
	float radius1, float radius2,
	float4 *cp, float4 *cn) // exact contact point and contact normal
{
	float t = radius1 / (radius1 + radius2);
	*cp = point1 + (point2 - point1) * t;
	*cn = normalize(point2 - point1) * (-1.f);
}

/*
* Compute impulse magnitude between two colliding rigid bodies
*/
__device__ float ComputeImpulseMagnitudeKernel(
	float4 velA, float4 velB, float4 angA, float4 angB, 
	float4 dispA, float4 dispB,	glm::mat3 IinvA,
	glm::mat3 IinvB, float mA, float mB, float4 n)
{
	glm::vec3 vA(velA.x, velA.y, velA.z);
	glm::vec3 vB(velB.x, velB.y, velB.z);

	glm::vec3 wA(angA.x, angA.y, angA.z);
	glm::vec3 wB(angB.x, angB.y, angB.z);

	glm::vec3 rA(dispA.x, dispA.y, dispA.z);
	glm::vec3 rB(dispB.x, dispB.y, dispB.z);

	glm::vec3 norm(n.x, n.y, n.z);

	glm::vec3 velocityA = vA + glm::cross(wA, rA);
	glm::vec3 velocityB = vB + glm::cross(wB, rB);

	float epsilon = 1;
	float numerator = -(1 + epsilon) * (glm::dot(velocityA, norm) - glm::dot(velocityB, norm));
	float a = 1.f / mA;
	float b = 1.f / mB;
	float c = glm::dot(glm::cross(IinvA * glm::cross(rA, norm), rA), norm);
	float d = glm::dot(glm::cross(IinvB * glm::cross(rB, norm), rB), norm);
	float denominator = a + b + c + d;
	float j = numerator / denominator;

	return j;
}

__global__ void HandleRigidBodyCollisionKernelUnmapped(
	float radius, // Input: particle radius - assuming uniform radius over all virtual particles
	float4 *pos, // Input: rigid body position
	float4 *vel, // Input: rigid body velocity
	float4 *ang, // Input: rigid body angular velocity
	float4 *linMom, // Input: rigid body linear momentum
	float4 *angMom, // Input: rigid body angular momentum
	glm::mat3 *Iinv, // Input: rigid body inverse inertia matrix
	float *mass, // Input: rigid body mass
	int *rbIndices, // Input: index showing where each particle belongs
	float4 *particlePos, // Input: particle position
	int *collidingRigidBodyIndex, // Input: index of rigid body of contact
	int *collidingParticleIndex, // Input: index of particle of contact
	float *contactDistance, // Input: contact distance
	int *contactID, // Input: sorted array of contacts (by distance)
	int *cumulative_particles, // Input: cumulative sum of rigid body particles
	int numRigidBodies)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numRigidBodies)
		return;

	int contact_index = contactID[cumulative_particles[index] - 1];
	float d = contactDistance[contact_index];
	if (d <= 0)
		return;

	int collidingRigidBody = collidingRigidBodyIndex[contact_index];
	if (index > collidingRigidBody) // ensure that collisions will only be handled once
		return;

	int collidingParticle = collidingParticleIndex[contact_index];

	float4 p1 = particlePos[contact_index];
	float4 p2 = particlePos[collidingParticle];

	float4 CM1 = pos[index];
	float4 CM2 = pos[collidingRigidBody];

	float4 cp, cn; // exact contact point and contact normal

	float4 displacementVector = p2 - p1;
	float displacementDistance = length(displacementVector);
	float dr = (2 * radius - displacementDistance) / 2;
	displacementVector = normalize(displacementVector);
	CM1 -= displacementVector * dr;

	FindExactContactPointKernelAuxil(
		p1, p2,
		radius, radius,
		&cp, &cn);

	float4 velA = vel[index]; float4 velB = vel[collidingRigidBody];

	float4 angA = ang[index]; float4 angB = ang[collidingRigidBody];
	glm::mat3 IinvA = Iinv[index]; glm::mat3 IinvB = Iinv[collidingRigidBody];
	float4 dispA = CM1 - cp; float4 dispB = CM2 - cp;
	float mA = mass[index]; float mB = mass[collidingRigidBody];

	float impulse = ComputeImpulseMagnitudeKernel(
		velA, velB, angA, angB,
		dispA, dispB, IinvA,
		IinvB, mA, mB, cn);

	

	// compute linear and angular impulses using Baraff's method
	float4 LinearImpulse = cn * impulse;
	glm::vec3 AngularImpulse = IinvA *
		(glm::cross(glm::vec3(dispA.x, dispA.y, dispA.z),
		glm::vec3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z)));

	// apply impulses to rigid body
	vel[index] += LinearImpulse / mA;
	ang[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

	AngularImpulse = IinvB *
		(glm::cross(glm::vec3(dispB.x, dispB.y, dispB.z),
		glm::vec3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z) * (-1.f)));

	// apply impulses to rigid body
	vel[collidingRigidBody] -= LinearImpulse / mB;
	ang[collidingRigidBody] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

	//// auxiliary variables
	//float4 impulseVector = cn * impulse;

	//// apply linear impulse
	//linMom[index] += impulseVector;
	//linMom[collidingRigidBody] -= impulseVector;
	//
	//// apply angular impulse
	//float4 angularImpulse = make_float4(cross(make_float3(dispA), make_float3(impulseVector)), 0);
	//angMom[index] += angularImpulse;

	//angularImpulse = make_float4(cross(make_float3(dispB), make_float3(impulseVector) * (-1.f)), 0);
	//angMom[collidingRigidBody] += angularImpulse;
}

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
	SimParams params) // simulation parameters
{
	// kernel invocation auxiliaries
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	// we have successfully detected collisions between particles and
	// all walls
	// now we need to sort contacts based on their penetration distance
	// for each rigid body
	// to determine appropriate contact we will use an index array
	// denoting each particle
	// we will then sort the indices belonging to each rigid body
	// according to their contact distance to find the most important
	// contact
	// if that distance is positive we will handle the collision
	// appropriately using Baraff's method

	int *particleIndicesGPU;
	checkCudaErrors(cudaMalloc((void**)&particleIndicesGPU, sizeof(int) * numParticles));
	CreateIndexKernel << < gridDim, blockDim >> >(particleIndicesGPU, numParticles);

	int *rigid_body_flags_CPU = new int[numParticles]; // assuming all particles belong to a rigid body
	int total_particles_processed = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerRB[index]; particle++)
		{
			rigid_body_flags_CPU[total_particles_processed++] = index;
		}
	}
	int *rigid_body_flags_GPU;
	checkCudaErrors(cudaMalloc((void**)&rigid_body_flags_GPU, sizeof(int) * numParticles));
	checkCudaErrors(cudaMemcpy(rigid_body_flags_GPU, rigid_body_flags_CPU, sizeof(int) * numParticles, cudaMemcpyHostToDevice));

	Cub_Double_Sort(contactDistance, rigid_body_flags_GPU, &particleIndicesGPU, numParticles);

	int *cum_sum_indices_CPU = new int[numRigidBodies];
	cum_sum_indices_CPU[0] = particlesPerRB[0];
	for (int index = 1; index < numRigidBodies; index++)
	{
		cum_sum_indices_CPU[index] = cum_sum_indices_CPU[index - 1] + particlesPerRB[index];
	}
	int *cum_sum_indices_GPU;
	checkCudaErrors(cudaMalloc((void**)&cum_sum_indices_GPU, sizeof(int) * numRigidBodies));
	checkCudaErrors(cudaMemcpy(cum_sum_indices_GPU, cum_sum_indices_CPU, sizeof(int) * numRigidBodies, cudaMemcpyHostToDevice));


	// we have now found the most important contacts for
	// all the rigid bodies in the scene
	// kernel invocation auxiliaries
	blockDim = dim3(numThreads, 1);
	gridDim = dim3((numRigidBodies + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
		gridDim.x = 1;
	HandleRigidBodyCollisionKernelUnmapped << < gridDim, blockDim >> >(
		params.particleRadius, // Input: particle radius - assuming uniform radius over all virtual particles
		rbPos, // Input: rigid body position
		rbVel, // Input: rigid body velocity
		rbAng, // Input: rigid body angular velocity
		rbLinMom, // Input: rigid body linear momentum
		rbAngMom, // Input: rigid body angular momentum
		Iinv, // Input: rigid body inverse inertia matrix
		rbMass, //Input: rigid body mass
		rbIndices, // Input: index showing where each particle belongs
		particlePos, // Input: particle position
		collidingRigidBodyIndex, // Input: index of rigid body of contact
		collidingParticleIndex, // Input: index of particle of contact
		contactDistance, // Input: contact distance
		particleIndicesGPU, // Input: sorted array of contacts (by distance)
		cum_sum_indices_GPU, // Input: cumulative sum of rigid body particles
		numRigidBodies);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// clean-up stray pointers
	if (particleIndicesGPU)
		checkCudaErrors(cudaFree(particleIndicesGPU));
	if (rigid_body_flags_GPU)
		checkCudaErrors(cudaFree(rigid_body_flags_GPU));
	if (rigid_body_flags_CPU)
		delete rigid_body_flags_CPU;
	if (cum_sum_indices_GPU)
		checkCudaErrors(cudaFree(cum_sum_indices_GPU));
	if (cum_sum_indices_CPU)
		delete cum_sum_indices_CPU;

}


__device__ float ComputeImpulseMagnitudeKernelAuxil(
	float4 vel, float4 ang, float4 disp,
	glm::mat3 Iinv, float m, float4 n)
{
	glm::vec3 v(vel.x, vel.y, vel.z);

	glm::vec3 w(ang.x, ang.y, ang.z);

	glm::vec3 r(disp.x, disp.y, disp.z);

	glm::vec3 norm(n.x, n.y, n.z);

	glm::vec3 velA = v + glm::cross(w, r);
	float epsilon = 1;
	float numerator = -(1 + epsilon) * (glm::dot(velA, norm));
	float a = 1.f / m;
	float b = glm::dot(glm::cross(Iinv * glm::cross(r, norm), r), norm);
	float denominator = a + b;
	float j = numerator / denominator;

	return j;
}
__global__ void HandleAugmentedRealityCollisionKernelUnmapped(
	float radius, // Input: particle radius - assuming uniform radius over all virtual particles
	float4 *pos, // Input: rigid body position
	float4 *vel, // Input: rigid body velocity
	float4 *ang, // Input: rigid body angular velocity
	float4 *linMom, // Input: rigid body linear momentum
	float4 *angMom, // Input: rigid body angular momentum
	glm::mat3 *Iinv, // Input: rigid body inverse inertia matrix
	float *mass, // Input: rigid body mass
	int *rbIndices, // Input: index showing where each particle belongs
	float4 *particlePos, // Input: particle position
	float4 *scenePos, // Input: scene particle position
	int *collidingParticleIndex, // Input: index of particle of contact
	float *contactDistance, // Input: contact distance
	int *contactID, // Input: sorted array of contacts (by distance)
	int *cumulative_particles, // Input: cumulative sum of rigid body particles
	int numRigidBodies)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numRigidBodies)
		return;

	int contact_index = contactID[cumulative_particles[index] - 1];
	for (int collision = contact_index; collision > contact_index - 4; collision--) // handle four most important collisions
	{
		float d = contactDistance[contact_index];
		if (d <= 0)
			return;

		int collidingParticle = collidingParticleIndex[contact_index];

		float4 p1 = particlePos[contact_index];

		float4 CM1 = pos[index];
		float4 p2 = scenePos[collidingParticle];

		float4 cp, cn; // exact contact point and contact normal

		float4 displacement = p1 - CM1;

		float4 displacementVector = p2 - p1;
		float displacementDistance = length(displacementVector);
		float dr = (2 * radius - displacementDistance) / 2;
		displacementVector = normalize(displacementVector);
		
		CM1 -= displacementVector * dr;

		p1 = CM1 + displacement;

		FindExactContactPointKernelAuxil(p1, p2, radius, radius, &cp, &cn);

		float4 velA = vel[index]; 

		float4 angA = ang[index];
		glm::mat3 IinvA = Iinv[index];
		float4 dispA = CM1 - cp;
		float mA = mass[index];

		float impulse = ComputeImpulseMagnitudeKernelAuxil(velA, angA, dispA, IinvA, mA, cn);

		// compute linear and angular impulses using Baraff's method
		float4 LinearImpulse = cn * impulse;
		glm::vec3 AngularImpulse = IinvA *
			(glm::cross(glm::vec3(dispA.x, dispA.y, dispA.z),
			glm::vec3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z)));

		// apply impulses to rigid body
		vel[index] += LinearImpulse / mA;
		ang[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

		// correct rigid body position
		//pos[index] = CM1;
	}

}

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
	SimParams params) // simulation parameters
{
	// kernel invocation auxiliaries
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	// we have successfully detected collisions between particles and
	// all scene particles
	// for each particle we have kept only the most important contact
	// i.e. the one that presents the largest penetration distance
	// now we need to sort contacts based on their penetration distance
	// for each rigid body
	// for augmented reality simulations we will only use the four most
	// important contacts for each rigid body
	// if their penetration distance is positive
	// we will handle the collision using Baraff's method

	int *particleIndicesGPU;
	checkCudaErrors(cudaMalloc((void**)&particleIndicesGPU, sizeof(int) * numParticles));
	CreateIndexKernel << < gridDim, blockDim >> >(particleIndicesGPU, numParticles);

	int *rigid_body_flags_CPU = new int[numParticles]; // assuming all particles belong to a rigid body
	int total_particles_processed = 0;
	for (int index = 0; index < numRigidBodies; index++)
	{
		for (int particle = 0; particle < particlesPerRB[index]; particle++)
		{
			rigid_body_flags_CPU[total_particles_processed++] = index;
		}
	}
	int *rigid_body_flags_GPU;
	checkCudaErrors(cudaMalloc((void**)&rigid_body_flags_GPU, sizeof(int) * numParticles));
	checkCudaErrors(cudaMemcpy(rigid_body_flags_GPU, rigid_body_flags_CPU, sizeof(int) * numParticles, cudaMemcpyHostToDevice));

	Cub_Sort_By_Key_Wrapper<int, float>(&particleIndicesGPU, &contactDistance, numParticles);

	Cub_Sort_By_Key_Wrapper<int, float>(&rigid_body_flags_GPU, &contactDistance, numParticles);

	Cub_Sort_By_Key_Wrapper<int, int>(&particleIndicesGPU, &rigid_body_flags_GPU, numParticles);


	int *cum_sum_indices_CPU = new int[numRigidBodies];
	cum_sum_indices_CPU[0] = particlesPerRB[0];
	for (int index = 1; index < numRigidBodies; index++)
	{
		cum_sum_indices_CPU[index] = cum_sum_indices_CPU[index - 1] + particlesPerRB[index];
	}
	int *cum_sum_indices_GPU;
	checkCudaErrors(cudaMalloc((void**)&cum_sum_indices_GPU, sizeof(int) * numRigidBodies));
	checkCudaErrors(cudaMemcpy(cum_sum_indices_GPU, cum_sum_indices_CPU, sizeof(int) * numRigidBodies, cudaMemcpyHostToDevice));


	// we have now found the most important contacts for
	// all the rigid bodies in the scene
	// kernel invocation auxiliaries
	blockDim = dim3(numThreads, 1);
	gridDim = dim3((numRigidBodies + numThreads - 1) / numThreads, 1);
	if (gridDim.x < 1)
		gridDim.x = 1;

	HandleAugmentedRealityCollisionKernelUnmapped << < gridDim, blockDim >> >(
		params.particleRadius, // Input: particle radius - assuming uniform radius over all virtual particles
		rbPos, // Input: rigid body position
		rbVel, // Input: rigid body velocity
		rbAng, // Input: rigid body angular velocity
		rbLinMom, // Input: rigid body linear momentum
		rbAngMom, // Input: rigid body angular momentum
		Iinv, // Input: rigid body inverse inertia matrix
		rbMass, //Input: rigid body mass
		rbIndices, // Input: index showing where each particle belongs
		particlePos, // Input: particle position
		scenePos, // Input: scene particle position
		collidingParticleIndex, // Input: index of particle of contact
		contactDistance, // Input: contact distance
		particleIndicesGPU, // Input: sorted array of contacts (by distance)
		cum_sum_indices_GPU, // Input: cumulative sum of rigid body particles
		numRigidBodies);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// clean-up stray pointers
	if (particleIndicesGPU)
		checkCudaErrors(cudaFree(particleIndicesGPU));
	if (rigid_body_flags_GPU)
		checkCudaErrors(cudaFree(rigid_body_flags_GPU));
	if (rigid_body_flags_CPU)
		delete rigid_body_flags_CPU;
	if (cum_sum_indices_GPU)
		checkCudaErrors(cudaFree(cum_sum_indices_GPU));
	if (cum_sum_indices_CPU)
		delete cum_sum_indices_CPU;

}