#include "BVHAuxiliary.cuh"
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#define inf 0x7f800000 
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

__global__ void CreateIndexKernel(int *indices, int elements)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= elements)
		return;
	indices[index] = index;
}

/*
* This kernel computes collisions with one wall at a time.
* It will be called 6 times in serial to compute collisions
* with all walls. Wall collision is tested by comparing the
* particle's coordinate along the wall direction dot(p, n)
* with the wall's position. Special care must be given to
* the signs. For every particle we store contact distance
* (negative if there is no contact), position and normal.
*/
__global__ void DetectWallCollisionKernel(
	float radius, // Input: particle radius
	float4 *pos, // Input: particle position
	int *rigidBodyIndex, // Input: rigid body index for each particle
	float wallPos, // Input: wall position
	float4 n, // Input: wall direction
	float4 *contactNormal, // Output: contact normal
	float4 *contactPosition, // Output: contact position
	float4 *contactWallPosition, // Output: wall position
	float *contactDistance, // Output: contact distance
	int numParticles) // number of particles
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	int correspondingRigidBody = rigidBodyIndex[index];
	if (correspondingRigidBody == -1)
		return; //if this is an independent virtual particle stop

	float4 p = pos[index];
	float4 normN = normalize(n); // normN is independent of sign
	float sign = dot(n, normN); // 1 if n is positive and -1 if n negative
	// ISSUE: signs do not work for all walls
	// if sign is -1 then we need < 0 comparison
	// if sign is 1 then we need a > 0 comparison
	// this way if distance > 0 we have successfully
	// detected wall penetration
	float distance = sign * (dot(p, normN) - wallPos + radius * sign);
	// this allows us to test multiple wall collisions
	// and pick the one that is the most important
	// ISSUE: how is contactDistance initialized?
	if (distance > contactDistance[index])
	{
		contactDistance[index] = distance;
		contactNormal[index] = n;
		contactPosition[index] = p;
		contactWallPosition[index] = n * wallPos;
	}
}

/*
* This function calculates the exact contact point between two particles.
*/
__device__ void FindExactContactPointKernel(
	float4 point1, float4 point2,
	float radius1, float radius2,
	float4 *cp, float4 *cn) // exact contact point and contact normal
{
	float t = radius1 / (radius1 + radius2);
	*cp = point1 + (point2 - point1) * t;
	*cn = normalize(point2 - point1) * (-1.f);
}

/*
* Compute impulse magnitude after wall collision using Baraff's method
*/
__device__ float ComputeImpulseMagnitudeKernel(
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


__global__ void HandleWallCollisionKernel(
	float radius, // Input: particle radius
	float4 *pos, // Input: rigid body position
	float4 *vel, // Input: rigid body velocity
	float4 *ang, // Input: rigid body angular velocity
	glm::mat3 *Iinv, // Input: rigid body inverse inertia matrix
	float *mass, //Input: rigid body mass
	float4 *contactNormal, // Input: contact normal
	float4 *contactWallPosition, // Input: contact position of wall
	float4 *contactParticlePosition, // Input: contact position of particle
	float *contactDistance, // Input: contact distance
	int numRigidBodies) // number of rigid bodies
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numRigidBodies)
		return;
	float d = contactDistance[index];
	if (d <= 0)
		return;
	float4 n = contactNormal[index];
	float4 wall = contactWallPosition[index];
	float4 p = contactParticlePosition[index];
	float4 pAuxil = n * dot(p, n);

	// load old variables
	float4 oldCM = pos[index];
	float4 oldVel = vel[index];
	float4 oldAng = ang[index];
	glm::mat3 oldInv = Iinv[index];
	float oldMass = mass[index];

	// displacement is old center of mass minus old particle position
	float4 oldDisp = oldCM - p;
	// move center of mass so that rigid body and wall no longer overlap
	float4 newCM = oldCM + wall - radius * n - pAuxil;

	// find new particle position
	float4 newP = newCM + oldDisp;
	float4 cn, cp;
	// find exact collision point
	FindExactContactPointKernel(newP, newP + n * radius, radius, 0, &cp, &cn);
	float4 r = newCM - cp;
	// computer impulse magnitude
	float impulse = ComputeImpulseMagnitudeKernel(oldVel, oldAng,
		r, oldInv,
		oldMass, cn);

	// compute linear and angular impulses using Baraff's method
	float4 LinearImpulse = cn * impulse;
	glm::vec3 AngularImpulse = oldInv * 
		(glm::cross(glm::vec3(r.x, r.y, r.z),
		glm::vec3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z)));

	// apply impulses to rigid body
	vel[index] += LinearImpulse / oldMass;
	ang[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);
}

__global__ void HandleWallCollisionKernelUnmapped(
	float radius, // Input: particle radius
	float4 *pos, // Input: rigid body position
	float4 *vel, // Input: rigid body velocity
	float4 *ang, // Input: rigid body angular velocity
	glm::mat3 *Iinv, // Input: rigid body inverse inertia matrix
	float *mass, //Input: rigid body mass
	float4 *contactNormal, // Input: contact normal
	float4 *contactWallPosition, // Input: contact position of wall
	float4 *contactParticlePosition, // Input: contact position of particle
	float *contactDistance, // Input: contact distance
	int *contactID, // Input: sorted array of contacts (by distance)
	int *cumulative_particles, // Input: cumulative sum of rigid body particles
	int numRigidBodies) // number of rigid bodies
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numRigidBodies)
		return;

	int contact_index = contactID[cumulative_particles[index] - 1];
	float d = contactDistance[contact_index];
	if (d <= 0)
		return;


	float4 n = contactNormal[contact_index];
	float4 wall = contactWallPosition[contact_index];
	float4 p = contactParticlePosition[contact_index];
	float4 pAuxil = n * dot(p, n);

	// load old variables
	float4 oldCM = pos[index];
	float4 oldVel = vel[index];
	float4 oldAng = ang[index];
	glm::mat3 oldInv = Iinv[index];
	float oldMass = mass[index];

	// displacement is old center of mass minus old particle position
	float4 oldDisp = oldCM - p;
	// move center of mass so that rigid body and wall no longer overlap
	float4 newCM = oldCM + wall - radius * n - pAuxil;

	// find new particle position
	float4 newP = newCM + oldDisp;
	float4 cn, cp;
	// find exact collision point
	FindExactContactPointKernel(newP, newP + n * radius, radius, 0, &cp, &cn);
	float4 r = newCM - cp;
	// computer impulse magnitude
	float impulse = ComputeImpulseMagnitudeKernel(oldVel, oldAng,
		r, oldInv,
		oldMass, cn);

	// compute linear and angular impulses using Baraff's method
	float4 LinearImpulse = cn * impulse;
	glm::vec3 AngularImpulse = oldInv *
		(glm::cross(glm::vec3(r.x, r.y, r.z),
		glm::vec3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z)));

	// apply impulses to rigid body
	vel[index] += LinearImpulse / oldMass;
	ang[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);
}

//void Cub_Double_Sort(float **d_float_key, int **d_int_key, int **d_value, const int& arraySize)
//{
//
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// create sorted auxiliaries
//	float *d_float_key_sorted;
//	int *d_int_key_sorted, *d_value_sorted;
//	checkCudaErrors(cudaMalloc((void**)&d_float_key_sorted, arraySize * sizeof(float)));
//
//	checkCudaErrors(cudaMalloc((void**)&d_int_key_sorted, arraySize * sizeof(int)));
//
//	checkCudaErrors(cudaMalloc((void**)&d_value_sorted, arraySize * sizeof(int)));
//
//	//float *test_distances = new float[arraySize];
//	//checkCudaErrors(cudaMemcpy(test_distances, *d_float_key, sizeof(float) * arraySize, cudaMemcpyDeviceToHost));
//	//bool print_test = false;
//	//for (int i = 0; i < arraySize; i++)
//	//{
//	//	if (test_distances[i] > 0)
//	//	{
//	//		print_test = true;
//	//		break;
//	//	}
//	//}
//	//if (print_test)
//	//{
//	//	for (int i = 0; i < arraySize; i++)
//	//	{
//	//		std::cout << "Distance (pre-sort): " << test_distances[i] << " ";
//	//		std::cout << std::endl;
//	//	}
//	//}
//
//	cub::CachingDeviceAllocator  g_allocator(true);
//	cub::DoubleBuffer<float> sortKeys_float;
//	cub::DoubleBuffer<int> sortValues;
//	size_t  temp_storage_bytes = 0;
//	void    *d_temp_storage = NULL;
//
//	// Allocate temporary storage
//	sortKeys_float.d_buffers[0] = *d_float_key;
//	sortValues.d_buffers[0] = *d_value;
//	sortKeys_float.d_buffers[1] = d_float_key_sorted;
//	sortValues.d_buffers[1] = d_value_sorted;
//
//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_float, sortValues, arraySize));
//	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_float, sortValues, arraySize));
//
//	d_value_sorted = sortValues.Current();
//	d_float_key_sorted = sortKeys_float.Current();
//	*d_value = sortValues.Alternate();
//	*d_float_key = sortKeys_float.Alternate();
//
//
//
//	//float *test_distances_sorted = new float[arraySize];
//	//checkCudaErrors(cudaMemcpy(test_distances_sorted, d_float_key_sorted, sizeof(float) * arraySize, cudaMemcpyDeviceToHost));
//	//if (print_test)
//	//{
//	//	for (int i = 0; i < arraySize; i++)
//	//	{
//	//		std::cout << "Distance (unsorted): " << test_distances[i] << " ";
//	//		std::cout << "Distance (sorted): " << test_distances_sorted[i] << " ";
//	//		std::cout << std::endl;
//	//	}
//	//}
//	if (d_temp_storage)
//		cudaFree(d_temp_storage);
//	temp_storage_bytes = 0;
//	d_temp_storage = NULL;
//
//	// Allocate temporary storage
//	cub::DoubleBuffer<float> sortIntKeys_float;
//	cub::DoubleBuffer<int> sortIntValues_float;
//	sortIntKeys_float.d_buffers[0] = *d_float_key;
//	sortIntValues_float.d_buffers[0] = *d_int_key;
//	sortIntKeys_float.d_buffers[1] = d_float_key_sorted;
//	sortIntValues_float.d_buffers[1] = d_int_key_sorted;
//
//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntKeys_float, sortIntValues_float, arraySize));
//	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntKeys_float, sortIntValues_float, arraySize));
//
//	d_int_key_sorted = sortIntValues_float.Current();
//	d_float_key_sorted = sortIntKeys_float.Current();
//	*d_int_key = sortIntValues_float.Alternate();
//	*d_float_key = sortIntKeys_float.Alternate();
//
//	if (d_temp_storage)
//		cudaFree(d_temp_storage);
//	temp_storage_bytes = 0;
//	d_temp_storage = NULL;
//
//	
//	//if (print_test)
//	//{
//	//	int *test_flags = new int[arraySize];
//	//	checkCudaErrors(cudaMemcpy(test_flags, d_int_key_sorted, sizeof(int) * arraySize, cudaMemcpyDeviceToHost));
//	//	int *test_indices = new int[arraySize];
//	//	checkCudaErrors(cudaMemcpy(test_indices, *d_value, sizeof(int) * arraySize, cudaMemcpyDeviceToHost));
//	//	std::cout << "Unsorted Contacts: " << std::endl;
//	//	for (int i = 0; i < arraySize; i++)
//	//	{
//	//		std::cout << "Particle Index (unsorted): " << test_indices[i] << " ";
//	//		std::cout << "Distance (unsorted): " << test_distances[i] << " ";
//	//		std::cout << "Rigid Body Index (unsorted): " << test_flags[i] << " ";
//	//		std::cout << std::endl;
//	//	}
//	//	delete test_flags;
//	//	delete test_indices;
//	//}
//	//delete test_distances;
//	cub::DoubleBuffer<int> sortKeys_int;
//	cub::DoubleBuffer<int> sortValues_int;
//	sortKeys_int.d_buffers[0] = d_int_key_sorted;
//	sortValues_int.d_buffers[0] = d_value_sorted;
//	sortKeys_int.d_buffers[1] = *d_int_key;
//	sortValues_int.d_buffers[1] = *d_value;
//
//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_int, sortValues_int, arraySize));
//	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_int, sortValues_int, arraySize));
//
//	*d_value = sortValues_int.Current();
//	*d_int_key = sortKeys_int.Current();
//	d_value_sorted = sortValues_int.Alternate();
//	d_int_key_sorted = sortKeys_int.Alternate();
//
//	//if (print_test)
//	//{
//	//	int *test_flags = new int[arraySize];
//	//	checkCudaErrors(cudaMemcpy(test_flags, *d_int_key, sizeof(int) * arraySize, cudaMemcpyDeviceToHost));
//	//	int *test_indices = new int[arraySize];
//	//	checkCudaErrors(cudaMemcpy(test_indices, *d_value, sizeof(int) * arraySize, cudaMemcpyDeviceToHost));
//	//	std::cout << "Sorted Contacts: " << std::endl;
//	//	for (int i = 0; i < arraySize; i++)
//	//	{
//	//		std::cout << "Particle Index (sorted): " << test_indices[i] << " ";
//	//		std::cout << "Rigid Body Index (sorted): " << test_flags[i] << " ";
//	//		std::cout << std::endl;
//	//	}
//	//	delete test_flags;
//	//	delete test_indices;
//	//}
//
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	if (d_float_key_sorted)
//		checkCudaErrors(cudaFree(d_float_key_sorted));
//	if (d_int_key_sorted)
//		checkCudaErrors(cudaFree(d_int_key_sorted));
//	if (d_value_sorted)
//		checkCudaErrors(cudaFree(d_value_sorted));
//	if (d_temp_storage)
//		checkCudaErrors(cudaFree(d_temp_storage));
//}

void Cub_Double_Sort(float **d_float_key, int **d_int_key, int **d_value, const int& arraySize)
{

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// create sorted auxiliaries
	float *d_float_key_sorted;
	int *d_int_key_sorted, *d_value_sorted;
	checkCudaErrors(cudaMalloc((void**)&d_float_key_sorted, arraySize * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_int_key_sorted, arraySize * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&d_value_sorted, arraySize * sizeof(int)));


	cub::CachingDeviceAllocator  g_allocator(true);
	cub::DoubleBuffer<float> sortKeys_float;
	cub::DoubleBuffer<int> sortValues;
	size_t  temp_storage_bytes = 0;
	void    *d_temp_storage = NULL;

	// Allocate temporary storage
	sortKeys_float.d_buffers[0] = *d_float_key;
	sortValues.d_buffers[0] = *d_value;
	sortKeys_float.d_buffers[1] = d_float_key_sorted;
	sortValues.d_buffers[1] = d_value_sorted;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_float, sortValues, arraySize));
	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_float, sortValues, arraySize));

	d_value_sorted = sortValues.Current();
	d_float_key_sorted = sortKeys_float.Current();
	*d_value = sortValues.Alternate();
	*d_float_key = sortKeys_float.Alternate();
	if (d_temp_storage)
		cudaFree(d_temp_storage);
	temp_storage_bytes = 0;
	d_temp_storage = NULL;

	// Allocate temporary storage
	cub::DoubleBuffer<float> sortIntKeys_float;
	cub::DoubleBuffer<int> sortIntValues_float;
	sortIntKeys_float.d_buffers[0] = *d_float_key;
	sortIntValues_float.d_buffers[0] = *d_int_key;
	sortIntKeys_float.d_buffers[1] = d_float_key_sorted;
	sortIntValues_float.d_buffers[1] = d_int_key_sorted;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntKeys_float, sortIntValues_float, arraySize));
	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortIntKeys_float, sortIntValues_float, arraySize));

	d_int_key_sorted = sortIntValues_float.Current();
	d_float_key_sorted = sortIntKeys_float.Current();
	*d_int_key = sortIntValues_float.Alternate();
	*d_float_key = sortIntKeys_float.Alternate();

	if (d_temp_storage)
		cudaFree(d_temp_storage);
	temp_storage_bytes = 0;
	d_temp_storage = NULL;


	cub::DoubleBuffer<int> sortKeys_int;
	cub::DoubleBuffer<int> sortValues_int;
	sortKeys_int.d_buffers[0] = d_int_key_sorted;
	sortValues_int.d_buffers[0] = d_value_sorted;
	sortKeys_int.d_buffers[1] = *d_int_key;
	sortValues_int.d_buffers[1] = *d_value;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_int, sortValues_int, arraySize));
	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sortKeys_int, sortValues_int, arraySize));

	*d_value = sortValues_int.Current();
	*d_int_key = sortKeys_int.Current();
	d_value_sorted = sortValues_int.Alternate();
	d_int_key_sorted = sortKeys_int.Alternate();

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

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
* Test every particle against all walls for possible wall collisions.
* Find most important collision (i.e. the one with the largest penetration)
* Handle only the most important collision for each rigid body
*/
void WallCollisionWrapper(
	float4 *particlePos, // particle positions
	float4 *rbPos, // rigid body center of mass
	float3 minPos, // scene AABB's smallest coordinates
	float3 maxPos, // scene AABB's largest coordinates
	float4 *rbVel, // rigid body linear velocity
	float4 *rbAng, // rigid body angular velocity
	glm::mat3 *Iinv, // current rigid body inverse inertia tensor
	float *rbMass, // rigid body mass
	int *rbIndices, // index showing where each particle belongs
	int *particlesPerRB, // number of particles per rigid body
	int numRigidBodies, // total number of scene's rigid bodies
	int numParticles, // number of particles to test
	int numThreads, // number of threads to use
	SimParams params) // simulation parameters
{
	// kernel invocation auxiliaries
	dim3 blockDim(numThreads, 1);
	dim3 gridDim((numParticles + numThreads - 1) / numThreads, 1);

	// contact info
	float4 *contactNormal, *contactPosition, *contactWallPosition;
	float *contactDistance;
	checkCudaErrors(cudaMalloc((void**)&contactNormal, sizeof(float) * 4 * numParticles));
	checkCudaErrors(cudaMalloc((void**)&contactPosition, sizeof(float) * 4 * numParticles));
	checkCudaErrors(cudaMalloc((void**)&contactWallPosition, sizeof(float) * 4 * numParticles));
	checkCudaErrors(cudaMalloc((void**)&contactDistance, sizeof(float) * numParticles));

	checkCudaErrors(cudaMemset(contactNormal, 0, sizeof(float) * 4 * numParticles));
	checkCudaErrors(cudaMemset(contactPosition, 0, sizeof(float) * 4 * numParticles));
	checkCudaErrors(cudaMemset(contactWallPosition, 0, sizeof(float) * 4 * numParticles));
	checkCudaErrors(cudaMemset(contactDistance, 0, sizeof(float) * numParticles));

	// wall variables
	// right wall
	float4 n = make_float4(1, 0, 0, 0);
	float3 n3 = make_float3(n);
	float wallPos = dot(n3, maxPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		wallPos,
		n,
		contactNormal,
		contactPosition,
		contactWallPosition, 
		contactDistance,
		numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// top wall
	n = make_float4(0, 1, 0, 0);
	n3 = make_float3(n);
	wallPos = dot(n3, maxPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		wallPos,
		n,
		contactNormal,
		contactPosition,
		contactWallPosition,
		contactDistance,
		numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// far wall
	n = make_float4(0, 0, 1, 0);
	n3 = make_float3(n);
	wallPos = dot(n3, maxPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		wallPos,
		n,
		contactNormal,
		contactPosition,
		contactWallPosition,
		contactDistance,
		numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// left wall
	n = make_float4(-1, 0, 0, 0);
	n3 = make_float3(n);
	wallPos = dot(n3, minPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		wallPos,
		n,
		contactNormal,
		contactPosition,
		contactWallPosition,
		contactDistance,
		numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// bottom wall
	n = make_float4(0, -1, 0, 0);
	n3 = make_float3(n);
	wallPos = dot(n3, minPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		wallPos,
		n,
		contactNormal,
		contactPosition,
		contactWallPosition,
		contactDistance,
		numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// near wall
	n = make_float4(0, 0, -1, 0);
	n3 = make_float3(n);
	wallPos = dot(n3, minPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		wallPos,
		n,
		contactNormal,
		contactPosition,
		contactWallPosition,
		contactDistance,
		numParticles);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

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

	float4 *contactNormal_RB, *contactPosition_RB, *contactWallPosition_RB;
	float *contactDistance_RB;
	checkCudaErrors(cudaMalloc((void**)&contactNormal_RB, sizeof(float) * 4 * numRigidBodies));
	checkCudaErrors(cudaMalloc((void**)&contactPosition_RB, sizeof(float) * 4 * numRigidBodies));
	checkCudaErrors(cudaMalloc((void**)&contactWallPosition_RB, sizeof(float) * 4 * numRigidBodies));
	checkCudaErrors(cudaMalloc((void**)&contactDistance_RB, sizeof(float) * numRigidBodies));
	checkCudaErrors(cudaMemset(contactDistance_RB, 0, sizeof(float) * numRigidBodies));

	
	//int startIndex = 0;
	//int endIndex = 0;
	//for (int index = 0; index < numRigidBodies; index++)
	//{
	//	std::cout << "Processing rigid body #" << index << std::endl;
	//	// for every rigid body
	//	endIndex += particlesPerRB[index];
	//	// create temporary arrays to hold sorted values
	//	float *sortedDistances;
	//	int *sortedIndices;
	//	checkCudaErrors(cudaMalloc((void**)&sortedDistances, sizeof(float) * particlesPerRB[index]));
	//	checkCudaErrors(cudaMalloc((void**)&sortedIndices, sizeof(int) * particlesPerRB[index]));
	//	
	//	// create temporary arrays to hold unsorted values
	//	float *unsortedDistances;
	//	int *unsortedIndices;
	//	checkCudaErrors(cudaMalloc((void**)&unsortedDistances, sizeof(float) * particlesPerRB[index]));
	//	checkCudaErrors(cudaMalloc((void**)&unsortedIndices, sizeof(int) * particlesPerRB[index]));
	//	
	//	// copy unsorted values to their placeholders
	//	checkCudaErrors(cudaMemcpy(unsortedDistances,
	//		&contactDistance[startIndex],
	//		sizeof(int) * particlesPerRB[index],
	//		cudaMemcpyDeviceToDevice)); 
	//	checkCudaErrors(cudaMemcpy(unsortedIndices,
	//		&particleIndicesGPU[startIndex],
	//		sizeof(int) * particlesPerRB[index],
	//		cudaMemcpyDeviceToDevice));

	//	cub::CachingDeviceAllocator  g_allocator(true);
	//	cub::DoubleBuffer<float> sortKeys; // keys to sort by - contact distance
	//	cub::DoubleBuffer<int> sortValues; // sort particle indices
	//	size_t  temp_storage_bytes = 0;
	//	void    *d_temp_storage = NULL;

	//	sortKeys.d_buffers[0] = unsortedDistances;
	//	sortValues.d_buffers[0] = unsortedIndices;

	//	sortKeys.d_buffers[1] = sortedDistances;
	//	sortValues.d_buffers[1] = sortedIndices;

	//	// sort contacts belonging to this rigid body only		
	//	
	//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage,
	//		temp_storage_bytes,
	//		sortKeys,
	//		sortValues,
	//		particlesPerRB[index]));


	//	checkCudaErrors(g_allocator.DeviceAllocate(&d_temp_storage,
	//		temp_storage_bytes));


	//	checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage,
	//		temp_storage_bytes,
	//		sortKeys,
	//		sortValues,
	//		particlesPerRB[index]));

	//	sortedIndices = sortValues.Current();
	//	sortedDistances = sortKeys.Current();

	//	unsortedIndices = sortValues.Alternate();
	//	unsortedDistances = sortKeys.Alternate();
	//	
	//	int mostImportantContact; // index of most important contact
	//	// the most important contact is the one with the largest distance
	//	// we are sorting in ascending order so it is the last one
	//	checkCudaErrors(cudaMemcpy(&mostImportantContact,
	//		&sortedIndices[particlesPerRB[index] - 1],
	//		sizeof(int),
	//		cudaMemcpyDeviceToHost));


	//	checkCudaErrors(cudaMemcpy(&contactDistance_RB[index],
	//		&contactDistance[mostImportantContact],
	//		sizeof(float),
	//		cudaMemcpyDeviceToDevice));

	//	checkCudaErrors(cudaMemcpy(&contactNormal_RB[index],
	//		&contactNormal[mostImportantContact],
	//		sizeof(float) * 4,
	//		cudaMemcpyDeviceToDevice));

	//	checkCudaErrors(cudaMemcpy(&contactPosition_RB[index],
	//		&contactPosition[mostImportantContact],
	//		sizeof(float) * 4,
	//		cudaMemcpyDeviceToDevice));

	//	checkCudaErrors(cudaMemcpy(&contactWallPosition_RB[index],
	//		&contactWallPosition[mostImportantContact],
	//		sizeof(float) * 4,
	//		cudaMemcpyDeviceToDevice));

	//	if (sortedDistances)
	//		checkCudaErrors(cudaFree(sortedDistances));
	//	if (sortedIndices)
	//		checkCudaErrors(cudaFree(sortedIndices));
	//	if (unsortedDistances)
	//		checkCudaErrors(cudaFree(unsortedDistances));
	//	if (unsortedIndices)
	//		checkCudaErrors(cudaFree(unsortedIndices));
	//	if (d_temp_storage)
	//		checkCudaErrors(cudaFree(d_temp_storage));

	//	startIndex += particlesPerRB[index];
	//}

	/*
	* Testing Cub_Double_Sort Output
	*/
	//const int blockSize = 47;
	//const int elements = 36;
	//const int arraySize = blockSize * elements;
	//float a[arraySize] = { 0 };
	//int id[arraySize] = { 0 };
	//int c[arraySize] = { 0 };

	//std::cout << "Input: ";
	//for (int i = 0; i < arraySize; i++)
	//{
	//	if (!(i % blockSize))
	//		a[i] = arraySize - i;
	//	c[i] = i + 1;
	//	std::cout << c[i] << " ";
	//}
	//std::cout << std::endl;

	//std::cout << "Distance: ";
	//for (int i = 0; i < arraySize; i++)
	//{
	//	std::cout << a[i] << " ";
	//}
	//std::cout << std::endl;

	//for (int i = 0; i < elements; i++)
	//{
	//	for (int j = 0; j < blockSize; j++)
	//	{
	//		id[i * blockSize + j] = i + 1;
	//	}
	//}



	////Cub_Reduce_Test(a, id, c, elements, blockSize);
	//float *d_float_key;
	//int *d_int_key, *d_value;
	//checkCudaErrors(cudaMalloc((void**)&d_float_key, arraySize * sizeof(float)));
	//checkCudaErrors(cudaMemcpy(d_float_key, a, arraySize * sizeof(float), cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMalloc((void**)&d_int_key, arraySize * sizeof(int)));
	//checkCudaErrors(cudaMemcpy(d_int_key, id, arraySize * sizeof(int), cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMalloc((void**)&d_value, arraySize * sizeof(int)));
	//checkCudaErrors(cudaMemcpy(d_value, c, arraySize * sizeof(int), cudaMemcpyHostToDevice));

	//Cub_Double_Sort(&d_float_key, &d_int_key, &d_value, arraySize);

	//checkCudaErrors(cudaMemcpy(c, d_value, arraySize * sizeof(int), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(a, d_float_key, arraySize * sizeof(int), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(id, d_int_key, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

	//std::cout << "Results: " << std::endl;
	//for (int i = 0; i < arraySize; i++)
	//{
	//	std::cout << "ID: " << c[i] << " ";
	//	std::cout << "Float key: " << a[i] << " ";
	//	std::cout << "Int key: " << id[i] << " ";
	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;


	//if (d_float_key)
	//cudaFree(d_float_key);
	//if (d_int_key)
	//cudaFree(d_int_key);
	//if (d_value)
	//cudaFree(d_value);

	//std::cout << "Enter char to exit...";
	//int x;
	//std::cin >> x;

	/*
	* End of Cub_Double_Sort testing
	*/
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

	Cub_Double_Sort(&contactDistance, &rigid_body_flags_GPU, &particleIndicesGPU, numParticles);

	/*
	* Testing Cub_Double_Sort Output
	*/
	//int *test_indices = new int[numParticles];

	//checkCudaErrors(cudaMemcpy(test_indices, rigid_body_flags_GPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));
	//std::cout << "Sorted Indices: " << std::endl;
	//for (int i = 0; i < numParticles; i++)
	//{
	//	std::cout << test_indices[i] << " ";
	//}
	//std::cout << std::endl;
	//delete test_indices;

	//float *test_distances = new float[numParticles];

	//checkCudaErrors(cudaMemcpy(test_distances, contactDistance, sizeof(float) * numParticles, cudaMemcpyDeviceToHost));
	//bool print_test = false;
	//for (int i = 0; i < numParticles; i++)
	//{
	//	if (test_distances[i] > 0)
	//	{
	//		print_test = true;
	//		break;
	//	}
	//}
	//if (print_test)
	//{
	//	int *test_flags = new int[numParticles];

	//	checkCudaErrors(cudaMemcpy(test_flags, rigid_body_flags_GPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));

	//	int *test_indices = new int[numParticles];

	//	checkCudaErrors(cudaMemcpy(test_indices, particleIndicesGPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));

	//	std::cout << "Sorted Contacts: " << std::endl;
	//	for (int i = 0; i < numParticles; i++)
	//	{
	//		std::cout << "Particle Index (sorted): " << test_indices[i] << " ";
	//		std::cout << "Distance (unsorted): " << test_distances[i] << " ";
	//		std::cout << "Rigid Body Index (unsorted): " << test_flags[i] << " ";
	//		std::cout << std::endl;
	//	}
	//	delete test_flags;
	//	delete test_indices;
	//}

	//delete test_distances;

	/*
	* Testing Cub_Double_Sort Output
	*/

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
	HandleWallCollisionKernelUnmapped << < gridDim, blockDim >> >(
		params.particleRadius, // Input: particle radius
		rbPos, // Input: rigid body position
		rbVel, // Input: rigid body velocity
		rbAng, // Input: rigid body angular velocity
		Iinv, // Input: rigid body inverse inertia matrix
		rbMass, //Input: rigid body mass
		contactNormal, // Input: contact normal
		contactWallPosition, // Input: contact position of wall
		contactPosition, // Input: contact position of particle
		contactDistance, // Input: contact distance
		particleIndicesGPU, // Input: sorted array of contacts (by distance)
		cum_sum_indices_GPU, // Input: cumulative sum of rigid body particles
		numRigidBodies); // number of rigid bodies

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	//HandleWallCollisionKernel << < gridDim, blockDim >> >(
	//	params.particleRadius, // Input: particle radius
	//	rbPos, // Input: rigid body position
	//	rbVel, // Input: rigid body velocity
	//	rbAng, // Input: rigid body angular velocity
	//	Iinv, // Input: rigid body inverse inertia matrix
	//	rbMass, //Input: rigid body mass
	//	contactNormal_RB, // Input: contact normal
	//	contactWallPosition_RB, // Input: contact position of wall
	//	contactPosition_RB, // Input: contact position of particle
	//	contactDistance_RB, // Input: contact distance
	//	numRigidBodies); // number of rigid bodies

	// clean-up stray pointers
	if (particleIndicesGPU)
		checkCudaErrors(cudaFree(particleIndicesGPU));
	if (contactNormal)
		checkCudaErrors(cudaFree(contactNormal));
	if (contactPosition)
		checkCudaErrors(cudaFree(contactPosition));
	if (contactDistance)
		checkCudaErrors(cudaFree(contactDistance));
	if (contactNormal_RB)
		checkCudaErrors(cudaFree(contactNormal_RB));
	if (contactPosition_RB)
		checkCudaErrors(cudaFree(contactPosition_RB));
	if (contactDistance_RB)
		checkCudaErrors(cudaFree(contactDistance_RB));
	if (contactWallPosition_RB)
		checkCudaErrors(cudaFree(contactWallPosition_RB));
	if (rigid_body_flags_GPU)
		checkCudaErrors(cudaFree(rigid_body_flags_GPU));
	if (rigid_body_flags_CPU)
		delete rigid_body_flags_CPU;
	if (cum_sum_indices_GPU)
		checkCudaErrors(cudaFree(cum_sum_indices_GPU));
	if (cum_sum_indices_CPU)
		delete cum_sum_indices_CPU;
	
}


