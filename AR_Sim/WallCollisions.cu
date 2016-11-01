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

template<typename valueType, typename keyType>
void Cub_Sort_By_Key_Wrapper(valueType **value, const keyType *const *const key, const int &arraySize);

void Cub_Double_Sort(float *float_key, int *int_key, int **d_value, const int& arraySize); 


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
	float4 *rigidBodyVel, // Input: rigid body velocity
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

	float4 vel = rigidBodyVel[correspondingRigidBody];
	float4 p = pos[index];
	float4 unit = make_float4(1, 1, 1, 0);
	float sign = dot(n, unit); // 1 if n is positive and -1 if n negative
	float4 normN = n * sign;
	// ISSUE: signs do not work for all walls
	// if sign is -1 then we need < 0 comparison
	// if sign is 1 then we need a > 0 comparison
	// this way if distance > 0 we have successfully
	// detected wall penetration
	float distance = sign * (dot(p, normN) - wallPos + radius * sign);
	// this allows us to test multiple wall collisions
	// and pick the one that is the most important
	// ISSUE: how is contactDistance initialized?
	// dot(vel, n) must be positive so that the rigid body is heading
	// towards the wall
	// otherwise its velocity will carry it inside the scene's AABB
	if (distance > contactDistance[index] && dot(vel, n) > 0)
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
	float4 *linMom, // Input: rigid body linear momentum
	float4 *angMom, // Input: rigid body angular momentum
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

	//// compute linear and angular impulses using Baraff's method
	//float4 LinearImpulse = cn * impulse;
	//glm::vec3 AngularImpulse = oldInv * 
	//	(glm::cross(glm::vec3(r.x, r.y, r.z),
	//	glm::vec3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z)));

	//// apply impulses to rigid body
	//vel[index] += LinearImpulse / oldMass;
	//ang[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);
	// compute linear and angular impulses using Baraff's method

	float4 LinearImpulse = cn * impulse;
	float3 AngularImpulse = cross(make_float3(r), make_float3(LinearImpulse));

	// apply impulses to rigid body
	linMom[index] += LinearImpulse;
	angMom[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

}

__global__ void HandleWallCollisionKernelUnmapped(
	float radius, // Input: particle radius
	float4 *pos, // Input: rigid body position
	float4 *vel, // Input: rigid body velocity
	float4 *ang, // Input: rigid body angular velocity
	float4 *linMom, // Input: rigid body linear momentum
	float4 *angMom, // Input: rigid body angular momentum
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
	float4 oldDisp = p - oldCM;
	// move center of mass so that rigid body and wall no longer overlap
	float4 newCM = oldCM + wall - radius * n - pAuxil;

	// find new particle position
	float4 newP = newCM + oldDisp;
	newP.w = 1.f;
	float4 cn, cp;
	// find exact collision point
	FindExactContactPointKernel(newP, newP + n * radius, radius, 0, &cp, &cn);
	float4 r = newCM - cp;
	// computer impulse magnitude
	float impulse = ComputeImpulseMagnitudeKernel(oldVel, oldAng,
		r, oldInv, oldMass, cn);

	// compute linear and angular impulses using Baraff's method
	float4 LinearImpulse = cn * impulse;
	glm::vec3 AngularImpulse = oldInv *
		(glm::cross(glm::vec3(r.x, r.y, r.z),
		glm::vec3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z)));

	// correct rigid body position
	//pos[index] = newCM;

	// apply impulses to rigid body
	vel[index] += LinearImpulse / oldMass;
	ang[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);

	//// compute linear and angular impulses using Baraff's method
	//float4 LinearImpulse = cn * impulse;
	//float3 AngularImpulse = cross(make_float3(r.x, r.y, r.z), make_float3(LinearImpulse.x, LinearImpulse.y, LinearImpulse.z));

	//// apply impulses to rigid body
	//linMom[index] += LinearImpulse;
	//angMom[index] += make_float4(AngularImpulse.x, AngularImpulse.y, AngularImpulse.z, 0);
}

void findExactContactPointTest(
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

float computeImpulseMagnitudeTest(
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

struct TestNode
{
	int intKey;
	float floatKey;
	int value;
};

bool comparison(const TestNode& node1, const TestNode& node2)
{
	if (node1.intKey < node2.intKey) return true;
	if (node1.intKey == node2.intKey) return node1.floatKey < node2.floatKey;

	return false;
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
	float4 *rbLinMom, // rigid body linear momentum
	float4 *rbAngMom, // rigid body angular momentum
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
		rbVel,
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
		rbVel,
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
		rbVel,
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
	float4 unit = make_float4(1, 1, 1, 1);
	float sign = dot(n, unit); // 1 if n is positive and -1 if n negative
	n3 = make_float3(n) * sign;
	wallPos = dot(n3, minPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		rbVel,
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
	sign = dot(n, unit); // 1 if n is positive and -1 if n negative
	n3 = make_float3(n) * sign;
	wallPos = dot(n3, minPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		rbVel,
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
	sign = dot(n, unit); // 1 if n is positive and -1 if n negative
	n3 = make_float3(n) * sign;
	wallPos = dot(n3, minPos);

	DetectWallCollisionKernel << < gridDim, blockDim >> >(
		params.particleRadius,
		particlePos,
		rbIndices,
		rbVel,
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


	//float *distances_CPU = new float[numParticles];
	//float4 *positions_CPU = new float4[numParticles];
	//checkCudaErrors(cudaMemcpy(distances_CPU, contactDistance, sizeof(float) * numParticles, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(positions_CPU, particlePos, sizeof(float) * 4 * numParticles, cudaMemcpyDeviceToHost));
	//bool toPrint = false;
	//for (int i = 0; i < numParticles; i++)
	//{
	//	if (distances_CPU[i] > 0)
	//	{
	//		toPrint = true;
	//		break;
	//	}
	//}

	//Cub_Double_Sort(contactDistance, rigid_body_flags_GPU, &particleIndicesGPU, numParticles);
	//std::cout << "Pre-sort..." << std::endl;
	//int *debug_indices = new int[numParticles];
	//checkCudaErrors(cudaMemcpy(debug_indices, particleIndicesGPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));
	//if (toPrint)for (int i = numParticles - 50; i < numParticles; i++)
	//	std::cout << "Sorted particle index @: " << i << " is: " << debug_indices[i] << std::endl;

	Cub_Sort_By_Key_Wrapper<int, float>(&particleIndicesGPU, &contactDistance, numParticles);

	//std::cout << "After float-key sorting..." << std::endl;
	//checkCudaErrors(cudaMemcpy(debug_indices, particleIndicesGPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));
	//if(toPrint)for (int i = numParticles - 50; i < numParticles; i++)
	//	std::cout << "Sorted particle index @: " << i << " is: " << debug_indices[i] << std::endl;

	Cub_Sort_By_Key_Wrapper<int, float>(&rigid_body_flags_GPU, &contactDistance, numParticles);

	//std::cout << "After float-key sorting int keys..." << std::endl;
	//checkCudaErrors(cudaMemcpy(debug_indices, rigid_body_flags_GPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));
	//if (toPrint)for (int i = numParticles - 50; i < numParticles; i++)
	//	std::cout << "Sorted particle flag @: " << i << " is: " << debug_indices[i] << std::endl;

	Cub_Sort_By_Key_Wrapper<int, int>(&particleIndicesGPU, &rigid_body_flags_GPU, numParticles);

	//std::cout << "After int-key sorting..." << std::endl;
	//checkCudaErrors(cudaMemcpy(debug_indices, particleIndicesGPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));
	//if (toPrint)for (int i = numParticles - 50; i < numParticles; i++)
	//	std::cout << "Sorted particle index @: " << i << " is: " << debug_indices[i] << std::endl;

	//delete debug_indices;
	int *cum_sum_indices_CPU = new int[numRigidBodies];
	cum_sum_indices_CPU[0] = particlesPerRB[0];
	for (int index = 1; index < numRigidBodies; index++)
	{
		cum_sum_indices_CPU[index] = cum_sum_indices_CPU[index - 1] + particlesPerRB[index];
	}
	int *cum_sum_indices_GPU;
	checkCudaErrors(cudaMalloc((void**)&cum_sum_indices_GPU, sizeof(int) * numRigidBodies));
	checkCudaErrors(cudaMemcpy(cum_sum_indices_GPU, cum_sum_indices_CPU, sizeof(int) * numRigidBodies, cudaMemcpyHostToDevice));

	

	//std::vector<TestNode> test;
	//test.resize(numParticles);
	//for (int i = 0; i < numParticles; i++)
	//{
	//	test[i].value = i;
	//	test[i].intKey = rigid_body_flags_CPU[i];
	//	test[i].floatKey = distances_CPU[i];
	//}

	//std::sort(test.begin(), test.end(), comparison);

	//int *particleIndicesCPU = new int[numParticles];
	//checkCudaErrors(cudaMemcpy(particleIndicesCPU, particleIndicesGPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));


	//if (toPrint)
	//	for (int i = 0; i < numParticles; i++)
	//	{
	//		std::cout << "CPU Node: " << std::endl;
	//		std::cout << "Index: " << test[i].intKey << " ";
	//		std::cout << "Distance: " << test[i].floatKey << " ";
	//		std::cout << "Value: " << test[i].value << " ";
	//	}
	//for (int i = 0; i < numParticles; i++)
	//{
	//	if (test[i].value != particleIndicesCPU[i])
	//	{
	//		std::cout << "Problem @: " << i << std::endl;
	//		std::cout << "CPU Node: " << std::endl;
	//		std::cout << "Index: " << test[i].intKey << " ";
	//		std::cout << "Distance: " << test[i].floatKey << " ";
	//		std::cout << "Value: " << test[i].value << " "; 
	//		std::cout << std::endl;
	//		std::cout << "GPU Node: " << std::endl;
	//		std::cout << "Index: " << rigid_body_flags_CPU[particleIndicesCPU[i]] << " ";
	//		std::cout << "Distance: " << distances_CPU[particleIndicesCPU[i]] << " ";
	//		std::cout << "Value: " << particleIndicesCPU[i] << " ";
	//		std::cout << std::endl;
	//	}
	//}
	//delete particleIndicesCPU;

	//for (int i = 0; i < numParticles; i++)
	//{
	//	if (distances_CPU[i] > 0)
	//	{
	//		float4 p = positions_CPU[i];
	//		float r = params.particleRadius;

	//		float4 unit = make_float4(1, 1, 1, 1);
	//		float sign = dot(n, unit); // 1 if n is positive and -1 if n negative
	//		float4 normN = n * sign;
	//		float distance = sign * (dot(p, normN) - wallPos + r * sign);

	//		std::cout << "GPU distance: " << distances_CPU[i] << std::endl;
	//		std::cout << "CPU distance: " << distance << std::endl;
	//		std::cout << "Left wall distance: " << p.x - maxPos.x + r << std::endl;
	//		std::cout << "Top wall distance: " << p.y - maxPos.y + r << std::endl;
	//		std::cout << "Far wall distance: " << p.z - maxPos.z + r << std::endl;
	//		std::cout << "Right wall distance: " << -(p.x - minPos.x - r) << std::endl;
	//		std::cout << "Bottom wall distance: " << -(p.y - minPos.y - r) << std::endl;
	//		std::cout << "Near wall distance: " << -(p.z - minPos.z - r) << std::endl;

	//	}
	//}
	//float4 *RB_CPU_POS = new float4[numRigidBodies];
	//checkCudaErrors(cudaMemcpy(RB_CPU_POS, rbPos, sizeof(float4) * numRigidBodies, cudaMemcpyDeviceToHost));
	//float4 *RB_CPU_VEL = new float4[numRigidBodies];
	//checkCudaErrors(cudaMemcpy(RB_CPU_VEL, rbVel, sizeof(float4) * numRigidBodies, cudaMemcpyDeviceToHost));
	//float4 *RB_CPU_ANG = new float4[numRigidBodies];
	//checkCudaErrors(cudaMemcpy(RB_CPU_ANG, rbAng, sizeof(float4) * numRigidBodies, cudaMemcpyDeviceToHost));

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
		rbLinMom, // Input: rigid body linear momentum
		rbAngMom, // Input: rigid body angular momentum
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
	//if (toPrint)for (int i = 0; i < numParticles; i++)
	//	if (distances_CPU[i] > 0){
	//	std::cout << "GPU distance @: " << i << " is: " << distances_CPU[i] << std::endl;
	//	}

	//if (toPrint)
	//{
	//	float4 *RB_GPU_POS = new float4[numRigidBodies];
	//	checkCudaErrors(cudaMemcpy(RB_GPU_POS, rbPos, sizeof(float4) * numRigidBodies, cudaMemcpyDeviceToHost));
	//	float4 *RB_GPU_VEL = new float4[numRigidBodies];
	//	checkCudaErrors(cudaMemcpy(RB_GPU_VEL, rbVel, sizeof(float4) * numRigidBodies, cudaMemcpyDeviceToHost));
	//	float4 *RB_GPU_ANG = new float4[numRigidBodies];
	//	checkCudaErrors(cudaMemcpy(RB_GPU_ANG, rbAng, sizeof(float4) * numRigidBodies, cudaMemcpyDeviceToHost));

	//	glm::mat3 *invInertia = new glm::mat3[numRigidBodies]; 
	//	checkCudaErrors(cudaMemcpy(invInertia, Iinv, sizeof(glm::mat3) * numRigidBodies, cudaMemcpyDeviceToHost));
	//	
	//	float *mass = new float[numRigidBodies];
	//	checkCudaErrors(cudaMemcpy(mass, rbMass, sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));

	//	int *contactID = new int[numParticles];
	//	checkCudaErrors(cudaMemcpy(contactID, particleIndicesGPU, sizeof(int) * numParticles, cudaMemcpyDeviceToHost));
	//	int currentParticle = 0;
	//	//for (int i = 0; i < numParticles; i++)
	//		//std::cout << "Sorted particle index @: " << i << " is: " << contactID[i] << std::endl;

	//	for (int i = 0; i < numParticles; i++)
	//		if (distances_CPU[contactID[i]] > 0){
	//		std::cout << "GPU distance @ GPU index: " << contactID[i] << " is: " << distances_CPU[contactID[i]] << std::endl;
	//		}

	//	for (int index = 0; index < numRigidBodies; index++)
	//	{

	//		int contactIndex = currentParticle;
	//		float distance = distances_CPU[contactIndex];
	//		for (int particle = currentParticle + 1; particle < currentParticle + particlesPerRB[index]; particle++)
	//		{
	//			if (distances_CPU[particle] > distance)
	//			{
	//				contactIndex = particle;
	//				distance = distances_CPU[contactIndex];
	//			}
	//		}
	//		currentParticle += particlesPerRB[index];

	//		int contactIndex_GPU = contactID[cum_sum_indices_CPU[index] - 1];
	//		float distance_GPU = distances_CPU[contactIndex_GPU];


	//		if (distance <= 0)
	//			continue;

	//		if (contactIndex_GPU != contactIndex)
	//		{
	//			std::cout << "GPU and CPU results are different for rigid body No. " << index + 1 << std::endl;
	//			int gpuIndex = -1;
	//			for (int i = 0; i < numParticles; i++)
	//			{
	//				if (contactID[i] == contactIndex)
	//				{
	//					gpuIndex = i;
	//					break;
	//				}
	//			}
	//			std::cout << "CPU index found @: " << gpuIndex << " (" << contactID[gpuIndex] << ")" << std::endl;
	//		}

	//		std::cout << "Changing CM of rigid body No." << index + 1 << std::endl;
	//		std::cout << "CPU colliding particle: " << contactIndex << " and distance: " << distance << std::endl;
	//		std::cout << "GPU colliding particle: " << contactIndex_GPU << " and distance: " << distance_GPU << std::endl;
	//		std::cout << "GPU contact info index @: " << cum_sum_indices_CPU[index] - 1 << std::endl;
	//		float radius = params.particleRadius;
	//		
	//		float4 p = positions_CPU[contactIndex];
	//		float4 radiusAuxil = radius * n;
	//		float4 wallPosAuxil = n * dot(maxPos, make_float3(n));
	//		float4 particlePosAuxil = n * dot(p, n);

	//		float4 disp = p - RB_CPU_POS[index];

	//		RB_CPU_POS[index] += wallPosAuxil - radiusAuxil - particlePosAuxil; // move center of mass in the direction of the normal

	//		float4 newP = RB_CPU_POS[index] + disp;
	//		float4 cn, cp;
	//		findExactContactPointTest(newP, newP + n * radius, radius, 0, cp, cn);
	//		float4 r = RB_CPU_POS[index] - cp;
	//		float4 localforce = cn * computeImpulseMagnitudeTest(RB_CPU_VEL[index], RB_CPU_ANG[index], r, invInertia[index], mass[index], cn);
	//		RB_CPU_VEL[index] += localforce / mass[index];
	//		glm::vec3 torque = invInertia[index] * (glm::cross(glm::vec3(r.x, r.y, r.z), glm::vec3(localforce.x, localforce.y, localforce.z)));
	//		RB_CPU_ANG[index] += make_float4(torque.x, torque.y, torque.z, 0);

	//		std::cout << "Showing results for rigid body No." << index + 1 << std::endl;
	//		std::cout << "CPU CM: (" << RB_CPU_POS[index].x << ", " <<
	//			RB_CPU_POS[index].y << ", " << RB_CPU_POS[index].z <<
	//			", " << RB_CPU_POS[index].w << ")" << std::endl;

	//		std::cout << "GPU CM: (" << RB_GPU_POS[index].x << ", " <<
	//			RB_GPU_POS[index].y << ", " << RB_GPU_POS[index].z <<
	//			", " << RB_GPU_POS[index].w << ")" << std::endl;

	//		std::cout << "CPU VEL: (" << RB_CPU_VEL[index].x << ", " <<
	//			RB_CPU_VEL[index].y << ", " << RB_CPU_VEL[index].z <<
	//			", " << RB_CPU_VEL[index].w << ")" << std::endl;

	//		std::cout << "GPU VEL: (" << RB_GPU_VEL[index].x << ", " <<
	//			RB_GPU_VEL[index].y << ", " << RB_GPU_VEL[index].z <<
	//			", " << RB_GPU_VEL[index].w << ")" << std::endl;

	//		std::cout << "CPU ANG: (" << RB_CPU_ANG[index].x << ", " <<
	//			RB_CPU_ANG[index].y << ", " << RB_CPU_ANG[index].z <<
	//			", " << RB_CPU_ANG[index].w << ")" << std::endl;

	//		std::cout << "GPU ANG: (" << RB_GPU_ANG[index].x << ", " <<
	//			RB_GPU_ANG[index].y << ", " << RB_GPU_ANG[index].z <<
	//			", " << RB_GPU_ANG[index].w << ")" << std::endl;
	//	}
	//	delete contactID;
	//	delete RB_GPU_POS;
	//	delete RB_GPU_VEL;
	//	delete RB_GPU_ANG;
	//	delete mass;
	//	delete invInertia;
	//}
	//delete distances_CPU;
	//delete positions_CPU;

	//delete RB_CPU_POS;
	//delete RB_CPU_VEL;
	//delete RB_CPU_ANG;
	//
	// clean-up stray pointers
	if (particleIndicesGPU)
		checkCudaErrors(cudaFree(particleIndicesGPU));
	if (contactNormal)
		checkCudaErrors(cudaFree(contactNormal));
	if (contactPosition)
		checkCudaErrors(cudaFree(contactPosition));
	if (contactWallPosition)
		checkCudaErrors(cudaFree(contactWallPosition));
	if (contactDistance)
		checkCudaErrors(cudaFree(contactDistance));
	if (rigid_body_flags_GPU)
		checkCudaErrors(cudaFree(rigid_body_flags_GPU));
	if (rigid_body_flags_CPU)
		delete rigid_body_flags_CPU;
	if (cum_sum_indices_GPU)
		checkCudaErrors(cudaFree(cum_sum_indices_GPU));
	if (cum_sum_indices_CPU)
		delete cum_sum_indices_CPU;
	
}


