#include "BVHAuxiliary.cuh"
/*
* Algorithm is not changed by switch to AoS architecure. The only difference is the storage of the results.
* They are now stored in distinct arrays, instead of inside a common structure. This should increase memory efficiency,
* as threads within a warp should access neighboring addresses.
*/
__global__ void kernelConstructRadixTreeSoA(int numberOfInternalNodes,
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	unsigned int *sortedMortonCodes)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int internalOffset = numberOfInternalNodes + 1;
	if (i >= numberOfInternalNodes) return;

	//Run radix tree construction algorithm
	//Determine direction of the range (+1 or -1)
	int dPrev = longestCommonPrefix(i, i - 1, internalOffset, sortedMortonCodes);
	int dNext = longestCommonPrefix(i, i + 1, internalOffset, sortedMortonCodes);
	int d = dNext - dPrev > 0 ? 1 : -1;

	// Compute upper bound for the length of the range
	int sigMin = longestCommonPrefix(i, i - d, internalOffset, sortedMortonCodes);
	int lmax = 2;

	while (longestCommonPrefix(i, i + lmax * d, internalOffset, sortedMortonCodes) > sigMin) {
		lmax *= 2;
	}

	// Find the other end using binary search
	int l = 0;
	int t = lmax / 2;
	for (t = lmax / 2; t >= 1; t /= 2)
	{
		if (longestCommonPrefix(i, i + (l + t) * d, internalOffset, sortedMortonCodes) > sigMin)
			l += t;
		if (t == 1)	break;
	}

	int j = i + l * d;

	// Find the split position using binary search
	int sigNode = longestCommonPrefix(i, j, internalOffset, sortedMortonCodes);

	int s = 0;
	double div;
	double t2;
	for (div = 2; t >= 1; div *= 2)
	{
		t2 = __int2double_rn(l) / div;
		t = __double2int_ru(t2);
		int temp = longestCommonPrefix(i, i + (s + t) * d, internalOffset, sortedMortonCodes);
		if (temp > sigNode)
		{
			s = s + t;
		}
		if (t == 1)	break;
	}

	int gamma = i + s * d + min(d, 0);

	//internal nodes are stored right after leaf nodes in our structure
	//this means they have an offset of internalOffset = N
	int SoAindex = i + internalOffset;

	isLeaf[SoAindex] = false; //current node is no leaf

	int minRangeLoc = min(i, j);
	int maxRangeLoc = max(i, j);
	if (minRangeLoc == gamma) {
		//left child is a leaf
		//left child index is gamma
		leftIndices[SoAindex] = gamma; //left child is gamma
		parentIndices[gamma] = SoAindex; //parent of gamma leaf is this internal node
	}
	else {
		//left child is an internal node so we add the proper offset
		leftIndices[SoAindex] = gamma + internalOffset;
		parentIndices[gamma + internalOffset] = SoAindex;
	}

	if (maxRangeLoc == gamma + 1) {
		//right child is a leaf with index gamma + 1
		rightIndices[SoAindex] = gamma + 1; //right child is gamma + 1
		parentIndices[gamma + 1] = SoAindex; //parent of (gamma + 1) leaf is this internal node
	}
	else {
		//right child is an internal node so we add the proper offset
		rightIndices[SoAindex] = gamma + 1 + internalOffset;
		parentIndices[gamma + 1 + internalOffset] = SoAindex;
	}
	minRange[SoAindex] = minRangeLoc;
	maxRange[SoAindex] = maxRangeLoc;
}

/**
* BVH Construction kernel
* Algorithm described in karras2012 paper (bottom-up approach).
* First part only - leaf node construction.
* Utilizes Structure of Arrays architecture.
* Since this function only accesses leaf nodes no offset is needed.
*/
__global__ void kernelConstructLeafNodesSoA(int numberOfLeaves,
	bool *isLeaf, //array containing a flag to indicate whether node is leaf
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	int *minRange, //array containing minimum (sorted) leaf covered by each node
	int *maxRange, //array containing maximum (sorted) leaf covered by each node
	float4 *CMs, //array containing centers of mass for each leaf
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *sortedIndices, //array containing corresponding unsorted indices for each leaf
	float *radii, //radii of all nodes - currently the same for all particles
	float4 *positions, //original positions
	float particleRadius) //common radius parameter
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //each thread refers to leaf node #(index)
	if (index >= numberOfLeaves) 
		return;
	
	isLeaf[index] = true; //current is leaf

	//corresponding index in original array of positions - used to access actual particle parameters
	int originalIndex = sortedIndices[index];

	float4 cm = FETCH(positions, originalIndex); //fetch only once
	CMs[index] = cm;
	
	minRange[index] = index;
	maxRange[index] = index;

	initBound(bounds + index, cm, radii[index]); //initialize associated bound
}

/**
* BVH Construction kernel
* Algorithm described in karras2012 paper (bottom-up approach).
* Second part - internal node creation
* Utilizes Structure of Arrays architecture
* Manipulates internal node so appropriate offset (N) is needed.
*/

__global__ void kernelConstructInternalNodesSoA(int numberOfLeaves,
	int *leftIndices, //array containing indices of the left children of each node
	int *rightIndices, //array containing indices of the right children of each node
	int *parentIndices, //array containing indices of the parent of each node
	AABB *bounds, //array containing bounding volume for each node - currently templated Array of Structures
	int *nodeCounter) //used by atomic operations - ensures that each 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numberOfLeaves) return;

	int SoAindex = parentIndices[index]; //each leaf accesses its parent
	int currentIndex = SoAindex - numberOfLeaves; //subtract offset to calculate
	int res = atomicAdd(nodeCounter + currentIndex, 1);

	// Go up and handle internal nodes
	
	while (1) {
		if (res == 0)return; //if accesing the node for the first time return - this ensures we access each node once
		uint leftChildIndex = leftIndices[SoAindex]; //load left child's index once
		uint rightChildIndex = rightIndices[SoAindex]; //load right child's index once
		mergeBounds(&bounds[SoAindex], bounds[leftChildIndex], bounds[rightChildIndex]);

		//if this node is the root return - root node is the first of the internal nodes located @ position No.(N)
		if (SoAindex == numberOfLeaves)return; 
		SoAindex = parentIndices[SoAindex]; //if not finished retrieve parent
		currentIndex = SoAindex - numberOfLeaves;
		res = atomicAdd(nodeCounter + currentIndex, 1);
	}
}

/*
* Collision detection and handling routine.
* Iteratively traverses the BVH tree starting from the root and identifies all collisions.
* Each threads handles one particle.
* Collisions are currently reported twice, once for each respective particle.
* Each particle is handled separately.
* TODO: Find a way to report each collision once - missing appropriate structure.
* Utilizes Structure of Arrays
* Particles that collide turn red. Otherwise they turn blue.
* Perhaps global loads can be substituted by shared memory loads - at least for the root.
*/

__global__
void collideBVHSoA(float4 *color, //particle's color, only used for testing purposes
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
	//AT: Is 64 the correct size to use?
	int stack[64]; //using stack of indices
	int* stackPtr = stack;
	*stackPtr++ = -1; //push -1 at beginning so that thread will return when stack is empty 
	int numCollisions = 0; //count number of collisions
	//Traverse nodes starting from the root.
	//load leaf positions once - better to use this array as successive threads will access successive memory positions
	float4 queryPos = CMs[index]; 
	float queryRad = radii[index]; //load particle radius once - currently all particles have the same radius
	int SoAindex = numParticles; //start at root
	AABB queryBound = bounds[index]; //load particle's bounding volume once
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
		if (overlapL && isLeftLeaf)
		{
			
			force += collideSpheresBVH(queryPos, CMs[leftChildIndex],
				queryVel, make_float3(FETCH(vel, sortedIndices[leftChildIndex])),
				queryRad, radii[leftChildIndex],
				params);

			if (length(make_float3(CMs[leftChildIndex] - queryPos)) < queryRad + radii[leftChildIndex])
			{
				numCollisions++;
				color[queryIndex] = make_float4(1, 0, 0, 0);
			}
			
		}


		if (overlapR && isRightLeaf)
		{
			force += collideSpheresBVH(queryPos, CMs[rightChildIndex],
				queryVel, make_float3(FETCH(vel, sortedIndices[rightChildIndex])),
				queryRad, radii[rightChildIndex],
				params);

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
		color[index] = make_float4(0, 0, 1, 0);
	// collide with cursor sphere
	float4 colPos = make_float4(params.colliderPos.x, params.colliderPos.y, params.colliderPos.z, 1);
	if (length(queryPos - colPos) <= queryRad + params.colliderRadius)
		force += collideSpheresBVH(queryPos,
		colPos,
		queryVel,
		make_float3(0.0f, 0.0f, 0.0f),
		queryRad,
		params.colliderRadius,
		params);
	vel[queryIndex] = make_float4(queryVel + force, 0);
}

__global__
void staticCollideBVHSoA(float4 *positions, //virtual particle positions
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
SimParams params //simulation parameters
)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //handling particle #(index)
	if (index >= numParticles) return;

	float3 force = make_float3(0.0f);

	int queryIndex = index; //testing for virtual particles
	float3 queryVel = make_float3(FETCH(vel, queryIndex)); //load particle original velocity once

	//AT: Is 64 the correct size to use?
	int stack[64]; //using stack of indices
	int* stackPtr = stack;
	*stackPtr++ = -1; //push -1 at beginning so that thread will return when stack is empty 
	//Traverse nodes starting from the root.
	//load leaf positions once
	float4 queryPos = positions[index];
	float queryRad = params.particleRadius; //load particle radius once - currently all particles have the same radius
	int SoAindex = numRangeData; //start at root
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
			force += collideSpheresBVH(queryPos, CMs[leftChildIndex],
				queryVel, make_float3(0, 0, 0),
				queryRad, radii[leftChildIndex],
				params);
		}
		if (overlapR && isRightLeaf)
		{
			//force += reflect(queryVel, make_float3(normals[sortedIndices[rightChildIndex]])) * params.boundaryDamping / 10;
			force += collideSpheresBVH(queryPos, CMs[rightChildIndex],
				queryVel, make_float3(0, 0, 0),
				queryRad, radii[rightChildIndex],
				params);
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

	vel[queryIndex] = make_float4(queryVel + force, 0);
}