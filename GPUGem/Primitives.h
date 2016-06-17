#include "cuda_runtime.h"


typedef struct particle
{
	particle *left;
	particle *right;
	//particle *leftmost;
	//particle *rightmost;
	particle *parent;

	//change range of leaves to indices
	//no need to fetch pointer form memory
	int leftmost;
	int rightmost;
	//trying to save collision using indices
	particle *collisions[8];

	unsigned int collisionCounter; //this must not exceed 7

	float radius;

	int id;
	bool isLeaf;

	float mass;
	//state vector
	float3 centroid;
	float3 linearMomentum;
	//no need for rotation yet
	//we use primitive spheres
	//float3 angularMomentum;
	//float4 quaternion;
	//added inequality operator overload
	//might be useful when incorporating radix sort
	//remove otherwise
	//bool operator != (particle &rhs) { return &*this == &rhs; }
} Particle;

//function overload also used during radix sort
//if not needed then remove
//int CoutCast(Particle val) { return 0; }


typedef struct collisionList
{
	void add(int, int);
}CollisionList;
