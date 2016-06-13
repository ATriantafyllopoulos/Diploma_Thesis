#include "cuda_runtime.h"

typedef struct particle
{
	particle *left;
	particle *right;
	particle *leftmost;
	particle *rightmost;
	particle *parent;

	//trying to save collision using indices
	int collisions[8];

	unsigned int collisionCounter; //this must not exceed 7

	float radius;

	float3 centroid;
	int id;
	bool isLeaf;

	//added inequality operator overload
	//might be useful when incorporating radix sort
	//remove otherwise
	bool operator != (particle &rhs) { return &*this == &rhs; }
} Particle;

//function overload also used during radix sort
//if not needed then remove
//int CoutCast(Particle val) { return 0; }


typedef struct collisionList
{
	void add(int, int);
}CollisionList;
