#include "cuda_runtime.h"

typedef struct particle
{
	particle *left;
	particle *right;
	particle *leftmost;
	particle *rightmost;
	particle *parent;

	float radius;

	float3 centroid;
	int id;
	bool isLeaf;

} Particle;


typedef struct collisionList
{
	void add(int, int);
}CollisionList;
