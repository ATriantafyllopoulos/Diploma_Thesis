#pragma once
#include "Primitives.h"

class CustomBVHtree
{
public:
	CustomBVHtree();
	~CustomBVHtree();

	Particle* getRoot();
	Particle* getLeaf(int id);
	Particle* getLeftChild(Particle* node);
	Particle* getRightChild(Particle* node);
	Particle* getRightmostLeafInLeftSubtree(Particle* node);
	Particle* getRightmostLeafInRightSubtree(Particle* node);
	
	Particle getParticle(Particle *leaf);

	bool isLeaf(Particle* node);

	int getNumOfLeaves();
	int getObjectID(Particle* leaf);
private:
	int numOfLeaves;
	Particle *root;
};

