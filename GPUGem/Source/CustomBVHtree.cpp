#include "CustomBVHtree.h"


CustomBVHtree::CustomBVHtree()
{
}


CustomBVHtree::~CustomBVHtree()
{
}

Particle* CustomBVHtree::getRoot()
{
	return root;
}

Particle* CustomBVHtree::getLeaf(int id)
{
	return NULL;
}

Particle* CustomBVHtree::getLeftChild(Particle* node)
{
	return node->left;
}

Particle* CustomBVHtree::getRightChild(Particle* node)
{
	return node->right;
}

Particle* CustomBVHtree::getRightmostLeafInLeftSubtree(Particle* node)
{
	return node;
}

Particle* CustomBVHtree::getRightmostLeafInRightSubtree(Particle* node)
{
	return node;
}

bool CustomBVHtree::isLeaf(Particle* node)
{
	return node->isLeaf;
}

int CustomBVHtree::getNumOfLeaves()
{
	return numOfLeaves;
}

int CustomBVHtree::getObjectID(Particle* leaf)
{
	return leaf->id;
}

Particle CustomBVHtree::getParticle(Particle *leaf)
{
	return *leaf;
}