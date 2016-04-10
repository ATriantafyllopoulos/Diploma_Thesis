#include "RigidBody.h"

/**
Complete constructor
Create object with m mass
Initial position : p
Initial velocity : v
Inertia : i
*/
RigidBody::RigidBody(const double &m, const glm::vec3 &p, const glm::vec3 &v, const glm::vec3 &i)
{
	mass = m;
	position = p;
	velocity = v;
	inertia = i;
}



RigidBody::~RigidBody()
{
}

// get functions
double RigidBody::getMass(void)
{
	return mass;
}

glm::vec3 RigidBody::getInertia(void)
{
	return inertia;
}


glm::vec3 RigidBody::getPosition(void)
{
	return position;
}


glm::vec3 RigidBody::getVelocity(void)
{
	return velocity;
}

// set functions
void RigidBody::setMass(const double &m)
{
	mass = m;
}


void RigidBody::setInertia(const glm::vec3 &i)
{
	inertia = i;
}


void RigidBody::setPosition(const glm::vec3 &p)
{
	position = p;
}


void RigidBody::setVelocity(const glm::vec3 &v)
{
	velocity = v;
}

