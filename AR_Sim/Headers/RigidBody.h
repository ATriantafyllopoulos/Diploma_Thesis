#pragma once

#include <glm/glm.hpp>
/**
Base class for rigid bodies. Contains all necessary variables and methods for a virtual world class.
This class should contain ALL and ONLY those methods and variables that the virtual world class needs.
It will be used to update each rigid bodies position and state vector during simulation.
It should be independent of the underlying physics engine (to be implemented as a derived class).
Its constructor should only be called by the main function, during program initialization or
after a user event (optional).
*/
class RigidBody
{
public:
	//complete set of constructors
	RigidBody(const double &m = 0, const glm::vec3 &p = glm::vec3(0, 0, 0), const glm::vec3 &v = glm::vec3(0, 0, 0), const glm::vec3 &i = glm::vec3(0, 0, 0));
	//RigidBody(const double &m = 0, const glm::vec3 &p = glm::vec3(0, 0, 0), const glm::vec3 &v = glm::vec3(0, 0, 0), const glm::vec3 &i = glm::vec3(0, 0, 0));

	//virtual destructor
    virtual ~RigidBody(){}
	
	//get functions
	double getMass(void);
	glm::vec3 getInertia(void);
	glm::vec3 getPosition(void);
	glm::vec3 getVelocity(void);

	//set functions
	void setMass(const double &m);
	void setInertia(const glm::vec3 &i);
	void setPosition(const glm::vec3 &p);
	void setVelocity(const glm::vec3 &v);

	//to be implemented by derived physics simulation class
	virtual void update(void) = 0;

protected:
	double mass; //rigid body's mass
	glm::vec3 inertia; //rigid body's local inertia vector

	glm::vec3 position; //rigid body's position
	glm::vec3 velocity; //rigid body's velocity
}; 

