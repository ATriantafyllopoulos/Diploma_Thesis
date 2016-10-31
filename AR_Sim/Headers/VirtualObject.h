#ifndef VIRTUALOBJECT_H
#define VIRTUALOBJECT_H
#include "Renderable_GL3.h"
#include "RigidBody.h"

/**
Abstract class uniting the two base classes into one entity.
To be inherited by shape specific classes and to be used by a virtual world object.
*/

class VirtualObject :
	public Renderable_GL3, public RigidBody
{
public:
	VirtualObject(const double &m = 0, const glm::vec3 &p = glm::vec3(0, 0, 0), const glm::vec3 &v = glm::vec3(0, 0, 0), const glm::vec3 &i = glm::vec3(0, 0, 0)) :
        RigidBody(m, p, v, i) {}
    virtual ~VirtualObject(){}

private:
	//virtual void createRenderingContext(void) = 0;
	//virtual void createPhysicsContext(void) = 0;
};

#endif
