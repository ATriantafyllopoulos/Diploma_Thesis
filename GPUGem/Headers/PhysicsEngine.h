#pragma once
#include "RigidBody.h"

#include <memory>
/**
Base (abstract) class responsible for updating objects and collision handling.
Derived class can be either a custom-made engine or the bullet library.
It will be accessed only by the virtual world class, and therefore needs
to contain ALL and ONLY those methods and variables necessary to that class.
Changed to abstact class.
Notes:
	- The update method is not used by the bulletphysics library. If it is not necessary for the custom physics engine as well it should be removed.
Update 08/06/2016: Currently not used or necessary.
*/
class PhysicsEngine
{
public:
	PhysicsEngine()
	{
	};
	virtual ~PhysicsEngine()
	{
	};

	virtual void update(const double &dt) = 0;
	virtual void addRigidBody(RigidBody *x) = 0;
};

