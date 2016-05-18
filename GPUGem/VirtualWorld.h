#pragma once

#include "Viewer.h"
#include "VirtualObject.h"
#include "PhysicsEngine.h"
#include <vector>
/**
Base class containing all necessary initialization and update routines.
It will serve as an interface between the platform API and the underlying physics and graphics engine.
*/
class VirtualWorld
{
public:
	VirtualWorld();
	~VirtualWorld();

	// get functions
	Viewer *getViewer(void);
	PhysicsEngine *getEngine(void);

	// set functions
	void setViewer(Viewer *v);
	void setEngine(PhysicsEngine *pe);

	void addVirtualObject(std::shared_ptr<VirtualObject> obj);
	void render(void);
	void update(const double &dt = 0.001);
	void resize(int w, int h);
private:
	Viewer *viewer;
	PhysicsEngine *engine;
	std::vector<std::shared_ptr<VirtualObject>> virtualObjects;
};

