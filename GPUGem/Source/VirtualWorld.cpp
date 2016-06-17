#include "VirtualWorld.h"


VirtualWorld::VirtualWorld()
{
}


VirtualWorld::~VirtualWorld()
{
}

/**
Adds new object to object list. Also adds it to viewer's and engine's list.
*/
void VirtualWorld::addVirtualObject(std::shared_ptr<VirtualObject> obj)
{
	/*clock_t tCurrent = clock();
	if ((tCurrent - tLastSecond) >= CLOCKS_PER_SEC)
	{
		tLastSecond += CLOCKS_PER_SEC;
		iFPSCount = iCurrentFPS;
		iCurrentFPS = 0;
	}
	if (ptrRenderScene)ptrRenderScene(lpParam);
	iCurrentFPS++;*/
	virtualObjects.push_back(obj);
	viewer->addToDraw(&*obj);
	//engine->addRigidBody(obj);
}

// get functions
Viewer * VirtualWorld::getViewer(void)
{
	return viewer;
}

PhysicsEngine * VirtualWorld::getEngine(void)
{
	return engine;
}

// set functions
void VirtualWorld::setViewer(Viewer *v)
{
	viewer = v;
}

void VirtualWorld::setEngine(PhysicsEngine *pe)
{
	engine = pe;
}

void VirtualWorld::render(void)
{
	viewer->render();
}

void VirtualWorld::update(const double &dt)
{
	engine->update(dt);

	for (unsigned int i = 0; i < virtualObjects.size(); i++)
		virtualObjects[i]->update();
}

void VirtualWorld::resize(int w, int h)
{
	viewer->resize(w, h);
}