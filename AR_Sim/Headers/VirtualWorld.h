//#pragma once

#include "Viewer_GL3.h"
#include <vector>
#include "particleSystem.h"
#include <glm/gtx/euler_angles.hpp>
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
	Viewer_GL3 *getViewer(void);
	//PhysicsEngine *getEngine(void);

	// set functions
	void setViewer(Viewer_GL3 *v);

    //void addVirtualObject(std::shared_ptr<VirtualObject> obj);
	void render(void);
	void update();
	void resize(int w, int h);
	void setMode(const int &mode);

	void initializeParticleSystem();

	int getNumberOfVirtualParticles() { return psystem->getNumParticles(); }

	void addParticleSystem(ParticleSystem *x){ psystem = x; }

	void addSphere(int x, int y);
	void throwSphere(int x, int y);

	void toggleARcollisions(void){ psystem->toggleARcollisions(); }

	void setObjectMode(int x);
	void togglePauseFrame() {psystem->togglePauseFrame();}
	void setCollisionDetectionMethod(int x) { psystem->setCollisionDetectionMethod(x); }
	void toggleShowRangeData(){ viewer->toggleShowRangeData(); }
	void increaseARrestitution(){ psystem->increaseARrestitution();}
	void decreaseARrestitution(){ psystem->decreaseARrestitution();}
	void initDemoMode();
	void DemoMode();
private:
	int objectMode;
	int numberOfParticles;
	uint3 gridSize;
	bool useOpenGL;
	ParticleSystem *psystem;

	// simulation parameters
	float timestep;
	float damping;
	float gravity;
	int iterations;
	float collideSpring;
	float collideDamping;
	float collideShear;
	float collideAttraction;
	int viewMode;
	Viewer_GL3 *viewer;
	//PhysicsEngine *engine;
    //std::vector<std::shared_ptr<VirtualObject>> virtualObjects;

	
};

