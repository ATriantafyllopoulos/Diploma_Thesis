#include "VirtualWorld.h"

VirtualWorld::VirtualWorld()
{
	timestep = 0.05f;
	damping = 1.0f;
	gravity = 0.0003f;
	iterations = 1;
	collideSpring = 0.5f;
	collideDamping = 0.02f;
	collideShear = 0.1f;
	collideAttraction = 0.0f;
	objectMode = M_BUNNY;
	viewMode = M_VIEW;
	runSimulation = false;
}

VirtualWorld::~VirtualWorld()
{
	delete psystem;
}

Viewer_GL3 * VirtualWorld::getViewer(void)
{
	return viewer;
}

void VirtualWorld::setViewer(Viewer_GL3 *v)
{
	viewer = v;
}

void VirtualWorld::render(void)
{
	viewer->setRendererVBO(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
	viewer->setNumberOfRangeData(psystem->getNumberOfRangeData());
	viewer->setRangeSampler(psystem->getRangeSampler());
	viewer->setRangeVAO(psystem->getRangeVAO());
	viewer->setRangeTexture(psystem->getRangeTexture());
	viewer->render();

}

void VirtualWorld::update()
{
	psystem->setIterations(iterations);
	psystem->setDamping(damping);
	psystem->setGravity(-gravity);
	psystem->setCollideSpring(collideSpring);
	psystem->setCollideDamping(collideDamping);
	psystem->setCollideShear(collideShear);
	psystem->setCollideAttraction(collideAttraction);
	if (runSimulation)
		psystem->update(timestep);

	if (viewMode == M_AR)
	{
		glm::mat4 newViewMatrix = psystem->getCameraTransformation();

		newViewMatrix = transpose(inverse(newViewMatrix));

		glm::quat q;
		q = quat_cast(newViewMatrix);
		glm::vec3 euler = glm::eulerAngles(q);// * 3.14159f / 180.f;
		glm::mat4 rotation = glm::yawPitchRoll(-euler.y, euler.x, -euler.z);
		for (int row = 0; row < 3; row++)
		{
			for (int col = 0; col < 3; col++)
				newViewMatrix[row][col] = rotation[row][col];
		}
		newViewMatrix[3][1] = -newViewMatrix[3][1];
		newViewMatrix[3][2] = -newViewMatrix[3][2];
		viewer->setViewMatrix(newViewMatrix);
	}
	viewer->setObjectNumber(psystem->getNumberOfObjects());
	viewer->setModelMatrixArray(psystem->getModelMatrixArray());
}

void VirtualWorld::resize(int w, int h)
{
	viewer->resize(w, h);
}

void VirtualWorld::setMode(const int &mode)
{
	viewMode = mode;
	viewer->setViewModeCommand(mode);
}

void VirtualWorld::initializeParticleSystem()
{

	if (psystem)
	{
		numberOfParticles = psystem->getNumParticles();
		gridSize = psystem->getGridSize();
		//psystem->reset(ParticleSystem::CONFIG_GRID);
	}
	
}

void VirtualWorld::addSphere(int x, int y)
{
	glm::vec4 viewPort = viewer->getViewport();
	float xPos = x;
	float yPos = viewPort.w - y;

	glm::vec3 win(xPos, yPos, viewer->getPixelDepth(x, int(yPos)));
	glm::vec3 worldSpaceCoordinates = glm::unProject(win, viewer->getViewMatrix(), viewer->getProjectionMatrix(), viewPort);
	glm::vec3 velocity = glm::vec3(0.f, -0.5f, 0.0f);
	worldSpaceCoordinates.y = 3.01f;
	//if (objectMode == M_BUNNY)
	//{
	//	/*psystem->addBunny(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 1.5f);
	//	viewer->increaseNumberOfObjects();
	//	viewer->addScaleFactor(1.5f);
	//	viewer->addObjectType(M_BUNNY);*/
	//	/*psystem->addBanana(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.f);
	//	viewer->increaseNumberOfObjects();
	//	viewer->addScaleFactor(0.02f);
	//	viewer->addObjectType(M_BANANA);*/
	//	psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.f, "banana");
	//	viewer->increaseNumberOfObjects();
	//	viewer->addScaleFactor(0.02f);
	//	viewer->addObjectType(M_BANANA);
	//}
	//else if (objectMode == M_TEAPOT)
	//{
	//	psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.f, "teapot");
	//	viewer->increaseNumberOfObjects();
	//	viewer->addScaleFactor(0.0002f);
	//	viewer->addObjectType(M_TEAPOT);
	//	/*psystem->addTeapot(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.0f);
	//	viewer->increaseNumberOfObjects();
	//	viewer->addScaleFactor(0.0002f);
	//	viewer->addObjectType(M_TEAPOT);*/
	//	//psystem->addNewSphere(1, worldSpaceCoordinates, velocity, 10, psystem->getParticleRadius()*2.0f);
	//}
	/*psystem->addBunny(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 1.5f);
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(1.5f);
	viewer->addObjectType(M_BUNNY);*/
	/*psystem->addTeapot(worldSpaceCoordinates, glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.0f);
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.0002f);
	viewer->addObjectType(M_TEAPOT);*/
	/*psystem->addObj(worldSpaceCoordinates, glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.5f, "banana");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.025f);
	viewer->addObjectType(M_BANANA);*/
	psystem->addObj(worldSpaceCoordinates, glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.5f, "cube");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.0025f);
	viewer->addObjectType(M_CUBE);
}

void VirtualWorld::throwSphere(int x, int y)
{
	glm::vec4 viewPort = viewer->getViewport();
	float xPos = x;
	float yPos = viewPort.w - y;

	glm::vec3 win(xPos, yPos, viewer->getPixelDepth(x, int(yPos)));
	glm::vec3 worldSpaceCoordinates = glm::unProject(win, viewer->getViewMatrix(), viewer->getProjectionMatrix(), viewPort);
	glm::vec3 velocity = glm::vec3(0.f, 0.0f, -0.3f);
	worldSpaceCoordinates.z = 0.01f;
	if (objectMode == M_BUNNY)
	{
		//		psystem->addTeapot(worldSpaceCoordinates, velocity);
		/*psystem->addBunny(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 1.5f);
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(1.5f);
		viewer->addObjectType(M_BUNNY);*/
		psystem->addBanana(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.f);
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(0.02f);
		viewer->addObjectType(M_BANANA);
		//		psystem->addNewSphere(1024, worldSpaceCoordinates, velocity, 10, psystem->getParticleRadius()*2.0f);
		//		psystem->addNewSphere(1, worldSpaceCoordinates, velocity, 10, psystem->getParticleRadius()*2.0f);
	}
	else if (objectMode == M_TEAPOT)
	{
		psystem->addTeapot(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.0f);
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(0.0002f);
		viewer->addObjectType(M_TEAPOT);
//		psystem->addTeapot(worldSpaceCoordinates, velocity);
		//psystem->addNewSphere(1, worldSpaceCoordinates, velocity, 10, psystem->getParticleRadius()*2.0f);
		//psystem->addRigidSphere(1024, worldSpaceCoordinates, velocity, 4, psystem->getParticleRadius()*2.0f);
	}
}

void VirtualWorld::setObjectMode(int x) 
{ 
	objectMode = x; 
}

void VirtualWorld::initDemoMode()
{
	std::srand(NULL);
	viewer->toggleShowRangeData(); //don't show range data
	psystem->toggleARcollisions(); //disable AR collisions
	psystem->setCollisionDetectionMethod(M_UNIFORM_GRID);
	//camera is static
//	viewMode = M_VIEW;viewer->viewModeCommand(M_VIEW);
	viewMode = M_VIEW; viewer->setViewModeCommand(M_VIEW);
	glm::vec3 vEye(0.0f, 0.0f, 3.1f);
	glm::vec3 vView(0.0f, 0.0f, -1.f);
	glm::vec3 vUp(0.0f, 1.0f, 0.0f);
	viewer->setViewMatrix(glm::lookAt(vEye, vView, vUp));
	psystem->setBBox(make_float3(-1, -0.8, -0.3), make_float3(1, 0.8, 1.3));
	//for (float x = -1; x < 1; x += 0.6)
	//	for (float y = -0.8; y < 0.8; y += 0.6)
	//		for (float z = 0.1; z < 0.9; z += 0.4)
	//		{
	//			glm::vec3 worldSpaceCoordinates(x, y, z);

	//			glm::vec3 velocity((float)std::rand() / (float)RAND_MAX / 10.f,
	//					(float)std::rand() / (float)RAND_MAX / 10.f,
	//					(float)std::rand() / (float)RAND_MAX / 10.f);
	//			//glm::vec3 velocity(0, 0, 0);
	//			//psystem->addBunny(worldSpaceCoordinates, glm::vec3(0, 0, 0), glm::vec3(0, 0.1, 0));
	//			psystem->addTeapot(worldSpaceCoordinates, velocity, glm::vec3(0, 0.0, 0), 2.0f);
	//			viewer->increaseNumberOfObjects();
	//			viewer->addScaleFactor(0.00020f);
	//			viewer->addObjectType(M_TEAPOT);
	//		}

	//// teapot 1
	//psystem->addTeapot(glm::vec3(0, 0.3, 0.0), glm::vec3(0, -0.1, 0.0), glm::vec3(0, 0.0, 0), 2.0f);
	//viewer->increaseNumberOfObjects();
	//viewer->addScaleFactor(0.00020f);
	//viewer->addObjectType(M_TEAPOT);

	//// teapot 2
	//psystem->addTeapot(glm::vec3(0, -0.3, 0.0), glm::vec3(0, 0, 0), glm::vec3(0, 0.0, 0), 2.0f);
	//viewer->increaseNumberOfObjects();
	//viewer->addScaleFactor(0.00020f);
	//viewer->addObjectType(M_TEAPOT);

	//// teapot 1
	//psystem->addBanana(glm::vec3(0.2, 0.0, 0.0), glm::vec3(0, 0.0, 0.0), glm::vec3(0, 0.0, 0), 2.0f);
	//viewer->increaseNumberOfObjects();
	//viewer->addScaleFactor(0.02f);
	//viewer->addObjectType(M_BANANA);

	//// teapot 2
	//psystem->addBanana(glm::vec3(-0.12, 0.0, 0.0), glm::vec3(0, 0, 0), glm::vec3(0, 0.0, 0), 2.0f);
	//viewer->increaseNumberOfObjects();
	//viewer->addScaleFactor(0.02f);
	//viewer->addObjectType(M_BANANA);

	// banana 1
	psystem->addBanana(glm::vec3(0.3, 0.15, 0.0), glm::vec3(0, -0.5, 0.0), glm::vec3(0, 0.0, 0), 2.0f);
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.02f);
	viewer->addObjectType(M_BANANA);

	// banana 2
	psystem->addBanana(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0, 0, 0), glm::vec3(0, 0.0, 0), 2.0f);
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.02f);
	viewer->addObjectType(M_BANANA);

	psystem->initCPU();
	psystem->setSceneAABB(make_float3(-1.5f, -1.f, -1.f), make_float3(1.f, 1.f, 1.f));
}

void VirtualWorld::DemoMode()
{
	
	psystem->setIterations(iterations);
	psystem->setDamping(damping);
	psystem->setGravity(-gravity);
	psystem->setCollideSpring(collideSpring);
	psystem->setCollideDamping(collideDamping);
	psystem->setCollideShear(collideShear);
	psystem->setCollideAttraction(collideAttraction);
	if (runSimulation)
		psystem->update(timestep);
		//psystem->updateCPU(timestep);

	viewer->setObjectNumber(psystem->getNumberOfObjects());
	viewer->setModelMatrixArray(psystem->getModelMatrixArray());

	viewer->setRendererVBO(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
	viewer->setNumberOfRangeData(psystem->getNumberOfRangeData());
	viewer->setRangeSampler(psystem->getRangeSampler());
	viewer->setRangeVAO(psystem->getRangeVAO());
	viewer->setRangeTexture(psystem->getRangeTexture());
	viewer->render();
}
