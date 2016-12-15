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
	/*psystem->setIterations(iterations);
	psystem->setDamping(damping);
	psystem->setGravity(-gravity);
	psystem->setCollideSpring(collideSpring);
	psystem->setCollideDamping(collideDamping);
	psystem->setCollideShear(collideShear);
	psystem->setCollideAttraction(collideAttraction);*/
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
	glm::vec3 velocity = glm::vec3(0.f, -0.1f, 0.0f);
	worldSpaceCoordinates.y = 0.2f;
	worldSpaceCoordinates.y = 0.7f;

	if (objectMode == M_BUNNY)
	{
		/*psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.0f, "banana");
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(0.020f);
		viewer->addObjectType(M_BANANA);*/
		/*psystem->addObj(worldSpaceCoordinates, glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.0f, "teapot");
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(0.0002f);
		viewer->addObjectType(M_TEAPOT);*/
		psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 1.0f, "cube");
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(0.0010f);
		viewer->addObjectType(M_CUBE);
	}
	else if (objectMode == M_TEAPOT)
	{
		psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.0f, "banana");
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(0.020f);
		viewer->addObjectType(M_BANANA);
		/*psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 1.5f, "bunny");
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(1.5f);
		viewer->addObjectType(M_BUNNY);*/
	}
	/*psystem->addTeapot(worldSpaceCoordinates, glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.0f);
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.0002f);
	viewer->addObjectType(M_TEAPOT);*/
	/*psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.0f, "banana");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.020f);
	viewer->addObjectType(M_BANANA);*/
	/*psystem->addObj(worldSpaceCoordinates, glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 1.0f, "cube");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.0010f);
	viewer->addObjectType(M_CUBE);*/
	/*psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 1.0f, "banana");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.010f);
	viewer->addObjectType(M_BANANA);*/
	/*psystem->addObj(worldSpaceCoordinates, glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.5f, "cube");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.0025f);
	viewer->addObjectType(M_CUBE);*/
}

void VirtualWorld::throwSphere(int x, int y)
{
	glm::vec4 viewPort = viewer->getViewport();
	float xPos = x;
	float yPos = viewPort.w - y;

	glm::vec3 win(xPos, yPos, viewer->getPixelDepth(x, int(yPos)));
	glm::vec3 worldSpaceCoordinates = glm::unProject(win, viewer->getViewMatrix(), viewer->getProjectionMatrix(), viewPort);
	glm::vec3 velocity = glm::vec3(0.f, 0.0f, -0.3f);
	worldSpaceCoordinates.y = 0.7;
	worldSpaceCoordinates.z = 0.01f;
	psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.0f, "teapot");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.0002f);
	viewer->addObjectType(M_TEAPOT);
//	if (objectMode == M_BUNNY)
//	{
//		//		psystem->addTeapot(worldSpaceCoordinates, velocity);
//		/*psystem->addBunny(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 1.5f);
//		viewer->increaseNumberOfObjects();
//		viewer->addScaleFactor(1.5f);
//		viewer->addObjectType(M_BUNNY);*/
//		psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.5f, "banana");
//		viewer->increaseNumberOfObjects();
//		viewer->addScaleFactor(0.025f);
//		viewer->addObjectType(M_BANANA);
//		//		psystem->addNewSphere(1024, worldSpaceCoordinates, velocity, 10, psystem->getParticleRadius()*2.0f);
//		//		psystem->addNewSphere(1, worldSpaceCoordinates, velocity, 10, psystem->getParticleRadius()*2.0f);
//	}
//	else if (objectMode == M_TEAPOT)
//	{
//		psystem->addTeapot(worldSpaceCoordinates, velocity, glm::vec3(0, 0, 0), 2.0f);
//		viewer->increaseNumberOfObjects();
//		viewer->addScaleFactor(0.0002f);
//		viewer->addObjectType(M_TEAPOT);
////		psystem->addTeapot(worldSpaceCoordinates, velocity);
//		//psystem->addNewSphere(1, worldSpaceCoordinates, velocity, 10, psystem->getParticleRadius()*2.0f);
//		//psystem->addRigidSphere(1024, worldSpaceCoordinates, velocity, 4, psystem->getParticleRadius()*2.0f);
//	}
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
	viewMode = M_VIEW; viewer->setViewModeCommand(M_VIEW);
	glm::vec3 vEye(0.0f, 0.0f, 3.1f);
	glm::vec3 vView(0.0f, 0.0f, -1.f);
	glm::vec3 vUp(0.0f, 1.0f, 0.0f);
	viewer->setViewMatrix(glm::lookAt(vEye, vView, vUp));
	Demo_TwoBananas();
	//Demo_TwoTeapots();
	//Demo_ThirtySixTeapots();
	//Demo_FiveHundredTeapots();
	//psystem->setBBox(make_float3(-1, -0.8, -0.3), make_float3(1, 0.8, 1.3));
	
	//for (float x = -1; x < 1; x += 0.6)
	//	for (float y = -0.8; y < 0.8; y += 0.6)
	//		for (float z = 0.1; z < 0.9; z += 0.4)
	///*psystem->setBBox(make_float3(-4.f, -4.f, -4.f), make_float3(4.f, 4.f, 4.f));
	//for (float x = -2; x < 2; x += 0.6)
	//	for (float y = -2; y < 2; y += 0.6)
	//		for (float z = 1; z < 3; z += 0.4)*/
	//		{
	//			glm::vec3 worldSpaceCoordinates(x, y, z);

	//			glm::vec3 velocity((float)std::rand() / (float)RAND_MAX / 10.f,
	//					(float)std::rand() / (float)RAND_MAX / 10.f,
	//					(float)std::rand() / (float)RAND_MAX / 10.f);
	//			//glm::vec3 velocity(0, 0, 0);
	//			//psystem->addBunny(worldSpaceCoordinates, glm::vec3(0, 0, 0), glm::vec3(0, 0.1, 0));
	//			psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0.0, 0), 2.0f, "teapot");
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

	
	std::cout << "Total number of rigid bodies: " << psystem->getNumberOfObjects() << std::endl;
	std::cout << "Total number of particles: " << psystem->getNumParticles() << std::endl;
	//psystem->initCPU();
	//
}

void VirtualWorld::Demo_TwoTeapots()
{
	//psystem->setSceneAABB(make_float3(-1.5f, -1.f, -1.f), make_float3(1.f, 1.f, 1.f));
	psystem->setSceneAABB(make_float3(-1000, -1000, -1000), make_float3(1000, 1000, 1000));
	// banana 1
	psystem->addObj(glm::vec3(0.4, 0.0, 0.0), glm::vec3(0, -0.0, 0), glm::vec3(0, 0.28, 0), 2.0f, "teapot");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.00020f);
	viewer->addObjectType(M_TEAPOT);

	// banana 2
	psystem->addObj(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 0), glm::vec3(0, 0, 0), 2.0f, "teapot");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.00020f);
	viewer->addObjectType(M_TEAPOT);
}

void VirtualWorld::Demo_TwoBananas()
{
	//psystem->setSceneAABB(make_float3(-1.5f, -1.f, -1.f), make_float3(1.f, 1.f, 1.f));
	psystem->setSceneAABB(make_float3(-1000, -1000, -1000), make_float3(1000, 1000, 1000));
	// banana 1
	psystem->addObj(glm::vec3(0.3, 0.15, 0.0), glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.5f, "banana");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.025f);
	viewer->addObjectType(M_BANANA);
	/*psystem->addObj(glm::vec3(0.3, 0.15, 0.0), glm::vec3(0, -0.5, 0), glm::vec3(0, 0, 0), 2.5f, "teapot");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.00025f);
	viewer->addObjectType(M_TEAPOT);*/
	// banana 2
	psystem->addObj(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0, -0.0, 0), glm::vec3(0, 0, 0), 2.5f, "banana");
	viewer->increaseNumberOfObjects();
	viewer->addScaleFactor(0.025f);
	viewer->addObjectType(M_BANANA);
}

void VirtualWorld::Demo_ThirtySixTeapots()
{
	//psystem->setSceneAABB(make_float3(-1.5f, -1.f, -1.f), make_float3(1.f, 1.f, 1.f));

	psystem->setSceneAABB(make_float3(-4.f, -4.f, -4.f), make_float3(4.f, 4.f, 4.f));
	//psystem->setSceneAABB(make_float3(-8.f, -8.f, -8.f), make_float3(8.f, 8.f, 8.f));
	/*for (float x = -4; x < 4.2; x += 1.2)
	for (float y = -4.8; y < 4.9; y += 1.2)
	for (float z = 0.1; z < 2.6; z += 0.8)*/

	//for (float x = -1; x < 1; x += 0.6)
	//for (float x = -2; x < 2; x += 0.6)
	for (float x = -3; x < 3; x += 0.6)
		//for (float y = -0.8; y < 0.8; y += 0.6)
		//for (float y = -1.8; y < 1.8; y += 0.6)
		//for (float y = -1.8; y < 2.8; y += 0.6)
		//for (float y = -2.8; y < 2.8; y += 0.6)
		for (float y = -3.8; y < 2.8; y += 0.6)
		for (float z = 0.1; z < 0.9; z += 0.4)
			//for (float z = 0.1; z < 1.4; z += 0.4)
			{
				glm::vec3 worldSpaceCoordinates(x, y, z);
				//glm::vec3 worldSpaceCoordinates(0, -0.5, -2);
				glm::vec3 velocity((float)std::rand() / (float)RAND_MAX / 10.f,
						(float)std::rand() / (float)RAND_MAX / 10.f,
						(float)std::rand() / (float)RAND_MAX / 10.f);
				//glm::vec3 velocity(0, 0, 0);
				//psystem->addBunny(worldSpaceCoordinates, glm::vec3(0, 0, 0), glm::vec3(0, 0.1, 0));
				psystem->addObj(worldSpaceCoordinates, glm::vec3(0, 0.0, 0), glm::vec3(0, 0.3, 0), 2.0f, "teapot");
				viewer->increaseNumberOfObjects();
				viewer->addScaleFactor(0.00020f);
				viewer->addObjectType(M_TEAPOT);
				//std::cout << "Virtual rigid bodies: " << psystem->getNumberOfObjects() << std::endl;
			}
	//for (float x = 0; x < 10; x += 1)
	//	for (float y = 0; y < 10; y += 1)
	//		for (float z = 0; z < 3; z += 1)
	//		{
	//			glm::vec3 worldSpaceCoordinates(x, y, z);

	//			glm::vec3 velocity((float)std::rand() / (float)RAND_MAX / 10.f,
	//					(float)std::rand() / (float)RAND_MAX / 10.f,
	//					(float)std::rand() / (float)RAND_MAX / 10.f);
	//			//glm::vec3 velocity(0, 0, 0);
	//			//psystem->addBunny(worldSpaceCoordinates, glm::vec3(0, 0, 0), glm::vec3(0, 0.1, 0));
	//			psystem->addObj(worldSpaceCoordinates, glm::vec3(0, 0.0, 0), glm::vec3(0, 0.3, 0), 2.0f, "teapot");
	//			viewer->increaseNumberOfObjects();
	//			viewer->addScaleFactor(0.00020f);
	//			viewer->addObjectType(M_TEAPOT);
	//			//std::cout << "Virtual rigid bodies: " << psystem->getNumberOfObjects() << std::endl;
	//		}

}

void VirtualWorld::Demo_FiveHundredTeapots()
{
	psystem->setBBox(make_float3(-4.f, -3.f, -4.f), make_float3(4.f, 4.f, 4.f));
	for (float x = -2; x < 2; x += 0.6)
		for (float y = -2; y < 2; y += 0.6)
			for (float z = 1; z < 3; z += 0.4)
			{
		glm::vec3 worldSpaceCoordinates(x, y, z);

		glm::vec3 velocity((float)std::rand() / (float)RAND_MAX / 10.f,
			(float)std::rand() / (float)RAND_MAX / 10.f,
			(float)std::rand() / (float)RAND_MAX / 10.f);
		//glm::vec3 velocity(0, 0, 0);
		//psystem->addBunny(worldSpaceCoordinates, glm::vec3(0, 0, 0), glm::vec3(0, 0.1, 0));
		psystem->addObj(worldSpaceCoordinates, velocity, glm::vec3(0, 0.0, 0), 2.0f, "teapot");
		viewer->increaseNumberOfObjects();
		viewer->addScaleFactor(0.00020f);
		viewer->addObjectType(M_TEAPOT);
			}
	std::cout << "Total number of rigid bodies: " << psystem->getNumberOfObjects() << std::endl;
}

void VirtualWorld::DemoMode()
{
	
	/*psystem->setIterations(iterations);
	psystem->setDamping(damping);
	psystem->setGravity(-gravity);
	psystem->setCollideSpring(collideSpring);
	psystem->setCollideDamping(collideDamping);
	psystem->setCollideShear(collideShear);
	psystem->setCollideAttraction(collideAttraction);*/
	static int iterations = 0;
	static float totalTime = 0;
	const int iterationLimit = 1000;
	if (runSimulation)
	{
		clock_t start = clock();
		psystem->update(timestep);
		iterations++;
		clock_t end = clock();
		float localTime = (end - start) / (CLOCKS_PER_SEC / 1000); //time difference in milliseconds
		totalTime += localTime;
		/*if (iterations < iterationLimit)
		{
			std::ofstream file("profiling.txt", std::ofstream::app);
			file << localTime << std::endl;
			file.close();
		}*/
	}
	
	if (iterations == iterationLimit)
	{
		std::cout << "Profiling stopped" << std::endl;
		std::cout << "Avg time spent on update for " << psystem->getNumberOfObjects() << " rigid bodies: " << totalTime / (float)iterations << std::endl;
	}
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

void VirtualWorld::Virtual_Benchmark()
{
	viewer->toggleShowRangeData(); //don't show range data
	psystem->toggleARcollisions(); //disable AR collisions
	psystem->setSceneAABB(make_float3(-4.f, -4.f, -4.f), make_float3(4.f, 4.f, 4.f));

	const int iteration_limit = 1000;
	const float xStarts[] = { -1, -2, -3};
	const float xStops[] = { 1, 2, 3};
	const float yStarts[] = { -0.8, -1.8, -1.8, -2.8 };
	const float yStops[] = { 0.8, 1.8, 2.8, 2.8 };
	const float zStarts[] = { 0.1};
	const float zStops[] = { 0.9};
	const int xPairs = 3, yPairs = 4, zPairs = 1;
	const float xStep = 0.6, yStep = 0.6, zStep = 0.4;
	for (int i = 0; i < xPairs; i++)
	{
		for (int j = 0; j < yPairs; j++)
		{
			for (int k = 0; k < zPairs; k++)
			{
				const float xStart = xStarts[0], xStop = xStops[0];
				const float yStart = yStarts[0], yStop = yStops[0];
				const float zStart = zStarts[0], zStop = zStops[0];
				/*std::cout << "x = [" << xStart << ", " << xStop << "]" << std::endl;
				std::cout << "y = [" << yStart << ", " << yStop << "]" << std::endl;
				std::cout << "z = [" << zStart << ", " << zStop << "]" << std::endl;*/
				for (float x = xStart; x < xStop; x += xStep)
				{
					for (float y = yStart; y < yStop; y += yStep)
					{
						for (float z = zStart; z < zStop; z += zStep)
						{
							glm::vec3 worldSpaceCoordinates(x, y, z);
							psystem->addObj(worldSpaceCoordinates, glm::vec3(0, 0.0, 0), glm::vec3(0, 0.3, 0), 2.0f, "teapot");
							/*viewer->increaseNumberOfObjects();
							viewer->addScaleFactor(0.00020f);
							viewer->addObjectType(M_TEAPOT);*/
						}
					}
				}
				/*std::cout << "Total number of rigid bodies: " << psystem->getNumberOfObjects() << std::endl;
				std::cout << "Total number of particles: " << psystem->getNumParticles() << std::endl;
				std::cout << std::endl;*/
				psystem->Empty_Particle_System();
				/*std::cout << "Total number of rigid bodies: " << psystem->getNumberOfObjects() << std::endl;
				std::cout << "Total number of particles: " << psystem->getNumParticles() << std::endl;
				std::cout << std::endl;*/
			}
		}
	}

	float dummy;
	std::cin >> dummy;
}
