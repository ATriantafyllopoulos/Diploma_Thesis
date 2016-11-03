#include "WindowsHandler.h"

//global static instance used to reference member function as glfw callbacks
static WindowsHandler *callBackInstance;
WindowsHandler::WindowsHandler() : WindowsHandler::WindowsHandler("Default Name", 640, 480)
{
	fpsCount = 0;
	fpsLimit = 1;
	title = "";
	width = 0;
	height = 0;
	viewMode = M_VIEW;
	objectMode = M_POINT_SPRITE;
	collisionMethod = M_BVH;
	virtualDemo = false;
}

WindowsHandler::WindowsHandler(std::string inTitle, int inWidth, int inHeight)
{
	callBackInstance = this;
    fpsCount = 0;
    fpsLimit = 1;
    title = inTitle;
    width = inWidth;
    height = inHeight;
    createWindow();

    glfwSetMouseButtonCallback(window, onMouseClickPure);
    glfwSetKeyCallback(window, keyCallbackPure);

    viewMode = M_AR;
    objectMode = M_BUNNY;
    collisionMethod = M_BVH;
    PrintMainMenu();
    sdkCreateTimer(&timer);

}


void WindowsHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

	if (key == GLFW_KEY_C && action == GLFW_PRESS)
	{
		world->toggleARcollisions();
	}
	else if (key == GLFW_KEY_V && action == GLFW_PRESS)
	{
		if(viewMode == M_AR)
		{
			std::cout << "Changing to free moving view mode." << std::endl;
			viewMode = M_VIEW;
			world->setMode(M_VIEW);
		}
		else if(viewMode == M_VIEW)
		{
			std::cout << "Changing to AR view mode." << std::endl;
			viewMode = M_AR;
			world->setMode(M_AR);
		}
	}
	else if (key == GLFW_KEY_O && action == GLFW_PRESS)
	{
		if(objectMode == M_BUNNY)
		{
			std::cout << "Changing to throwing teapots." << std::endl;
			objectMode = M_TEAPOT;
			world->setObjectMode(M_TEAPOT);
		}
		else if(objectMode == M_TEAPOT)
		{
			std::cout << "Changing to throwing bunnies." << std::endl;
			objectMode = M_BUNNY;
			world->setObjectMode(M_BUNNY);
		}
	}
	else if (key == GLFW_KEY_M && action == GLFW_PRESS)
	{
		if(collisionMethod == M_BVH)
		{
			std::cout << "Changing to collision detection using uniform grid." << std::endl;
			collisionMethod = M_UNIFORM_GRID;
			world->setCollisionDetectionMethod(M_UNIFORM_GRID);
		}
		else if(collisionMethod == M_UNIFORM_GRID)
		{
			std::cout << "Changing to collision detection using BVH." << std::endl;
			collisionMethod = M_BVH;
			world->setCollisionDetectionMethod(M_BVH);
		}
	}
	else if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS)
	{
		world->increaseARrestitution();
	}
	else if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS)
	{
		world->decreaseARrestitution();
	}
	else if (key == GLFW_KEY_R && action == GLFW_PRESS)
	{
		world->toggleShowRangeData();
	}
	else if (key == GLFW_KEY_E && action == GLFW_PRESS)
	{
		virtualDemo = true;
	}
	else if (key == GLFW_KEY_P && action == GLFW_PRESS)
	{
		//world->togglePauseFrame();
		world->toggleSimulation();
	}
	else if (key == GLFW_KEY_H && action == GLFW_PRESS)
	{
		PrintMainMenu();
	}
	else if (key == GLFW_KEY_F && action == GLFW_PRESS)
	{
		world->toggleSimulation();
		world->DemoMode();
		world->toggleSimulation();
	}
}

void WindowsHandler::keyCallbackPure(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if(callBackInstance)
		callBackInstance->keyCallback(window, key, scancode, action, mods);
}

void WindowsHandler::onMouseClick(GLFWwindow* win, int button, int action, int mods)
{

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
    	world->addSphere(xpos, ypos);
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
    {
    	world->throwSphere(xpos, ypos);
    }

}

void WindowsHandler::onMouseClickPure(GLFWwindow* win, int button, int action, int mods)
{
    if(callBackInstance)
    	callBackInstance->onMouseClick(win, button, action, mods);
}


WindowsHandler::~WindowsHandler()
{

	sdkDeleteTimer(&timer);
    glfwTerminate();
}

void WindowsHandler::PrintMainMenu()
{
	std::cout << "Current version only supports keyboard menu." << std::endl;
	std::cout << "Press C to toggle AR collisions ON and OFF." << std::endl;
	std::cout << "Press V to toggle between a free moving and an AR view mode." << std::endl;
	std::cout << "Press M to toggle between collision detection methods (BVH and Uniform Grid)." << std::endl;
	std::cout << "Press O to toggle between objects (Stanford Bunny and Teapot)." << std::endl;
	std::cout << "Press P pause in a current frame or continue video." << std::endl;
	std::cout << "Press R to toggle visualization of range data." << std::endl;
	std::cout << "Press H at any time to display these instructions again." << std::endl;
}

void WindowsHandler::createWindow()
{
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
    }
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE,GL_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow( width, height, title.c_str(), NULL, NULL);
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE; //necessary because system is unsupported
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 0);
	
	if (!window)
	{
		std::cerr << "Error occured during glfw initialization" << std::endl;
	}
    // Dark blue background
   // glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
}

void WindowsHandler::Run()
{
    do{
    	sdkStartTimer(&timer);
    	world->update();
    	world->render();
    	sdkStopTimer(&timer);
    	computeFPS();
    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );
}

void WindowsHandler::Demo()
{
	do{

		sdkStartTimer(&timer);
		world->DemoMode();
		sdkStopTimer(&timer);
		computeFPS();
	} // Check if the ESC key was pressed or the window was closed
	while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
			glfwWindowShouldClose(window) == 0 );
}
void WindowsHandler::computeFPS()
{

	fpsCount++;


	if (fpsCount == fpsLimit)
	{

		float ifps = 1000.f / sdkGetAverageTimerValue(&timer);
		fpsCount = 0;

		fpsLimit = (int)MAX(ifps, 1.f);
		std::string title = "Simulation FPS: ";
		std::ostringstream ss;
		ss << ifps;
		std::string s(ss.str());
		title += s;
		glfwSetWindowTitle(window, title.c_str());
		sdkResetTimer(&timer);
	}

}

/**
this function is called by the viewer's constructor
to which an instance of the WindowsHandler class is passed as a paremeter
should be called in main immediately after viewer is created
and definitely before any call to ShowWindow
*/
void WindowsHandler::setWorld(VirtualWorld *w)
{
    world = w;
}



