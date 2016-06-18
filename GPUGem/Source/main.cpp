#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <Windows.h>
#include "WindowsHandler.h"
#include "Viewer_GL3.h"
#include "objModel.h"

bool modelCreation(VirtualWorld &world);

int main()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	HINSTANCE hInstance = (HINSTANCE)GetModuleHandle(NULL); //get process ID
	int nCmdShow = 5; //set to show window
	WindowsHandler GLwin(hInstance, 640, 480, L"Renderer"); //create rendering window
	HWND GLwindow = GLwin.returnHandler(); //get rendering window handler
	VirtualWorld virtualWorld;
	Viewer_GL3 viewer(GLwindow); //create renderer
	virtualWorld.setViewer(&viewer);
	//PhysicsEngine engine;
	/*std::cout << "Setting up models..." << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	if (modelCreation(virtualWorld)) //setup world models
		std::cout << "Models were succesfully set up." << std::endl;
	else
		std::cout << "Error! Models were not set up." << std::endl;*/

	
	//virtualWorld.setEngine(&engine);
	/*VERY IMPORTANT CALL*/
	GLwin.setWorld(&virtualWorld); //set rendering window's virtual world

	ShowWindow(GLwindow, nCmdShow); //show window
	ShowCursor(FALSE);

	GLwin.Run(); //static_cast<int>( msg.wParam );

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 1;
}

/**
This function is responsible for creating and loading the correct number of models on the screen
*/
bool modelCreation(VirtualWorld &world)
{
	//create houses
	std::cout << "Loading house model..." << std::endl;
	CAssimpModel house;
	house.LoadModelFromFile("..//Models//House//house.3ds");
	if (house.isLoaded())
		std::cout << "House model loaded successfully." << std::endl;
	else
	{
		std::cout << "Error! House model not loaded." << std::endl;
		return false;
	}
	std::cout << std::endl;
	std::cout << std::endl;
	for (unsigned i = 0; i < 6; i++)
	{
		auto model = std::make_shared < CAssimpModel >(house);
		model->setPosition(glm::vec3(-80.0f + i*30.0f, 0.f, 0.f));
		model->setScale(glm::vec3(3.0f, 3.0f, 3.0f));
		world.addVirtualObject(model);
	}
	
	//create wolves
	std::cout << "Loading wolf model..." << std::endl;
	CAssimpModel wolf;
	wolf.LoadModelFromFile("..//Models//Wolf//Wolf.obj");
	if (wolf.isLoaded())
		std::cout << "Wolf model loaded successfully." << std::endl;
	else
	{
		std::cout << "Error! Wolf model not loaded." << std::endl;
		return false;
	}
	for (unsigned i = 0; i < 7; i++)
	{
		auto model = std::make_shared < CAssimpModel >(wolf);
		model->setPosition(glm::vec3(-75.0f + i*30.0f, 0.f, 0.f));
		model->setScale(glm::vec3(1.8f, 1.8f, 1.8f));
		world.addVirtualObject(model);
	}
	std::cout << std::endl;
	std::cout << std::endl;
	CAssimpModel::FinalizeVBO();
	return true;
}