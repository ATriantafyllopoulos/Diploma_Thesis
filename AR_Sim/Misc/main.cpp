#include "WindowsHandler.h"
cudaError_t findAABB(float4 *positions, float3 *d_out, int numberOfPrimitives);
cudaError_t simpleLoadRangeImage(
	unsigned short *image,
	float *positions,
	int imageWidth,
	int imageHeight);

cudaError_t simpleInitializer(
	float *positions,
	int numberOfDataToTest);

#include <iostream>
#include "Viewer_GL3.h"
void testCubReduce(int elements);
int cubTest(int elements)
{
	testCubReduce(elements);
	return 1;
}

int main(void)
{
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	WindowsHandler window("Simulation", 640, 480);
	VirtualWorld virtualWorld;
	ParticleSystem *psystem = new ParticleSystem(0, make_uint3(64, 64, 64), true);

	ParticleRenderer *renderer = new ParticleRenderer;
	if (!renderer)
	{
		std::cout << "Renderer initialization gone wrong" << std::endl;
	}
	Viewer_GL3 viewer(window.getWindow()); //create renderer
	virtualWorld.addParticleSystem(psystem);
	virtualWorld.initializeParticleSystem();

	renderer->setParticleRadius(psystem->getParticleRadius());
	renderer->setColorBuffer(psystem->getColorBuffer());
	viewer.addParticleRenderer(renderer);
	virtualWorld.setViewer(&viewer);
	window.setWorld(&virtualWorld); //set rendering window's virtual world
	//window.Run();
	virtualWorld.initDemoMode();
	window.Demo();

	checkCudaErrors(cudaDeviceReset());
	if (psystem) delete psystem;
	if (renderer) delete renderer;

	return 1;
}

