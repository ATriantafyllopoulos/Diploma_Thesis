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
	//return cubTest(10000);
	cudaError_t cudaStatus;
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //}
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
	//std::cout << "Setting up viewer..." << std::endl;
    virtualWorld.setViewer(&viewer);
	//std::cout << "Viewer is set up" << std::endl;
    window.setWorld(&virtualWorld); //set rendering window's virtual world
	//std::cout << "Everything is set up" << std::endl;
    //window.Run();
    virtualWorld.initDemoMode();
	//std::cout << "Initializing demo" << std::endl;
    window.Demo();
	//std::cout << "Quitting demo" << std::endl;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    if (psystem) delete psystem;
    if (renderer) delete renderer;

    return 1;
}

//
//cudaError_t findAABBCub(float4 *positions, float4 &min, float4 &max, int numberOfPrimitives);
//int mainTest(void)
//{
//	cudaError_t cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess)
//	{
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//	}
//
//	std::cout << "Hello world!" << std::endl;
//	int elements = 2048;
//	float *positions;
//	cudaMalloc((void**)&positions, sizeof(float4) * elements);
//	cudaStatus = simpleInitializer(
//			positions,
//			elements);
//	if (cudaStatus != cudaSuccess)
//	{
//		fprintf(stderr, "simpleInitializer failed!\n");
//		fprintf(stderr, "Cuda returned status: %d\n", cudaStatus);
//		fprintf(stderr, "Error code: %s\n", cudaGetErrorString(cudaStatus));
//	}
//
//	/*float *output;
//
//	cudaMalloc((void**)&output, sizeof(float) * 3 * elements);
//	cudaStatus = findAABBCub((float4 *)positions, (float3 *)output, elements);
//	if (cudaStatus != cudaSuccess)
//	{
//		fprintf(stderr, "findAABB failed!\n");
//		fprintf(stderr, "Cuda returned status: %d\n", cudaStatus);
//		fprintf(stderr, "Error code: %s\n", cudaGetErrorString(cudaStatus));
//	}
//
//	cudaFree(output);*/
//	float4 min, max;
//	cudaStatus = findAABBCub((float4 *)positions, min, max, elements);
//	if (cudaStatus != cudaSuccess)
//	{
//		fprintf(stderr, "findAABBCub failed!\n");
//		fprintf(stderr, "Cuda returned status: %d\n", cudaStatus);
//		fprintf(stderr, "Error code: %s\n", cudaGetErrorString(cudaStatus));
//	}
//	std::cout << min.x << min.y << min.z << min.w << std::endl;
//	std::cout << max.x << max.y << max.z << max.w << std::endl;
//	cudaFree(positions);
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
//	}
//	return 0;
//}
