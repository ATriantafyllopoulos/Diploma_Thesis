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

int simulationTest()
{
	float l = 6; // cube side length
	float m = 1; // cube mass
	glm::mat3 I(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // cube inertia matrix
	I *= m * l*l / 6;
	glm::mat3 Iinv(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // cube inverse inertia matrix
	Iinv *= 6 / (m * l*l);
	
	const int collisions = 4; // number of collisions
	float accumulated_impulse[collisions] = { 0 }; // total impulse per contact point

	glm::vec3 r[collisions]; //collision points
	r[0] = glm::vec3(-l / 2, -l / 2, -l / 2); // first collision point
	r[1] = glm::vec3(l / 2, -l / 2, -l / 2); // second collision point
	r[2] = glm::vec3(-l / 2, -l / 2, l / 2); // third collision point
	r[3] = glm::vec3(l / 2, -l / 2, l / 2); // fourth collision point

	const glm::vec3 n(0, 1, 0); // collision normal
	glm::vec3 v(0, -1, 0); // linear velocity at time of collision
	const float expected_impulse = -m * glm::dot(v, n); // expected impulse output
	glm::vec3 w(0, 0, 0); // angular velocity at time of collision
	const int iterations = 8; // number of iterations per simulation step
	const int UPPER_BOUND = 100; // upper bound for accumulated impulse

	std::cout << "Initial linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
	std::cout << "Initial angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;

	for (int k = 0; k < iterations; k++)
	{
		for (int c = 0; c < collisions; c++)
		{
			glm::vec3 p = r[c]; // contact to be processed at this iteration
			float mc = 1 / m + glm::dot(glm::cross(Iinv * glm::cross(p, n), p), n); // active mass at current collision
			if (abs(mc) < 0.00001) mc = 1.f;
			float v_rel = glm::dot(v + cross(w, p), n); // relative velocity at current contact
			float corrective_impulse = -v_rel / mc; // corrective impulse magnitude
			if (corrective_impulse < 0)
				std::cout << "Negative corrective impulse encountered: " << corrective_impulse << std::endl;

			float temporary_impulse = accumulated_impulse[c]; // make a copy of old accumulated impulse
			temporary_impulse = temporary_impulse + corrective_impulse; // add corrective impulse to accumulated impulse
			//clamp new accumulated impulse
			if (temporary_impulse < 0)
				temporary_impulse = 0; // allow no negative accumulated impulses
			else if (temporary_impulse > UPPER_BOUND)
					temporary_impulse = UPPER_BOUND; // max upper bound for accumulated impulse
			// compute difference between old and new impulse
			corrective_impulse = temporary_impulse - accumulated_impulse[c];
			accumulated_impulse[c] = temporary_impulse; // store new clamped accumulated impulse
			// apply new clamped corrective impulse difference to velocity
			glm::vec3 impulse_vector = corrective_impulse * n;
			v = v + impulse_vector / m;
			w = w + Iinv * glm::cross(p, impulse_vector);
			std::cout << "Iteration: " << k;
			std::cout << " Contact: " << c;
			std::cout << " Applied impulse: " << corrective_impulse;
			std::cout << " New linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")";
			std::cout << " New angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")";
			std::cout << std::endl;
		}
	}
	float total_applied_impulse = 0;
	for (int c = 0; c < collisions; c++)
		total_applied_impulse += accumulated_impulse[c];
	
	std::cout << "Final linear velocity: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
	std::cout << "Final angular velocity: (" << w.x << ", " << w.y << ", " << w.z << ")" << std::endl;

	std::cout << "Total accumulated impulse: " << total_applied_impulse <<
		" (ground truth: " << expected_impulse << ")" << std::endl;

	int x;
	std::cout << "Enter any key to exit: ";
	cin >> x;
	return 1;
}

int main(void)
{
	//return simulationTest();
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
	window.Run();
	/*virtualWorld.initDemoMode();
	window.Demo();*/

	checkCudaErrors(cudaDeviceReset());
	if (psystem) delete psystem;
	if (renderer) delete renderer;

	return 1;
}

