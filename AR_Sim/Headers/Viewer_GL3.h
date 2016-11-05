#ifndef VIEWER_GL3_H
#define VIEWER_GL3_H

#if ( (defined(__MACH__)) && (defined(__APPLE__)) )   
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <OpenGL/glext.h>
#else
#include <stdlib.h>
#include <time.h>
#include "Platform.h"
#include <GL/glew.h>
#include <GL/gl.h>
#endif
// OpenGL and GLEW Header Files and Libraries

#include "Platform.h"
#include <GL/glew.h>
//#pragma comment(lib, "glew32.lib")
//#pragma comment(lib, "opengl32.lib")
#include <memory>
#include <vector>
#include "../Particles/renderParticles.h"
#include "Menu.h"
 
// Include input and output operations
#include <glfw3.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/trigonometric.hpp>

#include "objModel.h"

#define PI 3.14159265359
/**
Viewer implementation based on OpenGL 3.3 version using glew
Uses glm library to simulate OpenGL 1 behavior
Currently dependent on .NET

Update 10.4.2016: adding camera control functionality
*/

class Viewer_GL3
{

public:

	static Viewer_GL3 *instance;

    Viewer_GL3(GLFWwindow *inWindow);
	~Viewer_GL3();
	
	void resize(GLint w, GLint h);
	void render(void); // render scene

	void addParticleRenderer(ParticleRenderer *x)
	{
		renderer = x;
		renderer->setWindowSize(windowWidth, windowHeight);
	}
	
	glm::vec4 getViewport(void){ return viewport; }
	float getPixelDepth(int x, int y);
	glm::mat4 getProjectionMatrix(){ return projectionMatrix; }
	glm::mat4 getViewMatrix(){ return viewMatrix; }

	void toggleShowRangeData(void){ showRangeData = !showRangeData; };
	// setters
	void setRendererVBO(unsigned int id, int numberOfParticles){ renderer->setVertexBuffer(id, numberOfParticles); }
	void setViewMatrix(const glm::mat4 &x){ viewMatrix = x; }
	void setModelMatrixArray(glm::mat4 *x){ modelMatrix = x; }
	void setObjectNumber(const int &x){ number_of_objects = x; }
	void setNumberOfRangeData(const int &x) { renderer->setNumberOfRangeData(x); }
	void setRangeSampler(const GLuint &x) { renderer->setRangeSampler(x); }
	void setRangeVAO(const GLuint &x) { renderer->setRangeVAO(x); }
	void setRangeTexture(const GLuint &x) { renderer->setRangeTexture(x); }
	void setViewModeCommand(const int &mode){ viewMode = mode; }

	void addScaleFactor(const float &newFactor);
	void addObjectType(const int &type);
	void increaseNumberOfObjects() { number_of_objects++; }
private:
	// camera controls
	void cameraUpdate();
	float getAngleX(), getAngleY();// Functions that get viewing angles

	// camera variables
	glm::vec3 vEye, vView, vUp;
	float fSpeed;
	float fSensitivity; // How many degrees to rotate per pixel moved by mouse (nice value is 0.10)

	// timer
	void ResetTimer();
	void UpdateTimer();
	clock_t tLastFrame;
	float fFrameInterval;

    CShaderProgram shader; // Our GLSL shader

    GLFWwindow *window;
	glm::mat4 projectionMatrix; // Store the projection matrix  
	glm::mat4 viewMatrix; // Store the view matrix  

    bool create(); // called by constructor
	void init(void); // called by constructor, after a successfull call to create

	
	bool showRangeData; // Show cursor flag
	
	GLuint testingVAO; //Vertex Array Buffer used for testing
	int numOfParticles; //total number of particles
    CShader shVertex, shFragment;

	// obj rendering related variables
	CAssimpModel objModels[3];

	int viewMode;
	GLint windowWidth; // Store the width of our window
	GLint windowHeight; // Store the height of our window

	ParticleRenderer *renderer;
	glm::vec4 viewport;

	int number_of_objects;
	glm::mat4 *modelMatrix; // pointer to model matrix array
	
	glm::vec3 *scaleFactor;

	// integer pointer to type of object throw
	int *objectTypeArray;
};

#endif

