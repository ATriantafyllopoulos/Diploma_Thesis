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
#include "Viewer.h"
// Include input and output operations
#include <glfw3.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include <string>
#include <iostream>
#include <glm/glm.hpp>   
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/trigonometric.hpp>

#define PI 3.14159265359
/**
Viewer implementation based on OpenGL 3.3 version using glew
Uses glm library to simulate OpenGL 1 behavior
Currently dependent on .NET

Update 10.4.2016: adding camera control functionality
*/

class Viewer_GL3 : public Viewer
{

public:

    Viewer_GL3(GLFWwindow *inWindow);
	~Viewer_GL3();
	
	void render(void); // render scene
	void addToDraw(Renderable *r); // add object to draw queue
	void resize(GLint w, GLint h);
	//camera functions
	void rotateWithMouse();
	void cameraUpdate();
	float getAngleX(), getAngleY();// Functions that get viewing angles
	void toggleShowRangeData(void){showRangeData =  !showRangeData;};
	glm::mat4 getProjectionMatrix(){ return projectionMatrix; }
	glm::mat4 getViewMatrix(){ return viewMatrix; }
//	void toggleShowRangeData(){ showRangeData = !showRangeData; }
private:
//	bool showRangeData;
	void ResetTimer();
	void UpdateTimer();
	clock_t tLastFrame;
	float fFrameInterval;

    CShaderProgram shader; // Our GLSL shader
	//HGLRC hrc; // Rendering context
	//HDC hdc; // Device context
	//HWND hwnd; // Window identifier
    GLFWwindow *window;
	glm::mat4 projectionMatrix; // Store the projection matrix  
	glm::mat4 viewMatrix; // Store the view matrix  
	glm::mat4 modelMatrix; // Store the model matrix  

	//std::vector<std::shared_ptr<Renderable_GL3>> models; // objects to be drawn on screen

    bool create(); // called by constructor
	void init(void); // called by constructor, after a successfull call to create

	//camera variables
	glm::vec3 vEye, vView, vUp;
	float fSpeed;
	float fSensitivity; // How many degrees to rotate per pixel moved by mouse (nice value is 0.10)
    //POINT pCur; // For mouse rotation
	
	bool showRangeData; // Show cursor flag
	

	//struct cudaGraphicsResource* testingVBO_CUDA; //CUDA resources pointer
	GLuint testingVAO; //Vertex Array Buffer used for testing
	int numOfParticles; //total number of particles
    CShader shVertex, shFragment;

};

#endif

