#ifndef VIEWER_GL3_H
#define VIEWER_GL3_H

#if ( (defined(__MACH__)) && (defined(__APPLE__)) )   
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <OpenGL/glext.h>
#else
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#endif
#include "Viewer.h"
// Include Windows functions
#ifndef UNICODE
#define UNICODE
#endif

#ifndef _UNICODE
#define _UNICODE
#endif
#include <Windows.h>
// Include input and output operations

#include <string>
#include <iostream>
#include "objModel.h"
#include "Renderable_GL3.h"
#include "shader.h"
#include <glm/glm.hpp>   
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#define PI 3.14159265359
/**
Viewer implementation based on OpenGL 3.3 version using glew
Uses glm library to simulate OpenGL 1 behavior
Currently dependent on .NET

Update 10.4.2016: adding camera control functionality
*/
class Viewer_GL3 :
	public Viewer
{

public:

	Viewer_GL3(HWND hwnd);
	~Viewer_GL3();
	
	void render(void); // render scene
	void addToDraw(Renderable *r); // add object to draw queue

	//camera functions
	void rotateWithMouse();
	void cameraUpdate();
	float getAngleX(), getAngleY();// Functions that get viewing angles

private:
	CShaderProgram shader; // Our GLSL shader  
	HGLRC hrc; // Rendering context
	HDC hdc; // Device context
	HWND hwnd; // Window identifier

	glm::mat4 projectionMatrix; // Store the projection matrix  
	glm::mat4 viewMatrix; // Store the view matrix  
	glm::mat4 modelMatrix; // Store the model matrix  

	std::vector<std::shared_ptr<Renderable_GL3>> models; // objects to be drawn on screen

	bool create(HWND hwnd); // called by constructor
	void init(void); // called by constructor, after a successfull call to create

	//camera variables
	glm::vec3 vEye, vView, vUp;
	float fSpeed;
	float fSensitivity; // How many degrees to rotate per pixel moved by mouse (nice value is 0.10)
	POINT pCur; // For mouse rotation

	CShader shVertex, shFragment, shLight;
};

#endif

