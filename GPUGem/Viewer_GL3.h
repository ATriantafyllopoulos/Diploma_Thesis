#ifndef VIEWER_GL3_H
#define VIEWER_GL3_H

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
#include "Renderable_GL3.h"
#include "shader.h"
#include <glm/glm.hpp>   
#include <glm/gtc/matrix_transform.hpp>
class Viewer_GL3 :
	public Viewer
{

public:

	Viewer_GL3(HWND hwnd);
	~Viewer_GL3();

	void render(void);
	void Viewer::addToDraw(Renderable *r);

private:
	Shader *shader; // Our GLSL shader  
	HGLRC hrc; // Rendering context
	HDC hdc; // Device context
	HWND hwnd; // Window identifier

	glm::mat4 projectionMatrix; // Store the projection matrix  
	glm::mat4 viewMatrix; // Store the view matrix  
	glm::mat4 modelMatrix; // Store the model matrix  

	std::vector<Renderable_GL3 *> models;

	bool create(HWND hwnd); // called by constructor
	void init(void); // called by constructor, after a successfull call to create
};

#endif

