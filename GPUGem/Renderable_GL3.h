#ifndef RENDERABLE_GL3_H
#define RENDERABLE_GL3_H

#include "Renderable.h"

#include <GL/glew.h>

#include <GL/wglew.h>
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#include "Shader.h"
#include <glm/gtc/matrix_transform.hpp>
class Renderable_GL3 :
	public Renderable
{
public:
	Renderable_GL3();
	~Renderable_GL3();

	void draw(){};
	virtual void draw(Shader *shader, const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight) = 0;

protected:
	int shapeSizeCounter;
	unsigned int shapeVao;
	unsigned int shapeVbo;
	unsigned int shapeIndexVbo;
	unsigned int colourVbo;
};

#endif