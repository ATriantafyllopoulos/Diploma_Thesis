#ifndef RENDERABLE_GL3_H
#define RENDERABLE_GL3_H

#include "Renderable.h"
#include "vertexBufferObject.h"
#include "texture.h"
#include "Shader.h"
#include <glm/gtc/matrix_transform.hpp>
class Renderable_GL3 :
	public Renderable
{
public:
	Renderable_GL3();
	virtual ~Renderable_GL3(){};

	void draw(){};
	virtual void draw(CShaderProgram *shader, const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight){ std::cout << "Error" << std::endl; };
	
	void setScale(const glm::vec3 &s);
protected:
	glm::mat4 normalMatrix;
	glm::mat4 modelMatrix;
	glm::vec3 scale;

	CTexture tAmbientTexture;
	CVertexBufferObject vboModelData;
	int shapeSizeCounter;
	unsigned int shapeVao;
	unsigned int shapeVbo;
	unsigned int shapeIndexVbo;
	unsigned int colourVbo;
};

#endif