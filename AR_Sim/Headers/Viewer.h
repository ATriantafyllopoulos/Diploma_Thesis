#ifndef VIEWER_H
#define VIEWER_H

#include "Renderable.h"
// OpenGL and GLEW Header Files and Libraries

#include "Platform.h"
#include <GL/glew.h>
//#pragma comment(lib, "glew32.lib")
//#pragma comment(lib, "opengl32.lib")
#include <memory>
#include <vector>
#include "../Particles/renderParticles.h"
#include "Menu.h"
/**
Basic renderer class. Accessed only by the virtual world object (as of 06.03.2016 main has access too).
It is platform independent. In future, the windows handler with be sbstituted by another platform-specific class.
Handles all graphics procedure, including scene initialization and rendering. It is responsible to draw each renderable object.
Currently using openGL 3.2 version.

Known associated bugs: 
- window resize propagation
*/
#include <glm/glm.hpp>   

class Viewer
{
public:
	/**
	* Used for dispatching functions
	*/
	static Viewer *instance;

	Viewer(); // is only valid in the context of a rendering window
	virtual ~Viewer();

	/**
	* Adds a renderable object in the drawable container
	*
	* @param (renderable object)
	*/
	virtual void addToDraw(Renderable *r) = 0;


	virtual void render() = 0;
	virtual void resize(GLint w, GLint h) = 0;
	virtual glm::mat4 getProjectionMatrix() = 0;
	virtual glm::mat4 getViewMatrix() = 0;
	virtual void viewModeCommand(const int &mode);
	virtual void toggleShowRangeData(void) = 0;
    void addParticleRenderer(ParticleRenderer *x)
    {
        renderer = x;
        renderer->setWindowSize(windowWidth, windowHeight);
    }
	void setRendererVBO(unsigned int id, int numberOfParticles){ renderer->setVertexBuffer(id, numberOfParticles); }
	glm::vec4 getViewport(void){ return viewport; }
	float getPixelDepth(int x, int y);
//	virtual void toggleShowRangeData() = 0;
	void setViewMatrix(const glm::mat4 &x){ renderer->setViewMatrix(x); }

protected:

	int viewMode;
	GLint windowWidth; // Store the width of our window
	GLint windowHeight; // Store the height of our window

	ParticleRenderer *renderer;
	glm::vec4 viewport;

public:
	void setNumberOfRangeData(const int &x) { renderer->setNumberOfRangeData(x); }
	void setRangeSampler(const GLuint &x) { renderer->setRangeSampler(x); }
	void setRangeVAO(const GLuint &x) { renderer->setRangeVAO(x); }
	void setRangeTexture(const GLuint &x) { renderer->setRangeTexture(x); }
};

#endif /* VIEWER_H_ */
