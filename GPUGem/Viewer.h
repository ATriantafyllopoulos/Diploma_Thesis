#ifndef VIEWER_H
#define VIEWER_H

#include "Renderable.h"
// OpenGL and GLEW Header Files and Libraries
#include <GL/glew.h>
#include <GL/wglew.h>
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#include <memory>
#include <vector>
/**
Basic renderer class. Accessed only by the virtual world object (as of 06.03.2016 main has access too).
It is platform independent. In future, the windows handler with be sbstituted by another platform-specific class.
Handles all graphics procedure, including scene initialization and rendering. It is responsible to draw each renderable object.
Currently using openGL 3.2 version.

Known associated bugs: 
- window resize propagation
*/
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
	void resize(GLint w, GLint h);

protected:

	GLint windowWidth; // Store the width of our window
	GLint windowHeight; // Store the height of our window
};

#endif /* VIEWER_H_ */