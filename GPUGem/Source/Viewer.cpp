#include "Viewer.h"


Viewer *Viewer::instance = NULL;
Viewer::Viewer()
{
	instance = this;
}

Viewer::~Viewer()
{
}

/**
Resize function. Called by virtual world after a resize event caught by the windows API.
Currently there is a propagation delay causing an unwanted effect.
*/
void Viewer::resize(GLint w, GLint h)
{
	windowWidth = w;
	windowHeight = h;
}
