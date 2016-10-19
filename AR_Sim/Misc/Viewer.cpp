#include "Viewer.h"


Viewer *Viewer::instance = NULL;
Viewer::Viewer()
{
	instance = this;
}

Viewer::~Viewer()
{
}

void Viewer::viewModeCommand(const int &mode)
{
	viewMode = mode;
}

float Viewer::getPixelDepth(int x, int y)
{
	float depth;
	glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
	return depth;
}
