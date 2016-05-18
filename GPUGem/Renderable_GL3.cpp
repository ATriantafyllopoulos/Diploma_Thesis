#include "Renderable_GL3.h"


Renderable_GL3::Renderable_GL3()
{
	modelMatrix = glm::mat4(1.0);
	normalMatrix = glm::mat4(1.0);
	scale = glm::vec3(1.0);
}

void Renderable_GL3::setScale(const glm::vec3 &s)
{
	scale = s;
}
