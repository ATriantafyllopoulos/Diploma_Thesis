#include "Renderable.h"

Renderable::Renderable(const glm::vec3 &c)
{
	colour = c;
}

void Renderable::setColour(const glm::vec3 &c)
{
	colour = c;
}

glm::vec3 Renderable::getColour(void)
{
	return colour;
}
