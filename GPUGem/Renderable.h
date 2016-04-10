#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "glm\vec3.hpp"
/**
* A class for rendering and updating
*/
class Renderable
{
public:

	Renderable(const glm::vec3 &c = glm::vec3(0, 0, 0));

	virtual ~Renderable()
	{
	};

	virtual void draw() = 0;

	void setColour(const glm::vec3 &c);
	glm::vec3 getColour(void);

private:
	glm::vec3 colour;
};

#endif
