#ifndef __SHADER_H
#define __SHADER_H

#if ( (defined(__MACH__)) && (defined(__APPLE__)) )   
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <OpenGL/glext.h>
#else
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#endif

#include <string>

/**
A wrapper class for the necessary shader routines.
Taken as is by online tutorials. Should be studied in depth.
*/
class Shader {
public:
	Shader();
	~Shader();
	Shader(const char *vsFile, const char *fsFile);
	void init(const char *vsFile, const char *fsFile);
	void bind();
	void unbind();
	unsigned int id();

private:
	unsigned int shader_id;
	unsigned int shader_vp;
	unsigned int shader_fp;
};
#endif