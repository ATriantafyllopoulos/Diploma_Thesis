#pragma once
#include <string>
#include <vector>
#include "Renderable_GL3.h"
#include <sstream>
/********************************

Class:	CObjModel

Purpose: Class for handling obj
		 model files.

********************************/

class CObjModel : public Renderable_GL3
{
public:
	bool loadModel(std::string sFileName, std::string sMtlFileName);
	void draw(Shader *shader, const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight);
	void releaseModel();

	int getPolygonCount();

	CObjModel();
private:
	bool bLoaded;
	int iAttrBitField;
	int iNumFaces;
	
	bool loadMaterial(std::string sFullMtlFileName);
};