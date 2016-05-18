#pragma once

#include "VirtualObject.h"
#include "Shader.h"
#include "vertexBufferObject.h"
#include "texture.h"


#pragma comment(lib, "assimp.lib")

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags


class CMaterial
{
public:
	int iTexture;
};

/**
Implemented using Assimp library
*/
class CAssimpModel : 
	public VirtualObject
{
public:
	bool LoadModelFromFile(char* sFilePath);

	static void FinalizeVBO();
	static void BindModelsVAO();
	void update(void) { ; }
	void draw(CShaderProgram *shader, const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight);
	
	bool isLoaded(void) {return bLoaded;}

	CAssimpModel(const double &m = 0, const glm::vec3 &p = glm::vec3(0, 0, 0), const glm::vec3 &v = glm::vec3(0, 0, 0), const glm::vec3 &i = glm::vec3(0, 0, 0)) :
		VirtualObject(m, p, v, i) {};
	~CAssimpModel();
private:
	bool bLoaded;
	static CVertexBufferObject vboModelData;
	static UINT uiVAO;
	static std::vector<CTexture> tTextures;
	std::vector<int> iMeshStartIndices;
	std::vector<int> iMeshSizes;
	std::vector<int> iMaterialIndices;
	int iNumMaterials;
};
