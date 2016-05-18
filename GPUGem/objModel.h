#pragma once

#include "Shader.h"
#include "vertexBufferObject.h"
#include "texture.h"
#include "common_header.h"
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include "VirtualObject.h"
class CMaterial
{
public:
	int iTexture;
};

class CAssimpModel : public VirtualObject
{
public:
	bool LoadModelFromFile(char* sFilePath);
	bool isLoaded(void){ return bLoaded; }
	void update(void){};
	static void FinalizeVBO();
	static void BindModelsVAO();
	void draw(CShaderProgram *shader, const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight);
	void RenderModel();
	CAssimpModel();
private:
	bool bLoaded;
	static CVertexBufferObject vboModelData;
	static UINT uiVAO;
	static vector<CTexture> tTextures;
	vector<int> iMeshStartIndices;
	vector<int> iMeshSizes;
	vector<int> iMaterialIndices;
	int iNumMaterials;
};

