#pragma once

#include "Renderable_GL3.h"
#include "Shader.h"
#include "vertexBufferObject.h"
#include "texture.h"

class CMaterial
{
public:
	int iTexture;
};

class CAssimpModel : 
	public Renderable_GL3
{
public:
	bool LoadModelFromFile(char* sFilePath);

	static void FinalizeVBO();
	static void BindModelsVAO();

	void draw(CShaderProgram *shader, const glm::mat4 &projectionMatrix, const glm::vec3 &cameraEye, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight);
	CAssimpModel();
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
