#pragma once

#include "Shader.h"
#include "vertexBufferObject.h"
#include "texture.h"

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include "VirtualObject.h"

#include <ctime>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <sstream>
#include <queue>
#include <map>
#include <set>

using namespace std;
#include "Platform.h"
#include <GL/glew.h>
#include <glm/glm.hpp>

// Some useful defines

#define FOR(q,n) for(int q=0;q<n;q++)
#define SFOR(q,s,e) for(int q=s;q<=e;q++)
#define RFOR(q,n) for(int q=n;q>=0;q--)
#define RSFOR(q,s,e) for(int q=s;q>=e;q--)

#define ESZ(elem) (int)elem.size()

class CMaterial
{
public:
	int iTexture;
};

class CAssimpModel
{
public:
	bool LoadModelFromFile(char* sFilePath);
	bool isLoaded(void){ return bLoaded; }
    void update(void){}
	static void FinalizeVBO();
	static void BindModelsVAO();
	//void draw(CShaderProgram *shader, const glm::mat4 &projectionMatrix, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight);
	void draw(CShaderProgram *shader);
	void RenderModel();
	CAssimpModel();
private:
	bool bLoaded;
	static CVertexBufferObject vboModelData;
    static uint uiVAO;
	static vector<CTexture> tTextures;
	vector<int> iMeshStartIndices;
	vector<int> iMeshSizes;
	vector<int> iMaterialIndices;
	int iNumMaterials;
	int iNumVertices;
};

