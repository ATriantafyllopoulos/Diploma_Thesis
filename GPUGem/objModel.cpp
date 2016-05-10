#include "objModel.h"

#pragma comment(lib, "assimp.lib")

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

CVertexBufferObject CAssimpModel::vboModelData;
UINT CAssimpModel::uiVAO;
std::vector<CTexture> CAssimpModel::tTextures;

/*-----------------------------------------------

Name:	GetDirectoryPath

Params:	sFilePath - guess ^^

Result: Returns directory name only from filepath.

/*---------------------------------------------*/

std::string GetDirectoryPath(std::string sFilePath)
{
	// Get directory path
	std::string sDirectory = "";
	for (int i = (int)sFilePath.size(); i >= 0; i--)
	if (sFilePath[i] == '\\' || sFilePath[i] == '/')
	{
		sDirectory = sFilePath.substr(0, i + 1);
		break;
	}
	return sDirectory;
}

CAssimpModel::CAssimpModel()
{
	bLoaded = false;
}

/*-----------------------------------------------

Name:	LoadModelFromFile

Params:	sFilePath - guess ^^

Result: Loads model using Assimp library.

/*---------------------------------------------*/

bool CAssimpModel::LoadModelFromFile(char* sFilePath)
{
	if (vboModelData.getBuffer() == 0)
	{
		vboModelData.createVBO();
		tTextures.reserve(50);
	}
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(sFilePath,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType);

	if (!scene)
	{
		//MessageBox(appMain.hWnd, "Couldn't load model ", "Error Importing Asset", MB_ICONERROR);
		return false;
	}

	const int iVertexTotalSize = sizeof(aiVector3D) * 2 + sizeof(aiVector2D);

	int iTotalVertices = 0;

	for (unsigned i = 0; i < scene->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[i];
		int iMeshFaces = mesh->mNumFaces;
		iMaterialIndices.push_back(mesh->mMaterialIndex);
		int iSizeBefore = vboModelData.GetCurrentSize();
		iMeshStartIndices.push_back(iSizeBefore / iVertexTotalSize);
		for (int j = 0; j < iMeshFaces; j++)
		{
			const aiFace& face = mesh->mFaces[j];
			for (int k = 0; k < 3; k++)
			{
				aiVector3D pos = mesh->mVertices[face.mIndices[k]];
				aiVector3D uv = mesh->mTextureCoords[0][face.mIndices[k]];
				aiVector3D normal = mesh->HasNormals() ? mesh->mNormals[face.mIndices[k]] : aiVector3D(1.0f, 1.0f, 1.0f);
				vboModelData.addData(&pos, sizeof(aiVector3D));
				vboModelData.addData(&uv, sizeof(aiVector2D));
				vboModelData.addData(&normal, sizeof(aiVector3D));
			}
		}
		int iMeshVertices = mesh->mNumVertices;
		iTotalVertices += iMeshVertices;
		iMeshSizes.push_back((vboModelData.GetCurrentSize() - iSizeBefore) / iVertexTotalSize);
	}
	iNumMaterials = scene->mNumMaterials;

	std::vector<int> materialRemap(iNumMaterials);

	for (int i = 0; i < iNumMaterials; i++)
	{
		const aiMaterial* material = scene->mMaterials[i];
		int a = 5;
		int texIndex = 0;
		aiString path;  // filename

		if (material->GetTexture(aiTextureType_DIFFUSE, texIndex, &path) == AI_SUCCESS)
		{
			std::string sDir = GetDirectoryPath(sFilePath);
			std::string sTextureName = path.data;
			std::string sFullPath = sDir + sTextureName;
			int iTexFound = -1;
			for (int j = 0; j < (int)tTextures.size(); j++)
				if (sFullPath == tTextures[j].GetPath())
			{
				iTexFound = j;
				break;
			}
			if (iTexFound != -1)materialRemap[i] = iTexFound;
			else
			{
				CTexture tNew;
				tNew.loadTexture2D(sFullPath, true);
				materialRemap[i] = (int)tTextures.size();
				tTextures.push_back(tNew);
			}
		}
	}
	for (int i = 0; i < (int)iMeshSizes.size(); i++)
	{
		int iOldIndex = iMaterialIndices[i];
		iMaterialIndices[i] = materialRemap[iOldIndex];
	}

	return bLoaded = true;
}

/*-----------------------------------------------

Name:	FinalizeVBO

Params: none

Result: Uploads all loaded model data in one global
models' VBO.

/*---------------------------------------------*/

void CAssimpModel::FinalizeVBO()
{
	glGenVertexArrays(1, &uiVAO);
	glBindVertexArray(uiVAO);
	vboModelData.bindVBO();
	vboModelData.uploadDataToGPU(GL_STATIC_DRAW);
	// Vertex positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(aiVector3D) + sizeof(aiVector2D), 0);
	// Texture coordinates
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(aiVector3D) + sizeof(aiVector2D), (void*)sizeof(aiVector3D));
	// Normal std::vectors
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(aiVector3D) + sizeof(aiVector2D), (void*)(sizeof(aiVector3D) + sizeof(aiVector2D)));
}

/*-----------------------------------------------

Name:	BindModelsVAO

Params: none

Result: Binds VAO of models with their VBO.

/*---------------------------------------------*/

void CAssimpModel::BindModelsVAO()
{
	glBindVertexArray(uiVAO);
}

/*-----------------------------------------------

Name:	RenderModel

Params: none

Result: Guess what it does ^^.

/*---------------------------------------------*/

void CAssimpModel::draw(CShaderProgram *shader, const glm::mat4 &projectionMatrix, const glm::vec3 &cameraEye, const glm::mat4 &viewMatrix, const int &windowWidth, const int &windowHeight)
{
	if (!bLoaded)
		return;
	shader->bind();
	for (int i = 0; i < (int)iMeshSizes.size(); i++)
	{
		int iMatIndex = iMaterialIndices[i];
		tTextures[iMatIndex].bindTexture();
		glDrawArrays(GL_TRIANGLES, iMeshStartIndices[i], iMeshSizes[i]);
	}
	shader->unbind();
}