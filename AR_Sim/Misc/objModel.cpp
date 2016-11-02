#include "objModel.h"

#pragma comment(lib, "assimp.lib")


CVertexBufferObject CAssimpModel::vboModelData;
uint CAssimpModel::uiVAO;
vector<CTexture> CAssimpModel::tTextures;

/*-----------------------------------------------

Name:	GetDirectoryPath

Params:	sFilePath - guess ^^

Result: Returns directory name only from filepath.

/*---------------------------------------------*/

string GetDirectoryPath(string sFilePath)
{
	// Get directory path
	string sDirectory = "";
	RFOR(i, ESZ(sFilePath) - 1)if (sFilePath[i] == '\\' || sFilePath[i] == '/')
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
	if (vboModelData.GetBufferID() == 0)
	{
		vboModelData.CreateVBO();
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
	aiVector3D cm(0, 0, 0);
	int number_of_vertices = 0;
	FOR(i, scene->mNumMeshes)
	{
		aiMesh* mesh = scene->mMeshes[i];
		int iMeshFaces = mesh->mNumFaces;
		iMaterialIndices.push_back(mesh->mMaterialIndex);
		int iSizeBefore = vboModelData.GetCurrentSize();
		iMeshStartIndices.push_back(iSizeBefore / iVertexTotalSize);
		FOR(j, iMeshFaces)
		{
			const aiFace& face = mesh->mFaces[j];
			FOR(k, 3)
			{
				aiVector3D pos = mesh->mVertices[face.mIndices[k]];
				cm += pos;
				number_of_vertices++;
			}
		}
	}
	cm /= (float)number_of_vertices;
	std::cout << "ASSIMP model center of mass: (" << cm.x << ", " << cm.y << ", " << cm.z << ")" << std::endl;
	aiVector3D test(0, 0, 0);
	FOR(i, scene->mNumMeshes)
	{
		aiMesh* mesh = scene->mMeshes[i];
		int iMeshFaces = mesh->mNumFaces;
		iMaterialIndices.push_back(mesh->mMaterialIndex);
		int iSizeBefore = vboModelData.GetCurrentSize();
		iMeshStartIndices.push_back(iSizeBefore / iVertexTotalSize);
		FOR(j, iMeshFaces)
		{
			const aiFace& face = mesh->mFaces[j];
			FOR(k, 3)
			{
				aiVector3D pos = mesh->mVertices[face.mIndices[k]] - cm;
				test += pos;
				aiVector3D uv = mesh->mTextureCoords[0][face.mIndices[k]];
				aiVector3D normal = mesh->HasNormals() ? mesh->mNormals[face.mIndices[k]] : aiVector3D(1.0f, 1.0f, 1.0f);
				vboModelData.AddData(&pos, sizeof(aiVector3D));
				vboModelData.AddData(&uv, sizeof(aiVector2D));
				vboModelData.AddData(&normal, sizeof(aiVector3D));
			}
		}
		int iMeshVertices = mesh->mNumVertices;
		iTotalVertices += iMeshVertices;
		iMeshSizes.push_back((vboModelData.GetCurrentSize() - iSizeBefore) / iVertexTotalSize);
	}
	iNumMaterials = scene->mNumMaterials;
	test /= (float)number_of_vertices;
	std::cout << "ASSIMP model corrected center of mass: (" << test.x << ", " << test.y << ", " << test.z << ")" << std::endl;
	vector<int> materialRemap(iNumMaterials);

	FOR(i, iNumMaterials)
	{
		const aiMaterial* material = scene->mMaterials[i];
		int a = 5;
		int texIndex = 0;
		aiString path;  // filename

		if (material->GetTexture(aiTextureType_DIFFUSE, texIndex, &path) == AI_SUCCESS)
		{
			string sDir = GetDirectoryPath(sFilePath);
			string sTextureName = path.data;
			string sFullPath = sDir + sTextureName;
			int iTexFound = -1;
			FOR(j, ESZ(tTextures))if (sFullPath == tTextures[j].GetPath())
			{
				iTexFound = j;
				break;
			}
			if (iTexFound != -1)materialRemap[i] = iTexFound;
			else
			{
				CTexture tNew;
				tNew.LoadTexture2D(sFullPath, true);
				materialRemap[i] = ESZ(tTextures);
				tTextures.push_back(tNew);
			}
		}
	}

	FOR(i, ESZ(iMeshSizes))
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
	vboModelData.BindVBO();
	vboModelData.UploadDataToGPU(GL_STATIC_DRAW);
	// Vertex positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(aiVector3D) + sizeof(aiVector2D), 0);
	// Texture coordinates
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(aiVector3D) + sizeof(aiVector2D), (void*)sizeof(aiVector3D));
	// Normal vectors
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

Result: Render model on screen.

/*---------------------------------------------*/

void CAssimpModel::RenderModel()
{
	if (!bLoaded)return;
	int iNumMeshes = ESZ(iMeshSizes);
	glBindTexture(GL_TEXTURE_2D, 0);
	FOR(i, iNumMeshes)
	{
		//int iMatIndex = iMaterialIndices[i];
		//tTextures[iMatIndex].BindTexture();	
		glDrawArrays(GL_TRIANGLES, iMeshStartIndices[i], iMeshSizes[i]);
	}
}

void CAssimpModel::draw(CShaderProgram *shader)
{
	if (!bLoaded)
		return;

	//transformations
	//TO BE ADDED: rotation

	//modelMatrix = glm::translate(glm::mat4(1.0), position); // translate
	//modelMatrix = modelMatrix * rotationMatrix; // rotate
	//modelMatrix = glm::scale(modelMatrix, scale); // scale
	//normalMatrix = glm::transpose(glm::inverse(modelMatrix)); // pseudo normals


	//
	//shader->setUniform("matrices.modelMatrix", modelMatrix);
	//glBindVertexArray(uiVAO);
	for (int i = 0; i < (int)iMeshSizes.size(); i++)
	{
		int iMatIndex = iMaterialIndices[i];
		tTextures[iMatIndex].BindTexture();
		//glBindTexture(GL_TEXTURE_2D, 0);
		glDrawArrays(GL_TRIANGLES, iMeshStartIndices[i], iMeshSizes[i]);
	}
	//glBindVertexArray(0);

}