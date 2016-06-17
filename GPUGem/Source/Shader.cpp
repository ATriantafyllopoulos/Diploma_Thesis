#include "Shader.h"

CShader::CShader()
{
	bLoaded = false;
}

/*-----------------------------------------------

Name:	loadShader

Params:	sFile - path to a file
a_iType - type of shader (fragment, vertex, geometry)

Result:	Loads and compiles shader.

/*---------------------------------------------*/

bool CShader::loadShader(std::string sFile, int a_iType)
{
	std::vector<std::string> sLines;

	if (!GetLinesFromFile(sFile, false, &sLines))return false;

	const char** sProgram = new const char*[(int)sLines.size()];
	for (int i = 0; i < (int)sLines.size(); i++)
		sProgram[i] = sLines[i].c_str();

	uiShader = glCreateShader(a_iType);

	glShaderSource(uiShader, (int)sLines.size(), sProgram, NULL);
	glCompileShader(uiShader);

	delete[] sProgram;

	int iCompilationStatus;
	glGetShaderiv(uiShader, GL_COMPILE_STATUS, &iCompilationStatus);

	if (iCompilationStatus == GL_FALSE)
	{
		char sInfoLog[1024];
		char sFinalMessage[1536];
		int iLogLength;
		glGetShaderInfoLog(uiShader, 1024, &iLogLength, sInfoLog);
		sprintf(sFinalMessage, "Error! Shader file %s wasn't compiled! The compiler returned:\n\n%s", sFile.c_str(), sInfoLog);
		MessageBox(NULL, sFinalMessage, "Error", MB_ICONERROR);
		return false;
	}
	iType = a_iType;
	bLoaded = true;

	return true;
}

/*-----------------------------------------------

Name:    GetLinesFromFile

Params:  sFile - path to a file
         bIncludePart - whether to add include part only
         vResult - vector of strings to store result to

Result:  Loads and adds include part.

/*---------------------------------------------*/

bool CShader::GetLinesFromFile(std::string sFile, bool bIncludePart, std::vector<std::string>* vResult)
{
	FILE* fp = fopen(sFile.c_str(), "rt");
	if(!fp)return false;

	std::string sDirectory;
	int slashIndex = -1;
	for (int i = (int)(sFile.size())-1; i >= 0; i--)
	{
		if(sFile[i] == '\\' || sFile[i] == '/')
		{
			slashIndex = i;
			break;
		}
	}

	sDirectory = sFile.substr(0, slashIndex+1);

	// Get all lines from a file

	char sLine[255];

	bool bInIncludePart = false;

	while(fgets(sLine, 255, fp))
	{
		std::stringstream ss(sLine);
		std::string sFirst;
		ss >> sFirst;
		if(sFirst == "#include")
		{
			std::string sFileName;
			ss >> sFileName;
			if ((int)sFileName.size() > 0 && sFileName[0] == '\"' && sFileName[(int)sFileName.size() - 1] == '\"')
			{
				sFileName = sFileName.substr(1, (int)sFileName.size() - 2);
				GetLinesFromFile(sDirectory+sFileName, true, vResult);
			}
		}
		else if(sFirst == "#include_part")
			bInIncludePart = true;
		else if(sFirst == "#definition_part")
			bInIncludePart = false;
		else if(!bIncludePart || (bIncludePart && bInIncludePart))
			vResult->push_back(sLine);
	}
	fclose(fp);

	return true;
}

/*-----------------------------------------------

Name:	isLoaded

Params:	none

Result:	True if shader was loaded and compiled.

/*---------------------------------------------*/

bool CShader::isLoaded()
{
	return bLoaded;
}

/*-----------------------------------------------

Name:	getShaderID

Params:	none

Result:	Returns ID of a generated shader.

/*---------------------------------------------*/

UINT CShader::getShaderID()
{
	return uiShader;
}

/*-----------------------------------------------

Name:	deleteShader

Params:	none

Result:	Deletes shader and frees memory in GPU.

/*---------------------------------------------*/

void CShader::deleteShader()
{
	if (!isLoaded())return;
	bLoaded = false;
	glDeleteShader(uiShader);
}

CShaderProgram::CShaderProgram()
{
	bLinked = false;
}

/*-----------------------------------------------

Name:	createProgram

Params:	none

Result:	Creates a new program.

/*---------------------------------------------*/

void CShaderProgram::createProgram()
{
	uiProgram = glCreateProgram();
}

/*-----------------------------------------------

Name:	addShaderToProgram

Params:	sShader - shader to add

Result:	Adds a shader (like source file) to
a program, but only compiled one.

/*---------------------------------------------*/

bool CShaderProgram::addShaderToProgram(CShader* shShader)
{
	if (!shShader->isLoaded())
		return false;

	glAttachShader(uiProgram, shShader->getShaderID());

	return true;
}

/*-----------------------------------------------

Name:	linkProgram

Params:	none

Result:	Performs final linkage of OpenGL program.

/*---------------------------------------------*/

bool CShaderProgram::linkProgram()
{
	glLinkProgram(uiProgram);
	int iLinkStatus;
	glGetProgramiv(uiProgram, GL_LINK_STATUS, &iLinkStatus);
	bLinked = iLinkStatus == GL_TRUE;
	return bLinked;
}

/*-----------------------------------------------

Name:	deleteProgram

Params:	none

Result:	Deletes program and frees memory on GPU.

/*---------------------------------------------*/

void CShaderProgram::deleteProgram()
{
	if (!bLinked)
		return;
	bLinked = false;
	glDeleteProgram(uiProgram);
}

/*-----------------------------------------------

Name:	bind

Params:	none

Result:	Tells OpenGL to use this program.

/*---------------------------------------------*/

void CShaderProgram::bind() 
{
	if (bLinked)
		glUseProgram(uiProgram);
}

/*-----------------------------------------------

Name:	unbind

Params:	none

Result:	Tells OpenGL to stop using this program.

/*---------------------------------------------*/
void CShaderProgram::unbind() 
{
	glUseProgram(0);
}
/*-----------------------------------------------

Name:	getProgramID

Params:	none

Result:	Returns OpenGL generated shader program ID.

/*---------------------------------------------*/

UINT CShaderProgram::getProgramID()
{
	return uiProgram;
}

/*-----------------------------------------------

Name:		uniformSetters

Params:	yes, there are :)

Result:	These set of functions set most common
uniform variables.

/*---------------------------------------------*/

// Setting floats

void CShaderProgram::setUniform(std::string sName, float* fValues, int iCount)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform1fv(iLoc, iCount, fValues);
}

void CShaderProgram::setUniform(std::string sName, const float fValue)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform1fv(iLoc, 1, &fValue);
}

// Setting vectors

void CShaderProgram::setUniform(std::string sName, glm::vec2* vVectors, int iCount)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform2fv(iLoc, iCount, (GLfloat*)vVectors);
}

void CShaderProgram::setUniform(std::string sName, const glm::vec2 vVector)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform2fv(iLoc, 1, (GLfloat*)&vVector);
}

void CShaderProgram::setUniform(std::string sName, glm::vec3* vVectors, int iCount)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform3fv(iLoc, iCount, (GLfloat*)vVectors);
}

void CShaderProgram::setUniform(std::string sName, const glm::vec3 vVector)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform3fv(iLoc, 1, (GLfloat*)&vVector);
}

void CShaderProgram::setUniform(std::string sName, glm::vec4* vVectors, int iCount)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform4fv(iLoc, iCount, (GLfloat*)vVectors);
}

void CShaderProgram::setUniform(std::string sName, const glm::vec4 vVector)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform4fv(iLoc, 1, (GLfloat*)&vVector);
}

// Setting 3x3 matrices

void CShaderProgram::setUniform(std::string sName, glm::mat3* mMatrices, int iCount)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniformMatrix3fv(iLoc, iCount, FALSE, (GLfloat*)mMatrices);
}

void CShaderProgram::setUniform(std::string sName, const glm::mat3 mMatrix)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniformMatrix3fv(iLoc, 1, FALSE, (GLfloat*)&mMatrix);
}

// Setting 4x4 matrices

void CShaderProgram::setUniform(std::string sName, glm::mat4* mMatrices, int iCount)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniformMatrix4fv(iLoc, iCount, FALSE, (GLfloat*)mMatrices);
}

void CShaderProgram::setUniform(std::string sName, const glm::mat4 mMatrix)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniformMatrix4fv(iLoc, 1, FALSE, (GLfloat*)&mMatrix);
}

// Setting integers

void CShaderProgram::setUniform(std::string sName, int* iValues, int iCount)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform1iv(iLoc, iCount, iValues);
}

void CShaderProgram::setUniform(std::string sName, const int iValue)
{
	int iLoc = glGetUniformLocation(uiProgram, sName.c_str());
	glUniform1i(iLoc, iValue);
}