#pragma once

/********************************

Class:		CVertexBufferObject

Purpose:	Wraps OpenGL vertex buffer
object.

********************************/
#include <string>

#include "Platform.h"
#include <GL/glew.h>
//#include <GL/gl.h>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"
class CVertexBufferObject
{
public:
	void CreateVBO(int a_iSize = 0);
	void DeleteVBO();

	void* MapBufferToMemory(int iUsageHint);
    void* MapSubBufferToMemory(int iUsageHint, uint uiOffset, uint uiLength);
	void UnmapBuffer();

	void BindVBO(int a_iBufferType = GL_ARRAY_BUFFER);
	void UploadDataToGPU(int iUsageHint);

    void AddData(void* ptrData, uint uiDataSize);

	void* GetDataPointer();
    uint GetBufferID();

	int GetCurrentSize();

	CVertexBufferObject();

private:
    uint uiBuffer;
	int iSize;
	int iCurrentSize;
	int iBufferType;
    std::vector<unsigned char> data;

	bool bDataUploaded;
};
