#ifndef TEXTURE_H
#define TEXTURE_H
#include <string>
#include <GL/glew.h>
#include <GL/wglew.h>
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#include <vector>
#include <iostream>
//#include <SOIL.h>
#include <FreeImage.h>
enum ETextureFiltering
{
	TEXTURE_FILTER_MAG_NEAREST = 0, // Nearest criterion for magnification
	TEXTURE_FILTER_MAG_BILINEAR, // Bilinear criterion for magnification
	TEXTURE_FILTER_MIN_NEAREST, // Nearest criterion for minification
	TEXTURE_FILTER_MIN_BILINEAR, // Bilinear criterion for minification
	TEXTURE_FILTER_MIN_NEAREST_MIPMAP, // Nearest criterion for minification, but on closest mipmap
	TEXTURE_FILTER_MIN_BILINEAR_MIPMAP, // Bilinear criterion for minification, but on closest mipmap
	TEXTURE_FILTER_MIN_TRILINEAR, // Bilinear criterion for minification on two closest mipmaps, then averaged
};

/********************************

Class:	CTexture

Purpose:	Wraps OpenGL texture
			object and performs
			their loading.

********************************/

class CTexture
{
public:
	void createFromData(BYTE* bData, int a_iWidth, int a_iHeight, int a_iBPP, GLenum format, bool bGenerateMipMaps = false);
	bool loadTexture2D(std::string a_sPath, bool bGenerateMipMaps = false);
	void bindTexture(int iTextureUnit = 0);

	void setFiltering(int a_tfMagnification, int a_tfMinification);

	void setSamplerParameter(GLenum parameter, GLenum value);

	int getMinificationFilter();
	int getMagnificationFilter();

	int getWidth();
	int getHeight();
	int getBPP();
	std::string GetPath();
	void releaseTexture();

	CTexture();
private:
	int iWidth, iHeight, iBPP; // Texture width, height, and bytes per pixel
	UINT uiTexture; // Texture name
	UINT uiSampler; // Sampler name
	bool bMipMapsGenerated;

	int tfMinification, tfMagnification;

	std::string sPath;
};

#endif
