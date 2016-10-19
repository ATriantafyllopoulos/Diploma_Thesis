#ifndef TEXTURE_H
#define TEXTURE_H
#include <string>

#include "Platform.h"
#include <GL/glew.h>
//#include <GL/wglew.h>
//#pragma comment(lib, "glew32.lib")
//#pragma comment(lib, "opengl32.lib")
#include <vector>
#include <iostream>
//#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"

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

Class:		CTexture

Purpose:	Wraps OpenGL texture
object and performs
their loading.

********************************/

class CTexture
{
public:
    void CreateEmptyTexture(int a_iWidth, int a_iHeight, GLenum format);
    void CreateFromData(unsigned char* bData, int a_iWidth, int a_iHeight, GLenum format, bool bGenerateMipMaps = false);

    bool ReloadTexture();

    bool LoadTexture2D(std::string a_sPath, bool bGenerateMipMaps = false);
    void BindTexture(int iTextureUnit = 0);

    void SetFiltering(int a_tfMagnification, int a_tfMinification);

    void SetSamplerParameter(GLenum parameter, GLenum value);

    int GetMinificationFilter();
    int GetMagnificationFilter();

    int GetWidth();
    int GetHeight();
    int GetBPP();

    uint GetTextureID();

    std::string GetPath();

    void DeleteTexture();

    CTexture();
private:

    int iWidth, iHeight, iBPP; // Texture width, height, and bytes per pixel
    uint uiTexture; // Texture name
    uint uiSampler; // Sampler name
    bool bMipMapsGenerated;

    int tfMinification, tfMagnification;

    std::string sPath;
};

#define NUMTEXTURES 1
extern CTexture tTextures[NUMTEXTURES];
void LoadAllTextures();
#endif
