#include "texture.h"

//#include <FreeImage.h>

//#pragma comment(lib, "FreeImage.lib")

CTexture::CTexture()
{
    bMipMapsGenerated = false;
}

/*-----------------------------------------------

Name:	CreateEmptyTexture

Params:	a_iWidth, a_iHeight - dimensions
format - format of data

Result:	Creates texture from provided data.

/*---------------------------------------------*/

void CTexture::CreateEmptyTexture(int a_iWidth, int a_iHeight, GLenum format)
{
    glGenTextures(1, &uiTexture);
    glBindTexture(GL_TEXTURE_2D, uiTexture);
    if (format == GL_RGBA || format == GL_BGRA)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, NULL);
    // We must handle this because of internal format parameter
    else if (format == GL_RGB || format == GL_BGR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, NULL);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, format, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, NULL);

    glGenSamplers(1, &uiSampler);
}

/*-----------------------------------------------

Name:	CreateFromData

Params:	a_sPath - path to the texture
format - format of data
bGenerateMipMaps - whether to create mipmaps

Result:	Creates texture from provided data.

/*---------------------------------------------*/

void CTexture::CreateFromData(unsigned char* bData, int a_iWidth, int a_iHeight, GLenum format, bool bGenerateMipMaps)
{
    // Generate an OpenGL texture ID for this texture
    glGenTextures(1, &uiTexture);
    glBindTexture(GL_TEXTURE_2D, uiTexture);
    if (format == GL_RGBA || format == GL_BGRA)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bData);
    // We must handle this because of internal format parameter
    else if (format == GL_RGB || format == GL_BGR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bData);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, format, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bData);
    if (bGenerateMipMaps)glGenerateMipmap(GL_TEXTURE_2D);
    glGenSamplers(1, &uiSampler);

    sPath = "";
    bMipMapsGenerated = bGenerateMipMaps;
    iWidth = a_iWidth;
    iHeight = a_iHeight;
}

/*-----------------------------------------------

Name:	LoadTexture2D

Params:	a_sPath - path to the texture
bGenerateMipMaps - whether to create mipmaps

Result:	Loads texture from a file, supports most
graphics formats.

/*---------------------------------------------*/

bool CTexture::LoadTexture2D(std::string a_sPath, bool bGenerateMipMaps)
{
    cv::Mat textureImage = cv::imread(a_sPath.c_str());

    unsigned char* bDataPointer = textureImage.data; // Retrieve the image data

    GLenum format;
    format = GL_BGR;
    iWidth = textureImage.cols;
    iHeight = textureImage.rows;
    CreateFromData(bDataPointer, iWidth, iHeight, format, bGenerateMipMaps);
    sPath = a_sPath;

    return true; // Success
}

void CTexture::SetSamplerParameter(GLenum parameter, GLenum value)
{
    glSamplerParameteri(uiSampler, parameter, value);
}

/*-----------------------------------------------

Name:	SetFiltering

Params:	tfMagnification - mag. filter, must be from
ETextureFiltering enum
tfMinification - min. filter, must be from
ETextureFiltering enum

Result:	Sets magnification and minification
texture filter.

/*---------------------------------------------*/

void CTexture::SetFiltering(int a_tfMagnification, int a_tfMinification)
{
    glBindSampler(0, uiSampler);

    // Set magnification filter
    if (a_tfMagnification == TEXTURE_FILTER_MAG_NEAREST)
        glSamplerParameteri(uiSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    else if (a_tfMagnification == TEXTURE_FILTER_MAG_BILINEAR)
        glSamplerParameteri(uiSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Set minification filter
    if (a_tfMinification == TEXTURE_FILTER_MIN_NEAREST)
        glSamplerParameteri(uiSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    else if (a_tfMinification == TEXTURE_FILTER_MIN_BILINEAR)
        glSamplerParameteri(uiSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    else if (a_tfMinification == TEXTURE_FILTER_MIN_NEAREST_MIPMAP)
        glSamplerParameteri(uiSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    else if (a_tfMinification == TEXTURE_FILTER_MIN_BILINEAR_MIPMAP)
        glSamplerParameteri(uiSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    else if (a_tfMinification == TEXTURE_FILTER_MIN_TRILINEAR)
        glSamplerParameteri(uiSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    tfMinification = a_tfMinification;
    tfMagnification = a_tfMagnification;
}

/*-----------------------------------------------

Name:	BindTexture

Params:	iTextureUnit - texture unit to bind texture to

Result:	Guess what it does :)

/*---------------------------------------------*/

void CTexture::BindTexture(int iTextureUnit)
{
    glActiveTexture(GL_TEXTURE0 + iTextureUnit);
    glBindTexture(GL_TEXTURE_2D, uiTexture);
    glBindSampler(iTextureUnit, uiSampler);
}

/*-----------------------------------------------

Name:	DeleteTexture

Params:	none

Result:	Frees all memory used by texture.

/*---------------------------------------------*/

void CTexture::DeleteTexture()
{
    glDeleteSamplers(1, &uiSampler);
    glDeleteTextures(1, &uiTexture);
}

/*-----------------------------------------------

Name:	Getters

Params:	none

Result:	... They get something :D

/*---------------------------------------------*/

int CTexture::GetMinificationFilter()
{
    return tfMinification;
}

int CTexture::GetMagnificationFilter()
{
    return tfMagnification;
}

int CTexture::GetWidth()
{
    return iWidth;
}

int CTexture::GetHeight()
{
    return iHeight;
}

int CTexture::GetBPP()
{
    return iBPP;
}

uint CTexture::GetTextureID()
{
    return uiTexture;
}

std::string CTexture::GetPath()
{
    return sPath;
}

bool CTexture::ReloadTexture()
{
    cv::Mat textureImage = cv::imread(sPath.c_str());

    unsigned char* bDataPointer = textureImage.data; // Retrieve the image data

    GLenum format;
    format = GL_BGR;
    iWidth = textureImage.cols;
    iHeight = textureImage.rows;
    glBindTexture(GL_TEXTURE_2D, uiTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, iWidth, iHeight, format, GL_UNSIGNED_BYTE, bDataPointer);
    if (bMipMapsGenerated)glGenerateMipmap(GL_TEXTURE_2D);

    return true; // Success
}
