#include "Viewer_GL3.h"


Viewer_GL3::Viewer_GL3(HWND hwnd)
{

	if (create(hwnd))
		init();

	vEye = glm::vec3(0.0f, 0.0f, 0.0f);
	vView = glm::vec3(0.0f, 0.0, -1.0f);
	vUp = glm::vec3(0.0f, 1.0f, 0.0f);
	fSpeed = 25.0f;
	fSensitivity = 0.01f;
}

Viewer_GL3::~Viewer_GL3()
{
	wglMakeCurrent(hdc, 0); // Remove the rendering context from our device context
	wglDeleteContext(hrc); // Delete our rendering context
	ReleaseDC(hwnd, hdc); // Release the device context from our window
}

void Viewer_GL3::addToDraw(Renderable *r)
{
	models.push_back(static_cast<Renderable_GL3*>(r));
}

bool Viewer_GL3::create(HWND hwnd)
{
	this->hwnd = hwnd; // Set the HWND for our window

	hdc = GetDC(hwnd); // Get the device context for our window

	PIXELFORMATDESCRIPTOR pfd; // Create a new PIXELFORMATDESCRIPTOR (PFD)
	memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR)); // Clear our  PFD
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR); // Set the size of the PFD to the size of the class
	pfd.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW; // Enable double buffering, opengl support and drawing to a window
	pfd.iPixelType = PFD_TYPE_RGBA; // Set our application to use RGBA pixels
	pfd.cColorBits = 32; // Give us 32 bits of color information (the higher, the more colors)
	pfd.cDepthBits = 32; // Give us 32 bits of depth information (the higher, the more depth levels)
	pfd.iLayerType = PFD_MAIN_PLANE; // Set the layer of the PFD

	int nPixelFormat = ChoosePixelFormat(hdc, &pfd); // Check if our PFD is valid and get a pixel format back
	if (nPixelFormat == 0) // If it fails
		return false;

	bool bResult = SetPixelFormat(hdc, nPixelFormat, &pfd); // Try and set the pixel format based on our PFD
	if (!bResult) // If it fails
		return false;

	HGLRC tempOpenGLContext = wglCreateContext(hdc); // Create an OpenGL 2.1 context for our device context
	wglMakeCurrent(hdc, tempOpenGLContext); // Make the OpenGL 2.1 context current and active

	GLenum error = glewInit(); // Enable GLEW
	if (error != GLEW_OK) // If GLEW fails
		return false;

	int attributes[] = {
		WGL_CONTEXT_MAJOR_VERSION_ARB, 3, // Set the MAJOR version of OpenGL to 3
		WGL_CONTEXT_MINOR_VERSION_ARB, 2, // Set the MINOR version of OpenGL to 2
		WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB, // Set our OpenGL context to be forward compatible
		0
	};

	if (wglewIsSupported("WGL_ARB_create_context") == 1) { // If the OpenGL 3.x context creation extension is available
		hrc = wglCreateContextAttribsARB(hdc, NULL, attributes); // Create and OpenGL 3.x context based on the given attributes
		wglMakeCurrent(NULL, NULL); // Remove the temporary context from being active
		wglDeleteContext(tempOpenGLContext); // Delete the temporary OpenGL 2.1 context
		wglMakeCurrent(hdc, hrc); // Make our OpenGL 3.0 context current
	}
	else {
		hrc = tempOpenGLContext; // If we didn't have support for OpenGL 3.x and up, use the OpenGL 2.1 context
	}

	int glVersion[2] = { -1, -1 }; // Set some default values for the version
	glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]); // Get back the OpenGL MAJOR version we are using
	glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]); // Get back the OpenGL MAJOR version we are using

	std::cout << "Using OpenGL: " << glVersion[0] << "." << glVersion[1] << std::endl; // Output which version of OpenGL we are using
	return true; // We have successfully created a context, return true
}

/*
Init function. No need for it to be a separate method.
In the future it might be incorporated to create.
*/
void Viewer_GL3::init(void)
{
	glClearColor(0.4f, 0.6f, 0.9f, 0.0f); // Set the clear color based on Microsofts CornflowerBlue (default in XNA)
	shader = new Shader("shader.vert", "shader.frag"); // Create our shader by loading our vertex and fragment shader  
}

/**
Rendering function
*/
void Viewer_GL3::render(void)
{

	glViewport(0, 0, windowWidth, windowHeight); // Set the viewport size to fill the window
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); // Clear required buffers

	//viewMatrix = glm::lookAt(glm::vec3(0, 0, 100), glm::vec3(0, 0, 0), glm::vec3(0.0f, 1.0f, 0.0f));
	viewMatrix = glm::lookAt(vEye, vView, vUp);
	projectionMatrix = glm::perspective(45.f, (float)windowWidth / (float)windowHeight, 0.5f, 1000.f);  // Create our perspective projection matrix
	//projectionMatrix = glm::ortho(-15.f, 30.f, -5.f, 60.f, 1.f, 10.f);
	cameraUpdate();
	for (unsigned i = 0; i < models.size(); i++)
	{
		models[i]->draw(shader, projectionMatrix, viewMatrix, windowWidth, windowHeight);
	}

	SwapBuffers(hdc); // Swap buffers so we can see our rendering
}

/*-----------------------------------------------

Name:	rotateWithMouse

Params:	none

Result:	Checks for moving of mouse and rotates
camera.

/*---------------------------------------------*/
void Viewer_GL3::rotateWithMouse()
{
	GetCursorPos(&pCur);
	RECT rRect; 
	GetWindowRect(hwnd, &rRect);
	int iCentX = (rRect.left + rRect.right) >> 1,
		iCentY = (rRect.top + rRect.bottom) >> 1;

	float deltaX = (float)(iCentX - pCur.x)*fSensitivity;
	float deltaY = (float)(iCentY - pCur.y)*fSensitivity;

	if (deltaX != 0.0f)
	{
		vView -= vEye;
		vView = glm::rotate(vView, deltaX, glm::vec3(0.0f, 1.0f, 0.0f));
		vView += vEye;
	}
	if (deltaY != 0.0f)
	{
		glm::vec3 vAxis = glm::cross(vView - vEye, vUp);
		vAxis = glm::normalize(vAxis);
		float fAngle = deltaY;
		float fNewAngle = fAngle + getAngleX();
		if (fNewAngle > -89.80f && fNewAngle < 89.80f)
		{
			vView -= vEye;
			vView = glm::rotate(vView, deltaY, vAxis);
			vView += vEye;
		}
	}
	SetCursorPos(iCentX, iCentY);
}

/*-----------------------------------------------

Name:	getAngleY

Params:	none

Result:	Gets Y angle of camera (head turning left
and right).

/*---------------------------------------------*/

float Viewer_GL3::getAngleY()
{
	glm::vec3 vDir = vView - vEye; vDir.y = 0.0f;
	glm::normalize(vDir);
	float fAngle = acos(glm::dot(glm::vec3(0, 0, -1), vDir))*(180.0f / PI);
	if (vDir.x < 0)fAngle = 360.0f - fAngle;
	return fAngle;
}

/*-----------------------------------------------

Name:		getAngleX

Params:	none

Result:	Gets X angle of camera (head turning up
and down).

/*---------------------------------------------*/

float Viewer_GL3::getAngleX()
{
	glm::vec3 vDir = vView - vEye;
	vDir = glm::normalize(vDir);
	glm::vec3 vDir2 = vDir; vDir2.y = 0.0f;
	vDir2 = glm::normalize(vDir2);
	float fAngle = acos(glm::dot(vDir2, vDir))*(180.0f / PI);
	if (vDir.y < 0)fAngle *= -1.0f;
	return fAngle;
}

/*-----------------------------------------------

Name:	update

Params:	none

Result:	Performs updates of camera - moving and
rotating.

/*---------------------------------------------*/
void Viewer_GL3::cameraUpdate()
{
	rotateWithMouse();

	// Get view direction
	glm::vec3 vMove = vView - vEye;
	vMove = glm::normalize(vMove);
	vMove *= fSpeed;

	glm::vec3 vStrafe = glm::cross(vView - vEye, vUp);
	vStrafe = glm::normalize(vStrafe);
	vStrafe *= fSpeed;

	int iMove = 0;
	glm::vec3 vMoveBy;
	//Get vector of move
	if ((GetAsyncKeyState('W') >> 15) & 1)
		vMoveBy += vMove;
	if ((GetAsyncKeyState('S') >> 15) & 1)
		vMoveBy -= vMove;
	if ((GetAsyncKeyState('D') >> 15) & 1)
		vMoveBy -= vStrafe;
	if ((GetAsyncKeyState('A') >> 15) & 1)
		vMoveBy += vStrafe;
	vEye += vMoveBy; vView += vMoveBy;

	if ((GetAsyncKeyState(27) >> 15) & 1)
		exit(1);
}