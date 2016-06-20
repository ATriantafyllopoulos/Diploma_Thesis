#include "Viewer_GL3.h"


Viewer_GL3::Viewer_GL3(HWND hwnd)
{
	if (create(hwnd))
		init();
}

/*-----------------------------------------------

Name:	ResetTimer

Params:	none

Result:	Resets application timer (for example
after re-activation of application).

/*---------------------------------------------*/

void Viewer_GL3::ResetTimer()
{
	tLastFrame = clock();
	fFrameInterval = 0.0f;
}

/*-----------------------------------------------

Name:	UpdateTimer

Params:	none

Result:	Updates application timer.

/*---------------------------------------------*/

void Viewer_GL3::UpdateTimer()
{
	clock_t tCur = clock();
	fFrameInterval = float(tCur - tLastFrame) / float(CLOCKS_PER_SEC);
	tLastFrame = tCur;
}

Viewer_GL3::~Viewer_GL3()
{
	shader.deleteProgram();
	//delete shader;
	shVertex.deleteShader();
	shFragment.deleteShader();
	physicsEngine.unregisterResources();
	//cudaGraphicsUnregisterResource(testingVBO_CUDA);
	glDeleteBuffers(1, &testingVAO);
	wglMakeCurrent(hdc, 0); // Remove the rendering context from our device context
	wglDeleteContext(hrc); // Delete our rendering context
	ReleaseDC(hwnd, hdc); // Release the device context from our window
}

void Viewer_GL3::addToDraw(Renderable *r)
{
	models.push_back((std::shared_ptr<Renderable_GL3>)(static_cast<Renderable_GL3*>(r)));
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

	// Try and set the pixel format based on our PFD
	if (!SetPixelFormat(hdc, nPixelFormat, &pfd)) // If it fails
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

	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glPointSize(10.0);

	shVertex.loadShader("..//Shaders//main_shader.vert", GL_VERTEX_SHADER);
	shFragment.loadShader("..//Shaders//main_shader.frag", GL_FRAGMENT_SHADER);
	shLight.loadShader("..//Shaders//dirLight.frag", GL_FRAGMENT_SHADER);

	shader.createProgram();

	shader.addShaderToProgram(&shVertex);
	shader.addShaderToProgram(&shFragment);
	shader.addShaderToProgram(&shLight);

	shader.linkProgram();

	//camera init parameters
	vEye = glm::vec3(0.0f, 0.0f, 15.0f);
	vView = glm::vec3(-1.0f, 0.0f, -1.0f);
	vUp = glm::vec3(0.0f, 1.0f, 0.0f);
	fSpeed = 25.0f;
	fSensitivity = 0.01f;

	viewMatrix = glm::lookAt(vEye, vView, vUp);	//create our view matrix
	projectionMatrix = glm::perspective(45.f, (float)windowWidth / (float)windowHeight, 0.5f, 1000.f);  //Create our perspective projection matrix

	ResetTimer();

	glGenVertexArrays(1, &testingVAO);
	glBindVertexArray(testingVAO);
	//131072
	numOfParticles = 2 << 11;
	GLfloat *vertices_position = new GLfloat[3 * numOfParticles]; //changed from static to dynamic allocation

	//Create a Vector Buffer Object that will store the vertices on video memory
	GLuint vbo;
	glGenBuffers(1, &vbo);
	//GLsizeiptr size = sizeof(vertices_position)*numOfParticles * 3;
	//Allocate space and upload the data from CPU to GPU
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_position) * numOfParticles * 3, vertices_position, GL_STATIC_DRAW);
	//Get the location of the attributes that enters in the vertex shader
	GLint position_attribute = glGetAttribLocation(shader.getProgramID(), "inPosition");
	//Specify how the data for position can be accessed
	glVertexAttribPointer(position_attribute, 3, GL_FLOAT, GL_FALSE, 0, 0);
	//Enable the attribute
	glEnableVertexAttribArray(position_attribute);
	glEnable(GL_PROGRAM_POINT_SIZE);

	glBindVertexArray(0);
	size_t num_bytes = sizeof(vertices_position) * numOfParticles * 3;
	physicsEngine.registerResources(testingVAO, numOfParticles, num_bytes);
	cudaError_t cudaStatus = physicsEngine.initialize();
	if (cudaStatus != cudaSuccess)
	{
		MessageBox(hwnd, "Initialization with CUDA failed.", "Error", MB_ICONINFORMATION);
		delete[] vertices_position;
		exit(1);
	}
	else
	{
		std::cout << "Initialization with CUDA was successful." << std::endl;
	}
	glBindVertexArray(testingVAO);
	delete[] vertices_position;
}

/**
Rendering function
*/
void Viewer_GL3::render(void)
{
	//static double offset = 0.1;
	
	glViewport(0, 0, windowWidth, windowHeight); // Set the viewport size to fill the window
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); // Clear required buffers
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_DEPTH_TEST);
	projectionMatrix = glm::perspective(45.f, (float)windowWidth / (float)windowHeight, 0.5f, 1000.f);  //Create our perspective projection matrix

	//CAssimpModel::BindModelsVAO();
	for (unsigned i = 0; i < models.size(); i++)
	{
		models[i]->draw(&shader, projectionMatrix, viewMatrix, windowWidth, windowHeight);
	}
	shader.bind();
	shader.setUniform("matrices.projMatrix", projectionMatrix);
	shader.setUniform("matrices.viewMatrix", viewMatrix);

	shader.setUniform("sunLight.vColor", glm::vec3(1.f, 1.f, 1.f));
	shader.setUniform("sunLight.vDirection", glm::vec3(sqrt(2.f) / 2.f, -sqrt(2.f) / 2.f, 0.f));
	shader.setUniform("sunLight.fAmbient", 0.5f);

	shader.setUniform("gSampler", 0);
	glm::mat4 modelMatrix = glm::mat4(1.0);
	glm::mat4 normalMatrix = glm::transpose(glm::inverse(modelMatrix));
	shader.setUniform("matrices.modelMatrix", modelMatrix);
	shader.setUniform("matrices.normalMatrix", normalMatrix);
	shader.setUniform("vColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
	//glBindVertexArray(testingVAO); //avoid multiple bind calls
	

	glDrawArrays(GL_POINTS, 0, numOfParticles);
	//glDrawArrays(GL_TRIANGLES, 0, 12);

	//glBindVertexArray(0); //avoid multiple bind calls
	shader.unbind();
	cameraUpdate();
	UpdateTimer();
	cudaError_t cudaStatus;
	/*cudaStatus = physicsEngine.animate();
	if (cudaStatus != cudaSuccess)
	{
		MessageBox(hwnd, "Animating particles with CUDA failed.", "Error", MB_ICONINFORMATION);
	}*/
	cudaStatus = physicsEngine.collisionDetection();
	if (cudaStatus != cudaSuccess)
	{
		MessageBox(hwnd, "Collision detection with CUDA failed.", "Error", MB_ICONINFORMATION);
		exit(1);
	}

	//animateWithCuda(testingVBO_CUDA, &num_bytes, offset);
	//offset = -offset;
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
	float fAngle = acos(glm::dot(glm::vec3(0.f, 0.f, -1.f), vDir))*(180.0f / PI);
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
	glm::vec3 vDir2 = vDir; 
	vDir2.y = 0.0f;
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
		vMoveBy += vMove * fFrameInterval;
	if ((GetAsyncKeyState('S') >> 15) & 1)
		vMoveBy -= vMove * fFrameInterval;
	if ((GetAsyncKeyState('A') >> 15) & 1)
		vMoveBy -= vStrafe * fFrameInterval;
	if ((GetAsyncKeyState('D') >> 15) & 1)
		vMoveBy += vStrafe * fFrameInterval;

	vEye += vMoveBy; 
	vView += vMoveBy;

	if ((GetAsyncKeyState(27) >> 15) & 1)
		exit(1);

	viewMatrix = glm::lookAt(vEye, vView, vUp);	//update our view matrix
}