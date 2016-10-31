#include "Viewer_GL3.h"

static Viewer_GL3 *callBackInstance;
Viewer_GL3::Viewer_GL3(GLFWwindow* inWindow)
{
    window = inWindow;
	renderer = NULL;
    if (create())
		init();
	
    viewMode = M_VIEW;
	callBackInstance = this;
	modelMatrix = NULL;
//	showRangeData = true;
}

/*-----------------------------------------------

Name:	ResetTimer

Params:	none

Result:	Resets application timer (for example
after re-activation of application).

---------------------------------------------*/

void Viewer_GL3::ResetTimer()
{
	tLastFrame = clock();
	fFrameInterval = 0.0f;
}

/*-----------------------------------------------

Name:	UpdateTimer

Params:	none

Result:	Updates application timer.

---------------------------------------------*/

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
	//cudaGraphicsUnregisterResource(testingVBO_CUDA);
	glDeleteBuffers(1, &testingVAO);
    //wglMakeCurrent(hdc, 0); // Remove the rendering context from our device context
    //wglDeleteContext(hrc); // Delete our rendering context
    //ReleaseDC(hwnd, hdc); // Release the device context from our window
}

void Viewer_GL3::addToDraw(Renderable *r)
{
    //models.push_back((std::shared_ptr<Renderable_GL3>)(static_cast<Renderable_GL3*>(r)));
}

bool Viewer_GL3::create()
{
    return true;
}

/*
Init function. No need for it to be a separate method.
In the future it might be incorporated to create.
*/
void Viewer_GL3::init(void)
{
    glClearColor(0.25, 0.25, 0.25, 1.0);

	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	

	shVertex.loadShader("Shaders/main_shader.vert", GL_VERTEX_SHADER);
	shFragment.loadShader("Shaders/main_shader.frag", GL_FRAGMENT_SHADER);

	shader.createProgram();

	shader.addShaderToProgram(&shVertex);
	shader.addShaderToProgram(&shFragment);

	shader.linkProgram();

	//camera init parameters
	vEye = glm::vec3(0.0f, 0.0f, 0.1f);
	vView = glm::vec3(0.0f, 0.0f, -1.0f);
	vUp = glm::vec3(0.0f, 1.0f, 0.0f);
	fSpeed = 25.0f;
	fSensitivity = 0.01f;

	viewMatrix = glm::lookAt(vEye, vView, vUp);	//create our view matrix
	projectionMatrix = glm::perspective(glm::radians(45.f), (float)windowWidth / (float)windowHeight, 0.1f, 100.f);

	ResetTimer();

	glDisable(GL_PROGRAM_POINT_SIZE);
	glPointParameterf(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
    //glEnable(GL_PROGRAM_POINT_SIZE);
    //glPointSize(1.0);
    //glPointSize(10.0);
    //glEnable();
    //ShowCursor(TRUE);
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

	showRangeData = true;
    resize(windowWidth, windowHeight);
	if (renderer)
		renderer->setWindowSize(windowWidth, windowHeight);

	objModels[0].LoadModelFromFile("Data/OBjmodels/bunny.obj");
	number_of_objects = 0;
	CAssimpModel::FinalizeVBO();

}

/**
Rendering function
*/
void Viewer_GL3::render(void)
{
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    //std::cout << "Window width: " << windowWidth << " Window Height: " << windowHeight << std::endl;
    glViewport(0, 0, windowWidth, windowHeight); // Set the viewport size to fill the window
	viewport = glm::vec4(0, 0, windowWidth, windowHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); // Clear required buffers
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_DEPTH_TEST);
	projectionMatrix = glm::perspective(glm::radians(45.f), (float)windowWidth / (float)windowHeight, 0.1f, 100.f);

	cameraUpdate();
	UpdateTimer();

	renderer->setProjectionMatrix(projectionMatrix);
	if(showRangeData)
		renderer->renderDepthImage();
	if (viewMode == M_VIEW)
		renderer->setViewMatrix(viewMatrix);
	renderer->display(ParticleRenderer::PARTICLE_SPHERES);

	if (number_of_objects)
	{
		CAssimpModel::BindModelsVAO();
		shader.bind();
		shader.setUniform("matrices.viewMatrix", viewMatrix);
		for (int i = 0; i < number_of_objects; i++)
		{
			modelMatrix[i] = glm::scale(modelMatrix[i], glm::vec3(1.0, 1.0, 1.0));
			shader.setUniform("matrices.modelMatrix", modelMatrix[i]);
			objModels[0].draw(&shader);
		}

		shader.unbind();
	}
    glfwSwapBuffers(window);
    glfwPollEvents();
}

/*-----------------------------------------------

Name:	rotateWithMouse

Params:	none

Result:	Checks for moving of mouse and rotates
camera.

---------------------------------------------*/
void Viewer_GL3::rotateWithMouse()
{
//	GetCursorPos(&pCur);
//	RECT rRect;
//	GetWindowRect(hwnd, &rRect);
//	int iCentX = (rRect.left + rRect.right) >> 1,
//		iCentY = (rRect.top + rRect.bottom) >> 1;

//	float deltaX = (float)(iCentX - pCur.x)*fSensitivity;
//	float deltaY = (float)(iCentY - pCur.y)*fSensitivity;

//	if (deltaX != 0.0f)
//	{
//		vView -= vEye;
//		vView = glm::rotate(vView, deltaX, glm::vec3(0.0f, 1.0f, 0.0f));
//		vView += vEye;
//	}
//	if (deltaY != 0.0f)
//	{
//		glm::vec3 vAxis = glm::cross(vView - vEye, vUp);
//		vAxis = glm::normalize(vAxis);
//		float fAngle = deltaY;
//		float fNewAngle = fAngle + getAngleX();
//		if (fNewAngle > -89.80f && fNewAngle < 89.80f)
//		{
//			vView -= vEye;
//			vView = glm::rotate(vView, deltaY, vAxis);
//			vView += vEye;
//		}
//	}
//	SetCursorPos(iCentX, iCentY);
}

/*-----------------------------------------------

Name:	getAngleY

Params:	none

Result:	Gets Y angle of camera (head turning left
and right).

---------------------------------------------*/
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

---------------------------------------------*/
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

---------------------------------------------*/
void Viewer_GL3::cameraUpdate()
{
	//	GetCursorPos(&pCur);
	//	RECT rRect;
	//	GetWindowRect(hwnd, &rRect);
	//	int iCentX = (rRect.left + rRect.right) >> 1,
	//		iCentY = (rRect.top + rRect.bottom) >> 1;

	//	float deltaX = (float)(iCentX - pCur.x)*fSensitivity;
	//	float deltaY = (float)(iCentY - pCur.y)*fSensitivity;

	//	if (deltaX != 0.0f)
	//	{
	//		vView -= vEye;
	//		vView = glm::rotate(vView, deltaX, glm::vec3(0.0f, 1.0f, 0.0f));
	//		vView += vEye;
	//	}
	//	if (deltaY != 0.0f)
	//	{
	//		glm::vec3 vAxis = glm::cross(vView - vEye, vUp);
	//		vAxis = glm::normalize(vAxis);
	//		float fAngle = deltaY;
	//		float fNewAngle = fAngle + getAngleX();
	//		if (fNewAngle > -89.80f && fNewAngle < 89.80f)
	//		{
	//			vView -= vEye;
	//			vView = glm::rotate(vView, deltaY, vAxis);
	//			vView += vEye;
	//		}
	//	}
	//	SetCursorPos(iCentX, iCentY);
if (viewMode == M_VIEW)
{
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (state == GLFW_PRESS)
	{
		double xpos, ypos;


		glfwGetCursorPos(window, &xpos, &ypos);
		//glfwGetWindowSize(window, &windowWidth, &windowHeight);
		int iCentX;
		int iCentY;
		glfwGetWindowPos(window, &iCentX, &iCentY);
		float deltaX = (float)(iCentX + windowWidth / 2 - xpos)*fSensitivity;
		float deltaY = (float)(iCentY + windowHeight / 2 - ypos)*fSensitivity;

//		float deltaX = xpos * fSensitivity;
//		float deltaY = ypos * fSensitivity;
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

		viewMatrix = glm::lookAt(vEye, vView, vUp);	//update our view matrix
		glfwSetCursorPos(window, iCentX + windowWidth / 2, iCentY + windowHeight / 2);
	}

	// Get view direction
	glm::vec3 vMove = vView - vEye;
	vMove = glm::normalize(vMove);
	vMove *= fSpeed;

	glm::vec3 vStrafe = glm::cross(vView - vEye, vUp);
	vStrafe = glm::normalize(vStrafe);
	vStrafe *= fSpeed;

	glm::vec3 vMoveBy;

	state = glfwGetKey(window, GLFW_KEY_W);
	if (state == GLFW_PRESS)
	{
		vMoveBy += vMove * fFrameInterval;
	}
	state = glfwGetKey(window, GLFW_KEY_S);
	if (state == GLFW_PRESS)
	{
		vMoveBy -= vMove * fFrameInterval;
	}
	state = glfwGetKey(window, GLFW_KEY_A);
	if (state == GLFW_PRESS)
	{
		vMoveBy -= vStrafe * fFrameInterval;
	}
	state = glfwGetKey(window, GLFW_KEY_D);
	if (state == GLFW_PRESS)
	{
		vMoveBy += vStrafe * fFrameInterval;
	}

	vEye += vMoveBy;
	vView += vMoveBy;
	viewMatrix = glm::lookAt(vEye, vView, vUp);	//update our view matrix
}
//	if ((GetKeyState(VK_LBUTTON) & 0x100) != 0)
//	{
//		if (showCursor)
//		{
//			ShowCursor(FALSE);
//			ShowCursor(FALSE);
//			showCursor = !showCursor;
//		}
//		if (viewMode == VIEW) rotateWithMouse();
//	}
//	else
//		if (!showCursor)
//		{
//		ShowCursor(TRUE);
//		ShowCursor(TRUE);
//		showCursor = !showCursor;
//		}
		

//	// Get view direction
//	glm::vec3 vMove = vView - vEye;
//	vMove = glm::normalize(vMove);
//	vMove *= fSpeed;

//	glm::vec3 vStrafe = glm::cross(vView - vEye, vUp);
//	vStrafe = glm::normalize(vStrafe);
//	vStrafe *= fSpeed;

//	int iMove = 0;
//	glm::vec3 vMoveBy;
//	//Get vector of move
//	if ((GetAsyncKeyState('W') >> 15) & 1)
//		vMoveBy += vMove * fFrameInterval;
//	if ((GetAsyncKeyState('S') >> 15) & 1)
//		vMoveBy -= vMove * fFrameInterval;
//	if ((GetAsyncKeyState('A') >> 15) & 1)
//		vMoveBy -= vStrafe * fFrameInterval;
//	if ((GetAsyncKeyState('D') >> 15) & 1)
//		vMoveBy += vStrafe * fFrameInterval;

//	vEye += vMoveBy;
//	vView += vMoveBy;

//	if ((GetAsyncKeyState(27) >> 15) & 1)
//		exit(1);

//	viewMatrix = glm::lookAt(vEye, vView, vUp);	//update our view matrix
}

/**
Resize function. Called by virtual world after a resize event caught by the windows API.
Currently there is a propagation delay causing an unwanted effect.
*/
void Viewer_GL3::resize(GLint w, GLint h)
{
	windowWidth = w;
	windowHeight = h;
	if (renderer)
	{
		renderer->setWindowSize(w, h);
		renderer->setFOV(60.0);
	}
	
	glViewport(0, 0, windowWidth, windowHeight);
	viewport = glm::vec4(0, 0, windowWidth, windowHeight);
	projectionMatrix = glm::perspective(glm::radians(45.f), (float)windowWidth / (float)windowHeight, 0.1f, 100.f);
	shader.bind();
	//std::cout << "Program crashes @Viewer_GL3.cpp, line 307." << std::endl;
	shader.setUniform("matrices.projMatrix", projectionMatrix);
	//std::cout << "This message is not printed." << std::endl;
	shader.unbind();
}


