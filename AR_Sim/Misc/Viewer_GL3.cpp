#include "Viewer_GL3.h"

static Viewer_GL3 *callBackInstance;

Viewer_GL3 *Viewer_GL3::instance = NULL;



Viewer_GL3::Viewer_GL3(GLFWwindow* inWindow)
{
	instance = this;
    window = inWindow;
	renderer = NULL;
    if (create())
		init();
	
    viewMode = M_VIEW;
	callBackInstance = this;
	modelMatrix = NULL;
	scaleFactor = NULL;
	objectTypeArray = NULL;
//	showRangeData = true;
}

float Viewer_GL3::getPixelDepth(int x, int y)
{
	float depth;
	glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
	return depth;
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
	shVertex.deleteShader();
	shFragment.deleteShader();
	glDeleteBuffers(1, &testingVAO);
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
	

	shVertex.loadShader("Shaders/objShader.vert", GL_VERTEX_SHADER);
	shFragment.loadShader("Shaders/objShader.frag", GL_FRAGMENT_SHADER);

	shader.createProgram();

	shader.addShaderToProgram(&shVertex);
	shader.addShaderToProgram(&shFragment);

	shader.linkProgram();

	shader.bind();
	shader.setUniform("sunLight.vColor", glm::vec3(1.f, 1.f, 1.f));
	shader.setUniform("gSampler", 0);
	shader.setUniform("vColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

	shader.setUniform("sunLight.vColor", glm::vec3(1.0f, 1.0f, 1.0f)); // color 
	shader.setUniform("sunLight.vDirection", glm::vec3(sqrt(2.0f) / 2, -sqrt(2.0f) / 2, 0)); // direction
	shader.setUniform("sunLight.fAmbient", 0.5f); // ambient

	shader.unbind();

	// camera init parameters
	/*vEye = glm::vec3(0.0f, 0.0f, 0.1f);
	vView = glm::vec3(0.0f, -0.5f, -1.0f);
	vUp = glm::vec3(0.0f, 1.0f, 0.0f);*/
	vEye = glm::vec3(0.0f, 0.0f, 0.0f);
	vView = glm::vec3(0.0f, 0.0f, -1.0f);
	vUp = glm::vec3(0.0f, 1.0f, 0.0f);
	//vEye = glm::vec3(-0.3f, 0.7f, 1.8f);
	/*vEye = glm::vec3(0.0f, 0.7f, 1.8f);
	vView = glm::vec3(0.0f, 0.0f, -1.0f);
	vUp = glm::vec3(0.0f, 1.0f, 0.0f);*/
	/*vEye = glm::vec3(0.0f, 1.8f, 2.5f);
	vView = glm::vec3(0.0f, 0.0f, 0.0f);
	vUp = glm::vec3(0.0f, 1.0f, 0.0f);*/
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
	objModels[1].LoadModelFromFile("Data/OBjmodels/teapot.obj");
	objModels[2].LoadModelFromFile("Data/OBjmodels/banana.obj");
	objModels[3].LoadModelFromFile("Data/OBjmodels/cube.obj");
	number_of_objects = 0;

	CAssimpModel::FinalizeVBO();
	
}

/*
* Add new scaling factor. Re-allocate scaling factor array.
* This must be explicitly called every time a new object is added for rendering.
* If it is not called there will be an error in renderer.
*/
void Viewer_GL3::addScaleFactor(const float &newFactor)
{
	glm::vec3 newScaleFactor(newFactor, newFactor, newFactor);
	glm::vec3 *newScaleArray = new glm::vec3[number_of_objects];
	memcpy(newScaleArray, scaleFactor, sizeof(glm::vec3) * (number_of_objects - 1));
	newScaleArray[number_of_objects - 1] = newScaleFactor;
	if (scaleFactor)
		delete scaleFactor;
	scaleFactor = newScaleArray;
}

void Viewer_GL3::addObjectType(const int &type)
{
	int *newObjectTypeArray = new int[number_of_objects];
	memcpy(newObjectTypeArray, objectTypeArray, sizeof(int) * (number_of_objects - 1));
	newObjectTypeArray[number_of_objects - 1] = type;
	if (objectTypeArray)
		delete objectTypeArray;
	objectTypeArray = newObjectTypeArray;
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
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_DEPTH_TEST);
	if (showRangeData)
		renderer->renderARScene(windowWidth, windowHeight);

	glEnable(GL_DEPTH_TEST);
	projectionMatrix = glm::perspective(glm::radians(45.f), (float)windowWidth / (float)windowHeight, 0.1f, 100.f);

	if (viewMode == M_VIEW)
		cameraUpdate();
	UpdateTimer();

	renderer->setProjectionMatrix(projectionMatrix);
	
	renderer->setViewMatrix(viewMatrix);
	
	//renderer->display(ParticleRenderer::PARTICLE_SPHERES);
	if (showRangeData)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	}
	else
	{
		glDisable(GL_BLEND);
	}
	renderer->renderDepthImage();
	if (number_of_objects)// (number_of_objects)
	{
		CAssimpModel::BindModelsVAO();
		shader.bind();
		shader.setUniform("matrices.viewMatrix", viewMatrix);
		for (int i = 0; i < number_of_objects; i++)
		{
			glm::mat4 model = glm::scale(modelMatrix[i], scaleFactor[i]);
			glm::mat4 normalMatrix = glm::transpose(glm::inverse(modelMatrix[i]));
			shader.setUniform("matrices.normalMatrix", normalMatrix);
			shader.setUniform("matrices.modelMatrix", model);
			switch (objectTypeArray[i])
			{
			case M_BUNNY:
				glBindTexture(GL_TEXTURE_2D, 3);
				objModels[0].RenderModel();
				break;
			case M_TEAPOT:
				glBindTexture(GL_TEXTURE_2D, 5);
				objModels[1].RenderModel();
				break;
			case M_BANANA:
				glBindTexture(GL_TEXTURE_2D, 2);
				objModels[2].RenderModel();
				break;
			case M_CUBE:
				glBindTexture(GL_TEXTURE_2D, 4);
				objModels[3].RenderModel();
				break;
			}
		}

		shader.unbind();
	}
	
    glfwSwapBuffers(window);
    glfwPollEvents();
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


