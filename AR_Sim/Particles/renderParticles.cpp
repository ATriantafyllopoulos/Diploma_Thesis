/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <GL/glew.h>

#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "renderParticles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0f),
      m_particleRadius(0.125f * 0.5f),
      m_program(0),
      m_vbo(0),
      m_colorVBO(0)
{
	glEnable(GL_PROGRAM_POINT_SIZE);
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    m_pos = 0;
}

void ParticleRenderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setVertexBuffer(unsigned int vbo, int numParticles)
{
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints()
{
    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
		glBindVertexArray(m_vbo); 

		glDrawArrays(GL_POINTS, 0, m_numParticles);

		glBindVertexArray(0);
    }
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */)
{
    switch (mode)
    {
        case PARTICLE_POINTS:
            glColor3f(1, 1, 1);
            glPointSize(m_pointSize);
            _drawPoints();
            break;

        default:
        case PARTICLE_SPHERES:
			particleShader.bind();
//          std::cout << "Current View Matrix is: " << std::endl;
//			for (int row = 0; row < 4; row++)
//			{
//				for (int col = 0; col < 4; col++)
//					std::cout << viewMatrix[row][col] << " ";
//				std::cout << std::endl;
//			}
//			std::cout << std::endl;
//          std::cout << std::endl;
            glEnable(GL_PROGRAM_POINT_SIZE);
			particleShader.setUniform("viewMatrix", viewMatrix);
			//particleShader.setUniform("vColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
            particleShader.setUniform("pointScale", (float)m_window_h / tanf(45 * 0.5f*(float)M_PI / 180.0f));
			particleShader.setUniform("pointRadius", m_particleRadius);

			glBindVertexArray(m_vbo);
			glDrawArrays(GL_POINTS, 0, m_numParticles);
			glBindVertexArray(0);
			particleShader.unbind();

            break;
    }
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }
    return program;
}

void ParticleRenderer::_initGL()
{
	//shader used for particle rendering
	particleVertex.loadShader("Shaders/point.vert", GL_VERTEX_SHADER);
	particleFragment.loadShader("Shaders/point.frag", GL_FRAGMENT_SHADER);
	particleShader.createProgram();
	particleShader.addShaderToProgram(&particleVertex);
	particleShader.addShaderToProgram(&particleFragment);
	particleShader.linkProgram();
	particleShader.bind();
	particleShader.setUniform("vColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
	particleShader.setUniform("pointRadius", m_particleRadius);
	particleShader.unbind();

	//initialize shader for static particles
	staticVertex.loadShader("Shaders/main_shader.vert", GL_VERTEX_SHADER);
	staticFragment.loadShader("Shaders/main_shader.frag", GL_FRAGMENT_SHADER);
	staticShader.createProgram();
	staticShader.addShaderToProgram(&staticVertex);
	staticShader.addShaderToProgram(&staticFragment);
	staticShader.linkProgram();

	staticShader.bind();
	staticShader.setUniform("matrices.projMatrix", projectionMatrix);
	staticShader.setUniform("matrices.viewMatrix", viewMatrix);
	staticShader.setUniform("sunLight.vColor", glm::vec3(1.f, 1.f, 1.f));
	staticShader.setUniform("gSampler", 0);
	glm::mat4 normalMatrix = glm::transpose(glm::inverse(glm::mat4(1.0)));
	staticShader.setUniform("matrices.modelMatrix", glm::mat4(1.0));
	staticShader.setUniform("matrices.normalMatrix", normalMatrix);
	staticShader.setUniform("vColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));//*/
    staticShader.unbind();


	sceneVertex.loadShader("Shaders/scene.vert", GL_VERTEX_SHADER);
	sceneFragment.loadShader("Shaders/scene.frag", GL_FRAGMENT_SHADER);
	sceneShader.createProgram();
	sceneShader.addShaderToProgram(&sceneVertex);
	sceneShader.addShaderToProgram(&sceneFragment);
	sceneShader.linkProgram();
	sceneShader.bind();
	sceneShader.setUniform("gSampler", 0);
	sceneShader.unbind();

	
	glGenVertexArrays(1, &quad_VertexArrayID);
	glBindVertexArray(quad_VertexArrayID);

	/*const GLfloat g_quad_vertex_buffer_data[] = {
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	1.0f, 1.0f, 0.0f,
	};*/
	/*const GLfloat quad_data[] = {
	0.0f, 0.0f, 0.0f, 0.f, 1.f,
	width, 0.0f, 0.0f, 1.f, 1.f,
	width, height, 0.0f, 1.f, 0.f,
	0.0f, 0.0f, 0.0f, 0.f, 1.f,
	0.0f, height, 0.0f, 0.f, 0.f,
	width, height, 0.0f, 1.f, 0.f,
	};*/
	//GLfloat quad_data[] = { // format = x, y, z, u, v
	//	-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	//	1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f
	//};
	int width = 480;
	int height = 640;
	const GLfloat quad_data[] = {
		-2 * width, -2 * height, 0.0f, 0.f, 1.f,
		2 * width, -2 * height, 0.0f, 1.f, 1.f,
		2 * width, 2 * height, 0.0f, 1.f, 0.f,
		-2 * width, -2 * height, 0.0f, 0.f, 1.f,
		-2 * width, 2 * height, 0.0f, 0.f, 0.f,
		2 * width, 2 * height, 0.0f, 1.f, 0.f,
	};

	GLuint quad_vertexbuffer;
	glGenBuffers(1, &quad_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad_data), quad_data, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, nullptr);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}

void ParticleRenderer::renderDepthImage()
{
	staticShader.bind();
	//staticShader.setUniform("matrices.projMatrix", projectionMatrix);
	staticShader.setUniform("matrices.viewMatrix", viewMatrix);

	//staticShader.setUniform("sunLight.vColor", glm::vec3(1.f, 1.f, 1.f));

	//staticShader.setUniform("gSampler", 0);
	//glm::mat4 normalMatrix = glm::transpose(glm::inverse(glm::mat4(1.0)));
	//staticShader.setUniform("matrices.modelMatrix", glm::mat4(1.0));
	//staticShader.setUniform("matrices.normalMatrix", normalMatrix);
	//staticShader.setUniform("vColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));//*/
	glBindTexture(GL_TEXTURE_2D, rangeTexture);
	glBindVertexArray(rangeVAO); //avoid multiple bind calls
	//glActiveTexture(rangeTexture);
	//glBindTexture(GL_TEXTURE_2D, rangeTexture);
	//glBindSampler(rangeTexture, rangeSampler);
	glDrawArrays(GL_POINTS, 0, numberOfRangeData);

	glBindVertexArray(0);
	staticShader.unbind();
}

void ParticleRenderer::renderARScene(int width, int height)
{
	/*glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rangeTexture);
	glBegin(GL_QUADS);
	glTexCoord2i(0, 0); glVertex2i(0, 0);
	glTexCoord2i(1, 0); glVertex2i(width, 0);
	glTexCoord2i(1, 1); glVertex2i(width, height);
	glTexCoord2i(0, 1); glVertex2i(0, height);
	glEnd();

	glDisable(GL_TEXTURE_2D);*/

	

	sceneShader.bind();
	glBindVertexArray(quad_VertexArrayID);
	glBindTexture(GL_TEXTURE_2D, rangeTexture);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	//glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);
	sceneShader.unbind();
}
