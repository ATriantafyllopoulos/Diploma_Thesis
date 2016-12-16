#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__
#include <glm/glm.hpp>
#include "Shader.h"
#include <iostream>
class ParticleRenderer
{
    public:
        ParticleRenderer();
        ~ParticleRenderer();

        void setPositions(float *pos, int numParticles);
        void setVertexBuffer(unsigned int vbo, int numParticles);
        void setColorBuffer(unsigned int vbo)
        {
            m_colorVBO = vbo;
        }

        enum DisplayMode
        {
            PARTICLE_POINTS,
            PARTICLE_SPHERES,
            PARTICLE_NUM_MODES
        };

        void display(DisplayMode mode = PARTICLE_POINTS);
        void displayGrid();

        void setPointSize(float size)
        {
            m_pointSize = size;
        }
        void setParticleRadius(float r)
        {
            m_particleRadius = r;
        }
        void setFOV(float fov)
        {
            m_fov = fov;
        }
        void setWindowSize(int w, int h)
        {           
            m_window_w = w;
            m_window_h = h;
            //std::cout << "Setting window size to: (" << m_window_w << ", " << m_window_h  << ")" << std::endl;
        }

		void setViewMatrix(const glm::mat4 &x){ viewMatrix = x; }
		void setProjectionMatrix(const glm::mat4 &x){ projectionMatrix = x; }

		void renderDepthImage();
		void renderARScene(int width, int height);

    protected: // methods
        void _initGL();
        void _drawPoints();
        GLuint _compileProgram(const char *vsource, const char *fsource);

    protected: // data
        float *m_pos;
        int m_numParticles;

        float m_pointSize;
        float m_particleRadius;
        float m_fov;
        int m_window_w, m_window_h;

        GLuint m_program;

        GLuint m_vbo;
        GLuint m_colorVBO;

		glm::mat4 projectionMatrix;
		glm::mat4 viewMatrix;

		CShader particleVertex, particleFragment;
		CShaderProgram particleShader;
		
		//auxiliary variables for rendering static particles
		CShaderProgram staticShader;
		CShader staticVertex, staticFragment;
		GLuint rangeSampler, rangeVAO, rangeTexture;
		int numberOfRangeData;
		GLuint quad_VertexArrayID;
		
		CShaderProgram sceneShader;
		CShader sceneVertex, sceneFragment;
	public:
		void setNumberOfRangeData(const int &x) { numberOfRangeData = x; }
		void setRangeSampler(const GLuint &x) { rangeSampler = x; }
		void setRangeVAO(const GLuint &x) { rangeVAO = x; }
		void setRangeTexture(const GLuint &x) { rangeTexture = x; }
		glm::mat4 getViewMatrix(){ return viewMatrix; }
		//void setViewMatrix(const glm::mat4 &x){ viewMatrix = x; }

};

#endif //__ RENDER_PARTICLES__
