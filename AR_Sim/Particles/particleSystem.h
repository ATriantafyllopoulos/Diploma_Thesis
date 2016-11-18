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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0
#include "Platform.h"
#include <iostream>
#include <GL/glew.h>
//#include "../../ICPOdometry.h"
#include "../Headers/Shader.h"
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include "vector_functions.h"
#include "particleSystem.cuh"
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
//#include <FreeImage.h>
#include <time.h>
//#pragma comment(lib, "FreeImage.lib")
#include "rigidBodyKernelWrappers.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <sstream>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
//#include "opencv2/opencv_modules.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
// Particle system class

#include "Menu.h"


class ParticleSystem
{
public:
	ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
	~ParticleSystem();

	enum ParticleConfig
	{
		CONFIG_RANDOM,
		CONFIG_GRID,
		_NUM_CONFIGS
	};

	enum ParticleArray
	{
		POSITION,
		VELOCITY,
	};

	
	void reset(ParticleConfig config);

	float *getArray(ParticleArray array);
	void   setArray(ParticleArray array, const float *data, int start, int count);

	int    getNumParticles() const
	{
		return m_numParticles;
	}

	unsigned int getCurrentReadBuffer() const
	{
		return m_virtualVAO;
	}
	unsigned int getColorBuffer()       const
	{
		return m_colorVBO;
	}

	void *getCudaPosVBO()              const
	{
		return (void *)m_cudaPosVBO;
	}
	void *getCudaColorVBO()            const
	{
		return (void *)m_cudaColorVBO;
	}

	void dumpGrid();
	void dumpParticles(uint start, uint count);

	void setCollisionDetectionMethod(int x) { collisionMethod = x; }
	void setIterations(int i)
	{
		m_solverIterations = i;
	}

	void setDamping(float x)
	{
		m_params.globalDamping = x;
	}
	void setGravity(float x)
	{
		m_params.gravity = make_float3(0.0f, x, 0.0f);
	}

	void setCollideSpring(float x)
	{
		m_params.spring = x;
	}
	void setCollideDamping(float x)
	{
		m_params.damping = x;
	}
	void setCollideShear(float x)
	{
		m_params.shear = x;
	}
	void setCollideAttraction(float x)
	{
		m_params.attraction = x;
	}

	void setColliderPos(float3 x)
	{
		m_params.colliderPos = x;
	}
	void setARsimulation(const bool &x)
	{
		simulateAR = x;
	}
	float getParticleRadius()
	{
		return m_params.particleRadius;
	}
	float3 getColliderPos()
	{
		return m_params.colliderPos;
	}
	float getColliderRadius()
	{
		return m_params.colliderRadius;
	}
	uint3 getGridSize()
	{
		return m_params.gridSize;
	}
	float3 getWorldOrigin()
	{
		return m_params.worldOrigin;
	}
	float3 getCellSize()
	{
		return m_params.cellSize;
	}


	void addSphere(int index, glm::vec3 pos, glm::vec3 vel, int r, float spacing);
	void addNewSphere(int particles, glm::vec3 pos, glm::vec3 vel, int r, float spacing);
	void toggleARcollisions(void)
	{ 
		simulateAR = !simulateAR;
		if (simulateAR) std::cout << "AR collisions: ON" << std::endl;
		else std::cout << "AR collisions: OFF" << std::endl;
	}
protected: // methods
	ParticleSystem() {}
	uint createVBO(uint size);

	void _initialize(int numParticles);
	void _finalize();

	void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: // data
	int collisionMethod;
	bool m_bInitialized, m_bUseOpenGL;
	uint m_numParticles; //number of independent virtual particles
	uint m_numRigidParticles; //number of virtual particles that are part of a rigid body
	int numThreads;
	// CPU data
	float *m_hPos;              // particle positions
	float *m_hVel;              // particle velocities

	uint  *m_hParticleHash;
	uint  *m_hCellStart;
	uint  *m_hCellEnd;

	// GPU data
	float *m_dPos;
	float *m_dVel;

	float *m_dSortedPos;
	float *m_dSortedVel;

	// grid data for sorting method
	uint  *m_dGridParticleHash; // grid hash value for each particle
	uint  *m_dGridParticleIndex;// particle index for each particle
	uint  *m_dCellStart;        // index of start of each cell in sorted list
	uint  *m_dCellEnd;          // index of end of cell

	uint   m_gridSortBits;

	uint   m_virtualVAO;		// vertex array object for particle rendering
	uint   m_posVbo;            // vertex buffer object for particle positions
	uint   m_colorVBO;          // vertex buffer object for colors

	float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
	float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

	struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

	// params
	SimParams m_params;
	uint3 m_gridSize;
	uint m_numGridCells;

	StopWatchInterface *m_timer;

	uint m_solverIterations;

	//previous frame bounding box
	float3 minPos, maxPos;


protected: //depth image loading
	bool readRangeData();
	bool openImage(std::string a_sPath);
	bool loadImageToTexture(unsigned short* bData, int a_iWidth, int a_iHeight, GLenum format, bool bGenerateMipMaps);
	bool openDepthImage(std::string a_sPath);
	bool loadDepthToVBO(unsigned short* bData, int width, int height);
	void updateFrame();
	void changeTexture();
	void changeDepthMap();

	

	int imageIndex, imageOffset;
	int numberOfRangeData;
	int imageWidth, imageHeight;
	GLuint rangeSampler;
	GLuint rangeVAO;
	struct cudaGraphicsResource *cudaRangeVAO;
	GLuint imageVAO; //Vertex Array Buffer used for testing
	GLuint rangeTexture;
	CShaderProgram shader;
	CShader shVertex, shFragment;
	//glm::mat4 projectionMatrix; // Store the projection matrix  
	//glm::mat4 viewMatrix; // Store the view matrix  
	//glm::mat4 modelMatrix; // Store the model matrix
	bool simulateAR;

public:
	void update(float deltaTime);

protected:
	void updateBVH(float deltaTime);
	void updateStaticParticlesBVH(float deltaTime);

	void updateBVHSoA(float deltaTime);
	void updateStaticParticlesBVHSoA(float deltaTime);

	void updateRigidBodies(float deltaTime);
	void staticUpdateRigidBodies(float deltaTime);

	void updateGrid(float deltaTime);
	void updateUniformGrid(float deltaTime);
	void updateUniformGridDEM(float deltaTime);
	void updateStaticParticles(float deltaTime);

	void updateBVHExperimental(float deltaTime);

	void reallocGridAuxiliaries();
protected:
	float *staticPos;
	float *staticNorm;
	void initializeStaticParticles();
	
	float *staticVel;
	float *staticSortedPos;
	float *staticSortedVel;
	// grid data for sorting method
	uint  *staticGridParticleHash; // grid hash value for each particle
	uint  *staticGridParticleIndex;// particle index for each particle
	uint  *staticCellStart;        // index of start of each cell in sorted list
	uint  *staticCellEnd;          // index of end of cell

public:
	//auxiliary functions used to update variables of particle renderer
    const int getNumberOfRangeData(void) { return numberOfRangeData; }
    const GLuint getRangeSampler(void) { return rangeSampler; }
    const GLuint getRangeVAO(void) { return rangeVAO; }
    const GLuint getRangeTexture(void) { return rangeTexture; }

private:
	//rigid body associated variables
	int numRigidBodies; //number of rigid bodies
	float *rbForces, *rbPositions, *rbVelocities; //rigid body parameters
	float *rbTorque; //rigid body torque
	float *rbAngularMomentum; //rigid body angular momentum
	float *rbLinearMomentum; //cumulative linear momentum of the rigid body
	float *rbAngularVelocity; //rigid body angular velocity
	glm::vec3 *rbAngularAcceleration; //current angular acceleration due to misaligned angular momentum and velocity
	glm::mat3 *rbInertia; //rigid body inertia tensor
	glm::mat3 *rbCurrentInertia; //rigid body inertia tensor
	glm::quat *rbQuaternion; //rigid body quaternion
	float *relativePos; //particles relative positions
	int *rbIndices; //index of associated rigid body for each particle
	float *rbRadii; //radii of each sphere
	float *rbMass; //total mass of rigid body
	
	void flushAndPrintRigidBodyParameters();
	//auxiliary variables used to pre-load, and subsequently reduce, rigid body results from collision detection
	float *pForce;
	float *pTorque;
	float *pPositions;
	int *pCountARCollions; //count number of AR collisions per particle

	//CPU counter of particles for each rigid body
	bool *isRigidBody; //flag to indicate whether thrown object is a rigid body - CPU
	int *particlesPerObjectThrown;
	int objectsThrown; //denoting either rigid bodies or point sprites
public:
	//rigid body interfaces
	void initRigidSphere(int particles, glm::vec3 pos, glm::vec3 vel, int r, float spacing);
	void addRigidSphere(int particles, glm::vec3 pos, glm::vec3 vel, float r, float spacing);

	void addBunny(glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 vel = glm::vec3(0, 0, 0), glm::vec3 ang = glm::vec3(0, 0, 0), float scale = 1.5f);
	void initBunny(glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 vel = glm::vec3(0, 0, 0), glm::vec3 ang = glm::vec3(0, 0, 0), float scale = 1.5f);

	void addTeapot(glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 vel = glm::vec3(0, 0, 0), glm::vec3 ang = glm::vec3(0, 0, 0), float scale = 1.5f);
	void initTeapot(glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 vel = glm::vec3(0, 0, 0), glm::vec3 ang = glm::vec3(0, 0, 0), float scale = 1.5f);
	
	void addBanana(glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 vel = glm::vec3(0, 0, 0), glm::vec3 ang = glm::vec3(0, 0, 0), float scale = 1.5f);
	void initBanana(glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 vel = glm::vec3(0, 0, 0), glm::vec3 ang = glm::vec3(0, 0, 0), float scale = 1.5f);

	void addObj(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale, const char* modelName);

private:
	void addObject(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale, const char* modelName, int objectType);
	void initObject(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale, const char* modelName, int objectType);
	int *firstObjectIndex;
	int *objectParticleStart;
	float *objectParticlePositions;
	int objectsUsed;
	std::vector<std::string> modelNameVector;
	void addRigidBody(int previousParticleCount,
			int particlesAdded,
			float *newRelativePos, //new relative position - 4 * particlesAdded
			float *newParticleVelocity, //new particle velocity - 4 * particlesAdded
			glm::mat3 *newInverseInertia, //new inverse inertia tensor - 1
			float *newRigidBodyCM, //new rigid body center of mass - 4
			float *newRigidBodyVelocity, //new rigid body velocity - 4
			float *newRigidBodyAngularVelocity, //new rigid body angular velocity - 4
			glm::vec3 *newRigidBodyAngularAcceleration, //1
			glm::quat *newRigidBodyQuaternion, //new rigid body quaternion - 4
			float *newRigidBodyForce, //new rigid body force - 4
			float *newRigidBodyMass, //1
			float *newRigidBodyAngularMomentum, //4
			float *newRigidBodyLinearMomentum, //4
			float *newRigidBodyTorque, //4
			float *newRigidBodyRadius, //1
			float *newParticleForce, //4 * particlesAdded
			float *newParticleTorque, //4 * particlesAdded
			float *newParticlePosition, //4 * particlesAdded
			int *newCountARCollions, //particlesAdded
			int *newParticleIndex, //particlesAdded
			bool isRigidBodyLocal);
	int firstBunnyIndex;
	int bunnyParticles;
	float *bunnyRelativePositions;

	int firstTeapotIndex;
	int teapotParticles;
	float *teapotRelativePositions;

	int firstBananaIndex;
	int bananaParticles;
	float *bananaRelativePositions;
private:
	//Structure of Arrays parameters and methods

	//SoA variables reserved for virtual particles
	unsigned int *mortonCodes, *sortedMortonCodes;
	int *indices, *sortedIndices, *parentIndices, *leftIndices, *rightIndices, *minRange, *maxRange;
	AABB *bounds;
	bool *isLeaf;
	float4 *CMs;
	float *radii;
	void initializeVirtualSoA();
	
	//SoA variables reserved for real scene particles
	unsigned int *r_mortonCodes, *r_sortedMortonCodes;
	int *r_indices, *r_sortedIndices, *r_parentIndices, *r_leftIndices, *r_rightIndices, *r_minRange, *r_maxRange;
	AABB *r_bounds;
	bool *r_isLeaf;
	float4 *r_CMs;
	float *r_radii;
	void initializeRealSoA();

protected:
	//camera motion estimation variables and methods
	void CameraMotionEstimation();


    cv::Mat firstRaw;
    cv::Mat secondRaw;
    //Sophus::SE3d T_wc_prev;
    //Sophus::SE3d T_wc_curr;

	bool pauseFrame;

	//Eigen::Matrix3f cameraRotation;
	//Eigen::Vector3f cameraTranslation;
	//Eigen::Quaternionf cameraQuaternion;
	glm::mat4 cameraTransformation;
	//cv::Ptr<cv::ORB> detector;
	//cv::Mat currentFrame;
	//cv::BFMatcher matcher;
public:
	glm::mat4 getCameraTransformation(){ return cameraTransformation; }
	//Eigen::Matrix3f getCameraRotation(){ return cameraRotation; }
	//Eigen::Vector3f getCameraTranslation(){ return cameraTranslation; }
public:
	void togglePauseFrame() { pauseFrame = !pauseFrame; }

private: //introducing numerical integrator
	void integrateRigidBodyCPU_RK(float deltaTime); //simulation parameters
	float* getState();
	void setState(float* state);
public:
	void increaseARrestitution()
	{
		if (m_params.ARrestitution < 0.9) m_params.ARrestitution += 0.1;
		std::cout << "AR restitution is: " << m_params.ARrestitution << std::endl;
	}
	void decreaseARrestitution()
	{
		if (m_params.ARrestitution > 0.1) m_params.ARrestitution -= 0.1;
		std::cout << "AR restitution is: " << m_params.ARrestitution << std::endl;
	}

	void setBBox(const float3 &x, const float3 &y){ minPos = x; maxPos = y;}

	void initCPUDemo();
	void initCPU();
	void updateCPU(const float &dt);
private:
	//CPU physics engine to test rigid body simulation
	void collisionsCPU();
	void findWallCollisions();
	void findParticleCollisions();
	bool testParticleCollision(
			const float4 &p1,
			const float4 &p2,
			const float &r1,
			const float &r2,
			float4 &CM1); //if they collide this will be moved
	void findExactContactPoint(
			const float4 &p1,
			const float4 &p2,
			const float &r1,
			const float &r2,
			float4 &cp,
			float4 &cn);
	float computeImpulseMagnitude(
			const float4 &v1, const float4 &v2,
			const float4 &w1, const float4 &w2,
			const float4 &r1, const float4 &r2,
			const glm::mat3 &Iinv1, const glm::mat3 &Iinv2,
			const float &m1, const float &m2,
			const float4 &n);
	float computeImpulseMagnitude(
			const float4 &v1,
			const float4 &w1,
			const float4 &r1,
			const glm::mat3 &IinvA,
			const float &mA,
			const float4 &n);
	void integrateCPU(const float &dt);
	void integrateRigidBody(
			const float &dt,
			const float &r,
			const float &m,
			float4 &linearMomentum,
			float4 &linearVelocity,
			float4 &angularMomentum,
			float4 &angularVelocity,
			float4 &massCenter,
			float4 &force,
			float4 &torque,
			glm::mat3 &IinvCurrent,
			glm::mat3 &IinvOriginal,
			glm::quat &quaternion);
	bool CheckWallCollisions(
			const float &radius,
			const float4 &CM);
	bool HandleWallCollisions(
			const float &radius, //particle radius
			const float &m,
			const float4 &p,
			const float4 &disp,
			float4 &v,
			float4 &w,
			const glm::mat3 &Iinv,
			float4 &CM,
			float4 &force);
	void WallCollision(
		const float &radius, // particle radius
		const float &m, // rigid body mass
		const float4 &disp, // particle displacement
		const glm::mat3 &Iinv, // rigid body inverse inertia matrix
		const float4 &p, // old particle position
		float4 &n, // wall normal
		float3 &wallPos, // wall plane
		float4 &CM, // rigid body center of mass
		float4 &force, // total force acting on rigid body
		float4 &v, // rigid body linear velocity
		float4 &w);  // rigid body angular velocity

	//rigid body variables
	float4 *CM_CPU;
	float4 *V_CPU;
	float4 *W_CPU;
	float4 *P_CPU;
	float4 *L_CPU;
	float4 *F_CPU;
	float4 *T_CPU;
	glm::mat3 *Iinv_CPU;
	glm::quat *Q_CPU;

	//particle variables
	float4 *COL_CPU;
	float4 *POS_CPU;
	float4 *displacement;
	int *indexRB;

	//assuming that we only use bunnies for the CPU simulation
	float bunnyRadius;
	float bunnyMass;
	glm::mat3 bunnyInertia;

private:
	// GPU physics engine
	void Integrate_RB_System(float deltaTime);
	void Handle_Wall_Collisions();
	void Find_Rigid_Body_Collisions_Uniform_Grid();
	void Find_And_Handle_Rigid_Body_Collisions_Uniform_Grid_DEM();
	void Handle_Rigid_Body_Collisions_Baraff_GPU();
	void Handle_Rigid_Body_Collisions_Baraff_CPU();

	// augmented reality engine
	void Find_Augmented_Reality_Collisions_Uniform_Grid();
	void Find_And_Handle_Augmented_Reality_Collisions_Uniform_Grid_DEM();
	void Handle_Augmented_Reality_Collisions_Baraff_GPU();
	void Handle_Augmented_Reality_Collisions_Baraff_CPU();
	void Handle_Augmented_Reality_Collisions_Catto_CPU();

	bool Uniform_Grid_Initialized;
	// BVH collision detection
	void Find_Rigid_Body_Collisions_BVH();
	void Find_Augmented_Reality_Collisions_BVH();
	// make these part of the class so they can be used between functions
	// compatible with old code as they will be overriden locally
	float *dPos;
	float *dCol;

	// contact info

	// per particle auxiliaries
	// ATM: one per particle
	// (most important contact only)

	// index of rigid body of contact
	int *collidingRigidBodyIndex;
	// index of particle of contact
	int *collidingParticleIndex;
	// penetration distance
	float *contactDistance;

public:
	void setSceneAABB(const float3 &minP, const float3 &maxP) { minPos = minP; maxPos = maxP; }
	void addMolecule(glm::vec3 pos, glm::vec3 vel);

	glm::mat4 *getModelMatrixArray(){ return modelMatrix; }
	const int &getNumberOfObjects(){ return numRigidBodies; }
private:
	glm::mat4 *modelMatrix; // pointer to model matrix array
	glm::quat *cumulativeQuaternion;

private: // universal addition methods
	void newAddObj(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale, const char* modelName);
	void newInitObject(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale, const char* modelName, int objectType);
	void newCopyObject(glm::vec3 pos, glm::vec3 vel, glm::vec3 ang, float scale, const char* modelName, int objectType);
	void newAddRigidBody();

public: // GLFW menu
	void toggleGravity();
	void changeSpring(const float &x);
	void changeDamping(const float &x);
	void changeGlobalDamping(const float &x);
	void changeShear(const float &x);
};

#endif // __PARTICLESYSTEM_H__
