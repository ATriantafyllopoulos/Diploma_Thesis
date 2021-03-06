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

#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"


#include "BVHcreation.h"

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
m_bInitialized(false),
m_bUseOpenGL(bUseOpenGL),
m_numParticles(numParticles),
m_hPos(0),
m_hVel(0),
m_dPos(0),
m_dVel(0),
staticPos(0),
staticNorm(0),
staticVel(0),
staticSortedPos(0),
staticSortedVel(0),
staticGridParticleHash(0),
staticGridParticleIndex(0),
sortedStaticGridParticleHash(0),
sortedStaticGridParticleIndex(0),
staticCellStart(0),
staticCellEnd(0),
m_gridSize(gridSize),
m_timer(NULL),
m_solverIterations(1),
numRigidBodies(0),
numThreads(128)
{
    std::cout << "Attempting to initialize particle system." << std::endl;
	//rbRadii = NULL;
	rbMass = NULL;
	relativePos = NULL;
	rbIndices = NULL;
	//rbForces = NULL;
	//rbTorque = NULL;
	rbVelocities = NULL;
	rbPositions = NULL;
	rbAngularVelocity = NULL;
	//rbAngularMomentum = NULL; 
	//rbLinearMomentum = NULL;
	//rbAngularAcceleration = NULL;
	rbInertia = NULL;
	rbCurrentInertia = NULL;
	rbQuaternion = NULL;
	pForce = NULL;
	//pPositions = NULL;
	pTorque = NULL;
	pCountARCollions = NULL;
	particlesPerObjectThrown = NULL;
	isRigidBody = NULL;

	objectsThrown = 0;
	mortonCodes = NULL;
	sortedMortonCodes = NULL;
	indices = NULL;
	sortedIndices = NULL;
	parentIndices = NULL;
	leftIndices = NULL;
	rightIndices = NULL;
	minRange = NULL;
	maxRange = NULL;
	bounds = NULL;
	isLeaf = NULL;
	CMs = NULL;
	radii = NULL;

	r_mortonCodes = NULL;
	r_sortedMortonCodes = NULL;
	r_indices = NULL;
	r_sortedIndices = NULL;
	r_parentIndices = NULL;
	r_leftIndices = NULL;
	r_rightIndices = NULL;
	r_minRange = NULL;
	r_maxRange = NULL;
	r_bounds = NULL;
	r_isLeaf = NULL;
	r_CMs = NULL;
	r_radii = NULL;

	collidingRigidBodyIndex = NULL;
	collidingParticleIndex = NULL;
	contactDistance = NULL;
	contact_normal = NULL;

	m_hVel = NULL;
	m_hCellStart = NULL;
	m_hCellEnd = NULL;

	modelMatrix = NULL;
	cumulativeQuaternion = NULL;

	firstObjectIndex = NULL;
	objectParticlePositions = NULL;
	objectParticleStart = NULL;
	objectsUsed = 0;

	minPos.x = 10000.f;
	minPos.y = 10000.f;
	minPos.z = 10000.f;
	maxPos.x = -10000.f;
	maxPos.y = -10000.f;
	maxPos.z = -10000.f;

	m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	//    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 18;    // increase this for larger grids

	// set simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = m_numParticles;

	//m_params.particleRadius = 1.0f / 80.0f;
	m_params.particleRadius = 1.0f / 64.0f;
	m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	m_params.colliderRadius = 0.2f;

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	//    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
	float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;

	m_params.gravity = make_float3(0.0f, -0.03f, 0.0f);
	m_params.globalDamping = 0.9f;
	simulateAR = true;
	m_params.ARrestitution = 0.5;
	m_params.RBrestitution = 1;
	m_params.ARfriction = 0.3;
	m_params.RBfriction = 0.1;
	m_params.Wdamping = 0.9999;
	m_params.Vdamping = 0.9999;

	imageIndex = 1;
	imageOffset = 1;

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;//0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;
	m_params.gravity = make_float3(0.0f, 0.0f, 0.0f);
	m_params.globalDamping = 1.0f;

	shVertex.loadShader("Shaders/main_shader.vert", GL_VERTEX_SHADER);
	shFragment.loadShader("Shaders/main_shader.frag", GL_FRAGMENT_SHADER);
	shader.createProgram();
	shader.addShaderToProgram(&shVertex);
	shader.addShaderToProgram(&shFragment);
	shader.linkProgram();
	cameraTransformation = glm::mat4(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f);

	
	readRangeData();
    collisionMethod = M_UNIFORM_GRID;
	if (numParticles)
	{
		_initialize(numParticles);
		initializeVirtualSoA(); //initialize SoA variables for virtual particles
	}
	pauseFrame = true;

    //T_wc_prev = Sophus::SE3d();
    //T_wc_curr = Sophus::SE3d();

    firstBunnyIndex = -1;
    bunnyRelativePositions = NULL;
    bunnyParticles = 0;
    firstTeapotIndex = -1;
    teapotParticles = 0;
    teapotRelativePositions = NULL;
	firstBananaIndex = -1;
	bananaParticles = 0;
	bananaRelativePositions = NULL;
    std::cout << "Particle system initialized successfully." << std::endl;
}

ParticleSystem::~ParticleSystem()
{
	_finalize();
	m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

//inline float lerp(float a, float b, float t)
//{
//    return a + t*(b-a);
//}

// create a color ramp
void colorRamp(float t, float *r)
{
	const int ncolors = 7;
	float c[ncolors][3] =
	{
		{ 1.0, 0.0, 0.0, },
		{ 1.0, 0.5, 0.0, },
		{ 1.0, 1.0, 0.0, },
		{ 0.0, 1.0, 0.0, },
		{ 0.0, 1.0, 1.0, },
		{ 0.0, 0.0, 1.0, },
		{ 1.0, 0.0, 1.0, },
	};
	t = t * (ncolors - 1);
	int i = (int)t;
	float u = t - floor(t);
	r[0] = lerp(c[i][0], c[i + 1][0], u);
	r[1] = lerp(c[i][1], c[i + 1][1], u);
	r[2] = lerp(c[i][2], c[i + 1][2], u);
	r[0] = 0;
	r[1] = 0;
	r[2] = 1;
}

void
ParticleSystem::_initialize(int numParticles)
{
	
	assert(!m_bInitialized);

	m_numParticles = numParticles;

	// allocate host storage
	m_hPos = new float[m_numParticles * 4];
	m_hVel = new float[m_numParticles * 4];
	memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

	// allocate GPU data
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	if (m_bUseOpenGL)
	{
		//m_posVbo = createVBO(memSize);
		glGenVertexArrays(1, &m_virtualVAO);
		glBindVertexArray(m_virtualVAO);

		glGenBuffers(1, &m_posVbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
		glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

		glGenBuffers(1, &m_colorVBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
		glBufferData(GL_ARRAY_BUFFER, memSize, NULL, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), 0);

		glBindVertexArray(0);
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
	}
	else
	{
		checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize));
	}

	allocateArray((void **)&m_dVel, memSize);

	allocateArray((void **)&m_dSortedPos, memSize);
	allocateArray((void **)&m_dSortedVel, memSize);

	allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
	allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

	allocateArray((void **)&m_dSortedGridParticleHash, m_numParticles*sizeof(uint));
	allocateArray((void **)&m_dSortedGridParticleIndex, m_numParticles*sizeof(uint));

	allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
	allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

	sdkCreateTimer(&m_timer);

	setParameters(&m_params);

	m_bInitialized = true;
	//initRigidSphere(glm::vec3(0, 0, 0), glm::vec3(0, 0, -0.1), 10, m_params.particleRadius * 2.0f);
	//addSphere(0, glm::vec3(0, 0, 0), glm::vec3(0, 0, -0.1), 10, m_params.particleRadius * 2.0f);
}

void ParticleSystem::initializeStaticParticles()
{
	unsigned int memSize = sizeof(float) * 4 * numberOfRangeData;

	if (staticPos)freeArray(staticPos);
	if (staticNorm)freeArray(staticNorm);
	if (staticVel)freeArray(staticVel);
	if (staticSortedVel)freeArray(staticSortedVel);
	if (staticSortedPos)freeArray(staticSortedPos);
	if (staticGridParticleHash)freeArray(staticGridParticleHash);
	if (staticGridParticleIndex)freeArray(staticGridParticleIndex);
	if (sortedStaticGridParticleHash)freeArray(sortedStaticGridParticleHash);
	if (sortedStaticGridParticleIndex)freeArray(sortedStaticGridParticleIndex);
	if (staticCellStart)freeArray(staticCellStart);
	if (staticCellEnd)freeArray(staticCellEnd);

	allocateArray((void **)&staticPos, memSize);
	allocateArray((void **)&staticNorm, memSize);
	allocateArray((void **)&staticVel, memSize);
	allocateArray((void **)&staticSortedVel, memSize);
	allocateArray((void **)&staticSortedPos, memSize);
	allocateArray((void **)&staticGridParticleHash, numberOfRangeData*sizeof(uint));
	allocateArray((void **)&staticGridParticleIndex, numberOfRangeData*sizeof(uint));
	allocateArray((void **)&sortedStaticGridParticleHash, numberOfRangeData*sizeof(uint));
	allocateArray((void **)&sortedStaticGridParticleIndex, numberOfRangeData*sizeof(uint));
	allocateArray((void **)&staticCellStart, m_numGridCells*sizeof(uint));
	allocateArray((void **)&staticCellEnd, m_numGridCells*sizeof(uint));

}

void ParticleSystem::reallocGridAuxiliaries()
{
	//freeArray(m_dVel);
	if(m_dSortedPos)checkCudaErrors(cudaFree(m_dSortedPos));
	if(m_dSortedVel)checkCudaErrors(cudaFree(m_dSortedVel));
	if(m_dGridParticleHash)checkCudaErrors(cudaFree(m_dGridParticleHash));
	if(m_dGridParticleIndex)checkCudaErrors(cudaFree(m_dGridParticleIndex));
	if(m_dSortedGridParticleHash)checkCudaErrors(cudaFree(m_dSortedGridParticleHash));
	if(m_dSortedGridParticleIndex)checkCudaErrors(cudaFree(m_dSortedGridParticleIndex));
	if(m_dCellStart)checkCudaErrors(cudaFree(m_dCellStart));
	if(m_dCellEnd)checkCudaErrors(cudaFree(m_dCellEnd));

	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	//allocateArray((void **)&m_dVel, memSize);
	allocateArray((void **)&m_dSortedPos, memSize);
	allocateArray((void **)&m_dSortedVel, memSize);
	allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
	allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));
	allocateArray((void **)&m_dSortedGridParticleHash, m_numParticles*sizeof(uint));
	allocateArray((void **)&m_dSortedGridParticleIndex, m_numParticles*sizeof(uint));
	allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
	allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));
}

void
ParticleSystem::_finalize()
{
	assert(m_bInitialized);
	
	if (m_hPos)delete[] m_hPos;
	if (m_hVel)delete[] m_hVel;
	if (m_hCellStart)delete[] m_hCellStart;
	if (m_hCellEnd)delete[] m_hCellEnd;

	if (m_dVel)cudaFree(m_dVel);
	if(m_dSortedPos)checkCudaErrors(cudaFree(m_dSortedPos));
	if(m_dSortedVel)checkCudaErrors(cudaFree(m_dSortedVel));
	if(m_dGridParticleHash)checkCudaErrors(cudaFree(m_dGridParticleHash));
	if(m_dGridParticleIndex)checkCudaErrors(cudaFree(m_dGridParticleIndex));
	if(m_dSortedGridParticleHash)checkCudaErrors(cudaFree(m_dSortedGridParticleHash));
	if(m_dSortedGridParticleIndex)checkCudaErrors(cudaFree(m_dSortedGridParticleIndex));
	if(m_dCellStart)checkCudaErrors(cudaFree(m_dCellStart));
	if(m_dCellEnd)checkCudaErrors(cudaFree(m_dCellEnd));

	if (m_bUseOpenGL)
	{
		unregisterGLBufferObject(m_cuda_posvbo_resource);
		glDeleteBuffers(1, (const GLuint *)&m_posVbo);
		glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
	}
	else
	{
		checkCudaErrors(cudaFree(m_cudaPosVBO));
		checkCudaErrors(cudaFree(m_cudaColorVBO));
	}
	if (staticPos)checkCudaErrors(cudaFree(staticPos));
	if (staticNorm)checkCudaErrors(cudaFree(staticNorm));
	if (staticVel)checkCudaErrors(cudaFree(staticVel));
	if (staticSortedVel)checkCudaErrors(cudaFree(staticSortedVel));
	if (staticSortedPos)checkCudaErrors(cudaFree(staticSortedPos));
	if (staticGridParticleHash)checkCudaErrors(cudaFree(staticGridParticleHash));
	if (staticGridParticleIndex)checkCudaErrors(cudaFree(staticGridParticleIndex));

	if (sortedStaticGridParticleHash)checkCudaErrors(cudaFree(sortedStaticGridParticleHash));
	if (sortedStaticGridParticleIndex)checkCudaErrors(cudaFree(sortedStaticGridParticleIndex));

	if (staticCellStart)checkCudaErrors(cudaFree(staticCellStart));
	if (staticCellEnd)checkCudaErrors(cudaFree(staticCellEnd));
	//cudaFree everything
	if (mortonCodes)cudaFree(mortonCodes);
	if (sortedMortonCodes)cudaFree(sortedMortonCodes);
	if (indices)cudaFree(indices);
	if (sortedIndices)cudaFree(sortedIndices);
	if (parentIndices)cudaFree(parentIndices);
	if (leftIndices)cudaFree(leftIndices);
	if (rightIndices)cudaFree(rightIndices);
	if (radii)cudaFree(radii);
	if (minRange)cudaFree(minRange);
	if (maxRange)cudaFree(maxRange);
	if (bounds)cudaFree(bounds);
	if (isLeaf)cudaFree(isLeaf);
	if (CMs)cudaFree(CMs);
	if (r_mortonCodes)cudaFree(r_mortonCodes);
	if (r_sortedMortonCodes)cudaFree(r_sortedMortonCodes);
	if (r_indices)cudaFree(r_indices);
	if (r_sortedIndices)cudaFree(r_sortedIndices);
	if (r_parentIndices)cudaFree(r_parentIndices);
	if (r_leftIndices)cudaFree(r_leftIndices);
	if (r_rightIndices)cudaFree(r_rightIndices);
	if (r_radii)cudaFree(r_radii);
	if (r_minRange)cudaFree(r_minRange);
	if (r_maxRange)cudaFree(r_maxRange);
	if (r_bounds)cudaFree(r_bounds);
	if (r_isLeaf)cudaFree(r_isLeaf);
	if (r_CMs)cudaFree(r_CMs);
}

void ParticleSystem::Empty_Particle_System()
{
	assert(m_bInitialized);

	if (m_hPos)delete[] m_hPos;
	if (m_hVel)delete[] m_hVel;
	if (m_hCellStart)delete[] m_hCellStart;
	if (m_hCellEnd)delete[] m_hCellEnd;

	if (m_dVel){checkCudaErrors(cudaFree(m_dVel)); m_dVel = NULL;}
	if (m_dSortedPos){ checkCudaErrors(cudaFree(m_dSortedPos)); m_dSortedPos = NULL; }
	if (m_dSortedVel){ checkCudaErrors(cudaFree(m_dSortedVel)); m_dSortedVel = NULL; }
	if (m_dGridParticleHash){ checkCudaErrors(cudaFree(m_dGridParticleHash)); m_dGridParticleHash = NULL; }
	if (m_dGridParticleIndex){ checkCudaErrors(cudaFree(m_dGridParticleIndex)); m_dGridParticleIndex = NULL; }
	if (m_dSortedGridParticleHash){ checkCudaErrors(cudaFree(m_dSortedGridParticleHash)); m_dSortedGridParticleHash = NULL; }
	if (m_dSortedGridParticleIndex){ checkCudaErrors(cudaFree(m_dSortedGridParticleIndex)); m_dSortedGridParticleIndex = NULL; }
	if (m_dCellStart){ checkCudaErrors(cudaFree(m_dCellStart)); m_dCellStart = NULL; }
	if (m_dCellEnd){ checkCudaErrors(cudaFree(m_dCellEnd)); m_dCellEnd = NULL; }

	if (m_bUseOpenGL)
	{
		unregisterGLBufferObject(m_cuda_posvbo_resource);
		glDeleteBuffers(1, (const GLuint *)&m_posVbo);
		glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
	}
	else
	{
		checkCudaErrors(cudaFree(m_cudaPosVBO));
		checkCudaErrors(cudaFree(m_cudaColorVBO));
	}
	if (staticPos){ checkCudaErrors(cudaFree(staticPos)); staticPos = NULL; }
	if (staticNorm){ checkCudaErrors(cudaFree(staticNorm)); staticNorm = NULL; }
	if (staticVel){ checkCudaErrors(cudaFree(staticVel)); staticVel = NULL; }
	if (staticSortedVel){ checkCudaErrors(cudaFree(staticSortedVel)); staticSortedVel = NULL; }
	if (staticSortedPos){ checkCudaErrors(cudaFree(staticSortedPos)); staticSortedPos = NULL; }
	if (staticGridParticleHash){ checkCudaErrors(cudaFree(staticGridParticleHash)); staticGridParticleHash = NULL; }
	if (staticGridParticleIndex){ checkCudaErrors(cudaFree(staticGridParticleIndex)); staticGridParticleIndex = NULL; }

	if (sortedStaticGridParticleHash){ checkCudaErrors(cudaFree(sortedStaticGridParticleHash)); sortedStaticGridParticleHash = NULL; }
	if (sortedStaticGridParticleIndex){ checkCudaErrors(cudaFree(sortedStaticGridParticleIndex)); sortedStaticGridParticleIndex = NULL; }

	if (staticCellStart){checkCudaErrors(cudaFree(staticCellStart)); staticCellStart = NULL;}
	if (staticCellEnd){ checkCudaErrors(cudaFree(staticCellEnd)); staticCellEnd = NULL; }
	//cudaFree everything
	if (r_mortonCodes){ checkCudaErrors(cudaFree(r_mortonCodes)); r_mortonCodes = NULL; }
	if (r_sortedMortonCodes){ checkCudaErrors(cudaFree(r_sortedMortonCodes)); r_sortedMortonCodes = NULL; }
	if (r_indices){ checkCudaErrors(cudaFree(r_indices)); r_indices = NULL; }
	if (r_sortedIndices){ checkCudaErrors(cudaFree(r_sortedIndices)); r_sortedIndices = NULL; }
	if (r_parentIndices){ checkCudaErrors(cudaFree(r_parentIndices)); r_parentIndices = NULL; }
	if (r_leftIndices){ checkCudaErrors(cudaFree(r_leftIndices)); r_leftIndices = NULL; }
	if (r_rightIndices){ checkCudaErrors(cudaFree(r_rightIndices)); r_rightIndices = NULL; }
	if (r_radii){ checkCudaErrors(cudaFree(r_radii)); r_radii = NULL; }
	if (r_minRange){ checkCudaErrors(cudaFree(r_minRange)); r_minRange = NULL; }
	if (r_maxRange){ checkCudaErrors(cudaFree(r_maxRange)); r_maxRange = NULL; }
	if (r_bounds){ checkCudaErrors(cudaFree(r_bounds)); r_bounds = NULL; }
	if (r_isLeaf){ checkCudaErrors(cudaFree(r_isLeaf)); r_isLeaf = NULL; }
	if (r_CMs){ checkCudaErrors(cudaFree(r_CMs)); r_CMs = NULL; }

	if (mortonCodes){ checkCudaErrors(cudaFree(mortonCodes)); mortonCodes = NULL; }
	if (sortedMortonCodes){ checkCudaErrors(cudaFree(sortedMortonCodes)); sortedMortonCodes = NULL; }
	if (indices){ checkCudaErrors(cudaFree(indices)); indices = NULL; }
	if (sortedIndices){ checkCudaErrors(cudaFree(sortedIndices)); sortedIndices = NULL; }
	if (parentIndices){ checkCudaErrors(cudaFree(parentIndices)); parentIndices = NULL; }
	if (leftIndices){ checkCudaErrors(cudaFree(leftIndices)); leftIndices = NULL; }
	if (rightIndices){ checkCudaErrors(cudaFree(rightIndices)); rightIndices = NULL; }
	if (radii){ checkCudaErrors(cudaFree(radii)); radii = NULL; }
	if (minRange){ checkCudaErrors(cudaFree(minRange)); minRange = NULL; }
	if (maxRange){ checkCudaErrors(cudaFree(maxRange)); maxRange = NULL; }
	if (bounds){ checkCudaErrors(cudaFree(bounds)); bounds = NULL; }
	if (isLeaf){ checkCudaErrors(cudaFree(isLeaf)); isLeaf = NULL; }
	if (CMs){ checkCudaErrors(cudaFree(CMs)); CMs = NULL; }
	if (collidingRigidBodyIndex){ checkCudaErrors(cudaFree(collidingRigidBodyIndex)); collidingRigidBodyIndex = NULL; }
	if (collidingParticleIndex){ checkCudaErrors(cudaFree(collidingParticleIndex)); collidingParticleIndex = NULL; }
	if (contactDistance){ checkCudaErrors(cudaFree(contactDistance)); contactDistance = NULL; }
	if (contact_normal){ checkCudaErrors(cudaFree(contact_normal)); contact_normal = NULL; }

	if (m_dPos){ checkCudaErrors(cudaFree(m_dPos)); m_dPos = NULL; }
	if (m_dVel){ checkCudaErrors(cudaFree(m_dVel)); m_dVel = NULL; }
	if (m_dGridParticleHash){ checkCudaErrors(cudaFree(m_dGridParticleHash)); m_dGridParticleHash = NULL; }
	if (m_dGridParticleIndex){ checkCudaErrors(cudaFree(m_dGridParticleIndex)); m_dGridParticleIndex = NULL; }
	if (m_dCellStart){ checkCudaErrors(cudaFree(m_dCellStart)); m_dCellStart = NULL; }
	if (m_dCellEnd){ checkCudaErrors(cudaFree(m_dCellEnd)); m_dCellEnd = NULL; }
	if (m_dSortedPos){ checkCudaErrors(cudaFree(m_dSortedPos)); m_dSortedPos = NULL; }
	if (m_dSortedVel){ checkCudaErrors(cudaFree(m_dSortedVel)); m_dSortedVel = NULL; }
	if (m_dSortedGridParticleHash){ checkCudaErrors(cudaFree(m_dSortedGridParticleHash)); m_dSortedGridParticleHash = NULL; }
	if (m_dSortedGridParticleIndex){ checkCudaErrors(cudaFree(m_dSortedGridParticleIndex)); m_dSortedGridParticleIndex = NULL; }
	if (staticPos){ checkCudaErrors(cudaFree(staticPos)); staticPos = NULL; }
	if (staticNorm){ checkCudaErrors(cudaFree(staticNorm)); staticNorm = NULL; }
	if (staticVel){ checkCudaErrors(cudaFree(staticVel)); staticVel = NULL; }
	if (staticSortedPos){ checkCudaErrors(cudaFree(staticSortedPos)); staticSortedPos = NULL; }
	if (staticSortedVel){ checkCudaErrors(cudaFree(staticSortedVel)); staticSortedVel = NULL; }
	if (staticGridParticleHash){ checkCudaErrors(cudaFree(staticGridParticleHash)); staticGridParticleHash = NULL; }
	if (staticGridParticleIndex){ checkCudaErrors(cudaFree(staticGridParticleIndex)); staticGridParticleIndex = NULL; }
	if (staticCellStart){ checkCudaErrors(cudaFree(staticCellStart)); staticCellStart = NULL; }
	if (staticCellEnd){ checkCudaErrors(cudaFree(staticCellEnd)); staticCellEnd = NULL; }
	if (sortedStaticGridParticleHash){ checkCudaErrors(cudaFree(sortedStaticGridParticleHash)); sortedStaticGridParticleHash = NULL; }
	if (sortedStaticGridParticleIndex){ checkCudaErrors(cudaFree(sortedStaticGridParticleIndex)); sortedStaticGridParticleIndex = NULL; }
	if (rbPositions){ checkCudaErrors(cudaFree(rbPositions)); rbPositions = NULL; }
	if (rbVelocities){ checkCudaErrors(cudaFree(rbVelocities)); rbVelocities = NULL; }
	if (rbAngularVelocity){ checkCudaErrors(cudaFree(rbAngularVelocity)); rbAngularVelocity = NULL; }
	if (rbInertia){ checkCudaErrors(cudaFree(rbInertia)); rbInertia = NULL; }
	if (rbCurrentInertia){ checkCudaErrors(cudaFree(rbCurrentInertia)); rbCurrentInertia = NULL; }
	if (rbQuaternion){ checkCudaErrors(cudaFree(rbQuaternion)); rbQuaternion = NULL; }
	if (relativePos){ checkCudaErrors(cudaFree(relativePos)); relativePos = NULL; }
	if (rbIndices){ checkCudaErrors(cudaFree(rbIndices)); rbIndices = NULL; }
	if (rbMass){ checkCudaErrors(cudaFree(rbMass)); rbMass = NULL; }
	if (pForce){ checkCudaErrors(cudaFree(pForce)); pForce = NULL; }
	if (pTorque){ checkCudaErrors(cudaFree(pTorque)); pTorque = NULL; }
	if (pCountARCollions){ checkCudaErrors(cudaFree(pCountARCollions)); pCountARCollions = NULL; }
	if (objectParticlePositions){ checkCudaErrors(cudaFree(objectParticlePositions)); objectParticlePositions = NULL; }

	if (isRigidBody){ delete isRigidBody; isRigidBody = NULL; }
	if (particlesPerObjectThrown){ delete particlesPerObjectThrown; particlesPerObjectThrown = NULL; }
	if (firstObjectIndex){ delete firstObjectIndex; firstObjectIndex = NULL; }
	if (objectParticleStart){ delete objectParticleStart; objectParticleStart = NULL; }
	if (modelMatrix){ delete modelMatrix; modelMatrix = NULL; }
	if (cumulativeQuaternion){ delete cumulativeQuaternion; cumulativeQuaternion = NULL; }

	m_numParticles = 0;
	numRigidBodies = 0;
	objectsUsed = 0;
	objectsThrown = 0;
	modelNameVector.resize(0);

}

void
ParticleSystem::dumpGrid()
{
	// dump grid information
	copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
	copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
	uint maxCellSize = 0;

	for (uint i = 0; i<m_numGridCells; i++)
	{
		if (m_hCellStart[i] != 0xffffffff)
		{
			uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

			//            printf("cell: %d, %d particles\n", i, cellSize);
			if (cellSize > maxCellSize)
			{
				maxCellSize = cellSize;
			}
		}
	}

	printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
	// debug
	copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float) * 4 * count);
	copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float) * 4 * count);

	for (uint i = start; i<start + count; i++)
	{
		//        printf("%d: ", i);
		printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i * 4 + 0], m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
	}
}

float *
ParticleSystem::getArray(ParticleArray array)
{
	assert(m_bInitialized);

	float *hdata = 0;
	float *ddata = 0;
	struct cudaGraphicsResource *cuda_vbo_resource = 0;

	switch (array)
	{
	default:
	case POSITION:
		hdata = m_hPos;
		ddata = m_dPos;
		cuda_vbo_resource = m_cuda_posvbo_resource;
		break;

	case VELOCITY:
		hdata = m_hVel;
		ddata = m_dVel;
		break;
	}

	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
	return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
	assert(m_bInitialized);

	switch (array)
	{
	default:
	case POSITION:
	{
		if (m_bUseOpenGL)
		{
			unregisterGLBufferObject(m_cuda_posvbo_resource);
			glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
			glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		}
	}
		break;

	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	}
}

inline float frand()
{
	return rand() / (float)RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
	srand(1973);

	for (uint z = 0; z<size[2]; z++)
	{
		for (uint y = 0; y<size[1]; y++)
		{
			for (uint x = 0; x<size[0]; x++)
			{
				uint i = (z*size[1] * size[0]) + (y*size[0]) + x;

				if (i < numParticles)
				{
					m_hPos[i * 4] = (spacing * x) + m_params.particleRadius + (frand())*jitter;
					m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius + (frand())*jitter;
					m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius + (frand())*jitter;
					m_hPos[i * 4 + 3] = 1.0f;

					m_hVel[i * 4] = 0.0f;
					m_hVel[i * 4 + 1] = 0.0f;
					m_hVel[i * 4 + 2] = 0.0f;
					m_hVel[i * 4 + 3] = 0.0f;
				}
			}
		}
	}
}

void
ParticleSystem::reset(ParticleConfig config)
{
	switch (config)
	{
	default:
	case CONFIG_RANDOM:
	{
		int p = 0, v = 0;

		for (uint i = 0; i < m_numParticles; i++)
		{
			float point[3];
			point[0] = frand();
			point[1] = frand();
			point[2] = frand();
			/*m_hPos[p++] = 2 * (point[0] - 0.5f);
			m_hPos[p++] = 2 * (point[1] - 0.5f);
			m_hPos[p++] = 2 * (point[2] - 0.5f);*/
			m_hPos[p++] = 7 * (point[0]);
			m_hPos[p++] = 7 * (point[1]);
			m_hPos[p++] = 7 * (point[2]);
			m_hPos[p++] = 1.0f; // radius
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
		}
	}
		break;

	case CONFIG_GRID:
	{
		float jitter = m_params.particleRadius*0.01f;
		uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
		uint gridSize[3];
		gridSize[0] = gridSize[1] = gridSize[2] = s;
		initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
	}
		break;
	}

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void ParticleSystem::initCPU()
{
	//malloc rigid body variables
	CM_CPU = new float4[numRigidBodies];
	V_CPU = new float4[numRigidBodies];
	W_CPU = new float4[numRigidBodies];
	P_CPU = new float4[numRigidBodies];
	L_CPU = new float4[numRigidBodies];
	F_CPU = new float4[numRigidBodies];
	T_CPU = new float4[numRigidBodies];
	Iinv_CPU = new glm::mat3[numRigidBodies];
	Q_CPU = new glm::quat[numRigidBodies];

	//malloc particle variables
	displacement = new float4[m_numParticles];
	POS_CPU = new float4[m_numParticles];
	COL_CPU = new float4[m_numParticles];
	indexRB = new int[m_numParticles];

	//copy rigid body variables
	checkCudaErrors(cudaMemcpy(CM_CPU, rbPositions, 4 * sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(V_CPU, rbVelocities, 4 * sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(W_CPU, rbAngularVelocity, 4 * sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(P_CPU, rbLinearMomentum, 4 * sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(L_CPU, rbAngularMomentum, 4 * sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(F_CPU, rbForces, 4 * sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(T_CPU, rbTorque, 4 * sizeof(float) * numRigidBodies, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Iinv_CPU, rbInertia, sizeof(glm::mat3) * numRigidBodies, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Q_CPU, rbQuaternion, sizeof(glm::quat) * numRigidBodies, cudaMemcpyDeviceToHost));

	//copy particle variables
	checkCudaErrors(cudaMemcpy(POS_CPU, relativePos, 4 * sizeof(float) * m_numParticles, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(displacement, relativePos, 4 * sizeof(float) * m_numParticles, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(indexRB, rbIndices, sizeof(int) * m_numParticles, cudaMemcpyDeviceToHost));

	//checkCudaErrors(cudaMemcpy(&bunnyRadius, rbRadii, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&bunnyMass, rbMass, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&bunnyInertia, rbInertia, sizeof(glm::mat3), cudaMemcpyDeviceToHost));
}


void ParticleSystem::toggleGravity()
{
	if (m_params.gravity.y < 0)
	{
		m_params.gravity.y = 0;
		std::cout << "Disabling gravity" << std::endl;
	}
	else
	{
		m_params.gravity.y = -0.0003f;
		std::cout << "Enabling gravity" << std::endl;
	}
}

void ParticleSystem::changeSpring(const float &x)
{
	m_params.spring += x;
	if (m_params.spring < 0.f)
		m_params.spring = 0.f;
	else if (m_params.spring > 1.f)
		m_params.spring = 1.f;
	std::cout << "Spring set to: " << m_params.spring << std::endl;
}

void ParticleSystem::changeDamping(const float &x)
{
	m_params.damping += x;
	if (m_params.damping < 0.f)
		m_params.damping = 0.f;
	else if (m_params.damping > 0.2f)
		m_params.damping = 0.2f;
	std::cout << "Damping set to: " << m_params.damping << std::endl;
}

void ParticleSystem::changeGlobalDamping(const float &x)
{
	m_params.globalDamping += x;
	if (m_params.globalDamping < 0.f)
		m_params.globalDamping = 0.f;
	else if (m_params.globalDamping > 1.f)
		m_params.globalDamping = 1.f;
	std::cout << "Global damping set to: " << m_params.globalDamping << std::endl;
}

void ParticleSystem::changeShear(const float &x)
{
	m_params.shear += x;
	if (m_params.shear < 0.f)
		m_params.shear = 0.f;
	else if (m_params.shear > 0.5f)
		m_params.shear = 0.5f;
	std::cout << "Shear set to: " << m_params.shear << std::endl;
}