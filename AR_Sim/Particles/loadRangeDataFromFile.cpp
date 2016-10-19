#include "particleSystem.h"
#include "ParticleAuxiliaryFunctions.h"
#include "BVHcreation.h"

cudaError_t findAABBCub(
		float4 *d_in,
		float4 &cpuMin,
		float4 &cpuMax,
		float4 *gpuMin,
		float4 *gpuMax,
		int numberOfPrimitives);

cudaError_t loadRangeImage(
	unsigned short *image,
	float **VAOdata,
	float **staticPos,
	float **staticNorm,
	glm::mat4 &cameraTransformation,
	float *staticRadii,
	float particleRadius,
	int imageWidth,
	int imageHeight);
uint nextpow2(uint v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

bool ParticleSystem::readRangeData()
{

	//if (!openDepthImage(std::string("Data/RangeImages/desk_1_1_depth.png"))) return false;
	//return openImage(std::string("Data/RangeImages/desk_1_1.png"));
	/*if (!openDepthImage(std::string("Data/RangeImages/desk/desk_1/desk_1_1_depth.png"))) return false;
	if (!openImage(std::string("Data/RangeImages/desk/desk_1/desk_1_1_1.png"))) return false;*/
    //std::cout << "Attempting to RGB-D data." << std::endl;

    //std::cout << "Attempting to load depth image." << std::endl;
	//if (!openDepthImage(std::string("Data/RangeImages/banana/banana_1/banana_1_1_1_depth.png"))) return false;
	if (!openDepthImage(std::string("Data/RangeImages/rgbd_dataset_freiburg1_xyz/depth/Depth (1).png"))) return false;
	if (!openImage(std::string("Data/RangeImages/rgbd_dataset_freiburg1_xyz/rgb/Texture (1).png"))) return false;
    //std::cout << "Depth image loaded successfully." << std::endl;

    //std::cout << "Attempting to load RGB image." << std::endl;
	//if (!openImage(std::string("Data/RangeImages/banana/banana_1/banana_1_1_1.png"))) return false;
    //std::cout << "RGB image loaded successfully." << std::endl;
	float4 cpuMin, cpuMax;
	checkCudaErrors(findAABBCub((float4 *)staticPos, cpuMin, cpuMax, NULL, NULL, numberOfRangeData));
	minPos = make_float3(cpuMin);
	maxPos = make_float3(cpuMax);
	//m_params.worldOrigin = minPos;
    //*m_params.gridSize.x = nextpow2(uint((maxPos.x - minPos.x) / m_params.particleRadius + 0.5));
	//m_params.gridSize.y = nextpow2(uint((maxPos.y - minPos.y) / m_params.particleRadius + 0.5));
	//m_params.gridSize.z = nextpow2(uint((maxPos.z - minPos.z) / m_params.particleRadius + 0.5));*/
	//
    //std::cout << "RGB-D data loaded successfully." << std::endl;
	return true;
}

bool ParticleSystem::openImage(std::string a_sPath)
{
    cv::Mat textureImage = cv::imread(a_sPath.c_str());

    unsigned short *data = (unsigned short *)textureImage.data;
	loadImageToTexture(data, textureImage.cols, textureImage.rows, GL_BGR, true);
	return true; // Success
}

bool ParticleSystem::loadImageToTexture(unsigned short* bData, int a_iWidth, int a_iHeight, GLenum format, bool bGenerateMipMaps)
{
	// Generate an OpenGL texture ID for this texture
	if (rangeTexture)
		glDeleteTextures(1, &rangeTexture);
	glGenTextures(1, &rangeTexture);
	glBindTexture(GL_TEXTURE_2D, rangeTexture);
	if (format == GL_RGBA || format == GL_BGRA)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bData);
	// We must handle this because of internal format parameter
	else if (format == GL_RGB || format == GL_BGR)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bData);
	else
		glTexImage2D(GL_TEXTURE_2D, 0, format, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bData);
	if (bGenerateMipMaps)glGenerateMipmap(GL_TEXTURE_2D);
	
	if (rangeSampler)
		glDeleteSamplers(1, &rangeSampler);
	glGenSamplers(1, &rangeSampler);
	glBindSampler(0, rangeSampler);

	// Set magnification filter
	glSamplerParameteri(rangeSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Set minification filter
	glSamplerParameteri(rangeSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	return true;
}

bool ParticleSystem::openDepthImage(std::string a_sPath)
{
	cv::Mat imgDepth = cv::imread(a_sPath.c_str(), CV_16UC1);
	firstRaw = imgDepth.clone();
//	for(unsigned int i = 0; i < 480; i++)
//	{
//		for(unsigned int j = 0; j < 640; j++)
//		{
//			imgDepth.at<unsigned short>(i, j) /= 5000.f;
//		}
//	}
	for(unsigned int i = 0; i < 480; i++)
	{
		for(unsigned int j = 0; j < 640; j++)
		{
			//std::cout << firstRaw.at<unsigned short>(i, j) << " ";
			firstRaw.at<unsigned short>(i, j) /= 5.f;
		}
		//std::cout << std::endl;
	}
	//std::cout << firstRaw << std::endl;
	imageWidth = imgDepth.cols;
	imageHeight = imgDepth.rows;
	numberOfRangeData = imageWidth * imageHeight;
	unsigned short *data = (unsigned short  *)imgDepth.data;

	

    //std::cout << "Attempting to initialize SoA." << std::endl;
	initializeRealSoA(); //initiliaze SoA variables for real scene
    //std::cout << "SoA initialized successfully." << std::endl;
    //std::cout << "Attempting to initialize static particles." << std::endl;
    initializeStaticParticles();
    //std::cout << "Static particles initialized successfully." << std::endl;

    //std::cout << "Attempting to upload depth data to CUDA." << std::endl;
	loadDepthToVBO(data, imageWidth, imageHeight);
    //std::cout << "Depth data uploaded to CUDA successfully." << std::endl;
	
	return true; // Success
}

bool ParticleSystem::loadDepthToVBO(unsigned short* bData, int width, int height)
{
    //std::cout << "Attempting to initialize VAO." << std::endl;
	glGenVertexArrays(1, &rangeVAO);
    //std::cout << "VAO generation was successful." << std::endl;
	glBindVertexArray(rangeVAO);
    //std::cout << "VAO binding was successful." << std::endl;
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * width * height * 5, NULL, GL_STATIC_DRAW);

    //std::cout << "VBO generation was successful." << std::endl;

    GLint position_attribute = glGetAttribLocation(shader.getProgramID(), "inPosition");
	glEnableVertexAttribArray(position_attribute);
	GLint texture_attribute = glGetAttribLocation(shader.getProgramID(), "inCoord");
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3) + sizeof(float2), 0);
	glEnableVertexAttribArray(texture_attribute);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float3) + sizeof(float2), (void*)sizeof(float3));
	glBindVertexArray(0);
    //std::cout << "VAO initialized successfully." << std::endl;
	cudaError_t cudaStatus;
	cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaRangeVAO, rangeVAO, cudaGraphicsMapFlagsWriteDiscard);
	if (cudaStatus != cudaSuccess)
    {
		std::cout << "cudaGraphicsGLRegisterBuffer for rangeVAO failed." << std::endl;
		exit(1);
    }
	unsigned short *d_data;
	cudaStatus = cudaMalloc((void**)&d_data, sizeof(unsigned short) *numberOfRangeData);
	if (cudaStatus != cudaSuccess)
		exit(1);

	cudaStatus = cudaMemcpy(d_data, bData, sizeof(unsigned short) * numberOfRangeData, cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess)
		exit(1);
	cudaGraphicsMapResources(1, &cudaRangeVAO, 0);
	size_t num_bytes;
	float *VAOdata;
	cudaGraphicsResourceGetMappedPointer((void **)&VAOdata, &num_bytes, cudaRangeVAO);
	cudaStatus = loadRangeImage(d_data,
		&VAOdata,
		&staticPos,
		&staticNorm,
		cameraTransformation,
		r_radii,
		m_params.particleRadius,
		width, height);
	if (cudaStatus != cudaSuccess)
    {
        std::cout << "CUDA load range image kernels failed." << std::endl;
        std::cout << "CUDA error code: " << cudaStatus << std::endl;
        std::cout << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
    }
	cudaGraphicsUnmapResources(1, &cudaRangeVAO, 0);

	cudaFree(d_data);

//	float *normals = new float[numberOfRangeData];
//
//	std::ofstream myfile ("normals.txt");
//	cudaMemcpy(normals, staticNorm, sizeof(float) * 4 * numberOfRangeData, cudaMemcpyDeviceToHost);
//	if (myfile.is_open())
//	{
//		for(int i = 0; i < numberOfRangeData; i++)
//			myfile << normals[4 * i] << " " << normals[4 * i + 1] << " " << normals[4 * i + 2] << " " << normals[4 * i + 3] << '\n';
//	}
//
//	delete normals;
//	float *pos  = new float[numberOfRangeData];
//	float *normals = new float[numberOfRangeData];
//	cudaMemcpy(pos, staticPos, sizeof(float) * 4 * numberOfRangeData, cudaMemcpyDeviceToHost);
//	for(int i = 0; i < numberOfRangeData; i++)
//	{
//
//	}
//	delete normals;
//	delete pos;
	//delete dummy;
	return true;
}

bool myfunction(cv::DMatch i, cv::DMatch j) { return (i.distance < j.distance); }

void ParticleSystem::updateFrame()
{
	imageIndex += imageOffset;
	if (imageIndex == 798)
		imageOffset = -1;
	else if (imageIndex == 1)
		imageOffset = 1;
	changeTexture();

	changeDepthMap();
}

void ParticleSystem::changeTexture()
{
	//std::string a_sPath = "Data/RangeImages/desk/desk_1/desk_1_";
	//std::string a_sPath = "Data/RangeImages/banana/banana_1/banana_1_1_";
	std::string a_sPath = "Data/RangeImages/rgbd_dataset_freiburg1_xyz/rgb/Texture (";
    a_sPath += patch::to_string(imageIndex);
    a_sPath += ").png";
	//a_sPath += ".png";

	cv::Mat textureImage = cv::imread(a_sPath.c_str());
	//currentFrame = textureImage.clone();
	unsigned short *bDataPointer = (unsigned short  *)textureImage.data;
	
	GLenum format = GL_BGR;
	int a_iWidth = textureImage.cols;
	int a_iHeight = textureImage.rows;
	bool bGenerateMipMaps = true;
	glBindTexture(GL_TEXTURE_2D, rangeTexture);
	if (format == GL_RGBA || format == GL_BGRA)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bDataPointer);
	// We must handle this because of internal format parameter
	else if (format == GL_RGB || format == GL_BGR)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bDataPointer);
	else
		glTexImage2D(GL_TEXTURE_2D, 0, format, a_iWidth, a_iHeight, 0, format, GL_UNSIGNED_BYTE, bDataPointer);
	if (bGenerateMipMaps)glGenerateMipmap(GL_TEXTURE_2D);

}

void ParticleSystem::changeDepthMap()
{
	//std::string a_sPath = "Data/RangeImages/desk/desk_1/desk_1_";
//	std::string a_sPath = "Data/RangeImages/banana/banana_1/banana_1_1_";
//    a_sPath += patch::to_string(imageIndex);
//	a_sPath += "_depth.png";
	std::string a_sPath = "Data/RangeImages/rgbd_dataset_freiburg1_xyz/depth/Depth (";
	a_sPath += patch::to_string(imageIndex);
	a_sPath += ").png";


	cv::Mat imgDepth = cv::imread(a_sPath.c_str(), CV_16UC1);
	secondRaw = imgDepth.clone();
//	for(unsigned int i = 0; i < 480; i++)
//	{
//		for(unsigned int j = 0; j < 640; j++)
//		{
//			imgDepth.at<unsigned short>(i, j) /= 5000.f;
//		}
//	}

	for(unsigned int i = 0; i < 480; i++)
	{
		for(unsigned int j = 0; j < 640; j++)
		{
			secondRaw.at<unsigned short>(i, j) /= 5.f;
		}
	}
	if (!pauseFrame)
	{
		CameraMotionEstimation();
	}

	imageWidth = imgDepth.cols;
	imageHeight = imgDepth.rows;
	numberOfRangeData = imageWidth * imageHeight;
	unsigned short *data = (unsigned short  *)imgDepth.data;
	cudaFree(0);

	//initializeRealSoA(); //initiliaze SoA variables for real scene
	//initializeStaticParticles();
	//std::cout << "Number of range data: " << numberOfRangeData << std::endl;
	unsigned short *d_data;
	checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(unsigned short) * numberOfRangeData));

	checkCudaErrors(cudaMemcpy(d_data, data, sizeof(unsigned short) * numberOfRangeData, cudaMemcpyHostToDevice));


    cudaGraphicsMapResources(1, &cudaRangeVAO, 0);
	size_t num_bytes;
	float *VAOdata;
	cudaGraphicsResourceGetMappedPointer((void **)&VAOdata, &num_bytes, cudaRangeVAO);
	checkCudaErrors(loadRangeImage(d_data,
		&VAOdata,
		&staticPos,
		&staticNorm,
		cameraTransformation,
		r_radii,
		m_params.particleRadius,
		imageWidth,
		imageHeight));

    cudaGraphicsUnmapResources(1, &cudaRangeVAO, 0);
	if (d_data)
		checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ParticleSystem::CameraMotionEstimation()
{
//	ICPOdometry icpOdom(640, 480, 319.5, 239.5, 528, 528);
//	icpOdom.initICPModel((unsigned short *)firstRaw.data);
//	icpOdom.initICP((unsigned short *)secondRaw.data);
//
//	T_wc_prev = T_wc_curr;
//
//	Sophus::SE3d T_prev_curr = T_wc_prev.inverse() * T_wc_curr;
//
//	int threads = 64;
//	int blocks = 80;
//	icpOdom.getIncrementalTransformation(T_prev_curr, threads, blocks);
//
//	T_wc_curr = T_wc_prev * T_prev_curr;
//
//	std::swap(firstRaw, secondRaw);
//
//	Eigen::Matrix4f currentPose = T_wc_curr.cast<float>().matrix();
//	//Eigen::Vector3f trans = currentPose.topRightCorner(3, 1);
//	//Eigen::Matrix3f rot = currentPose.topLeftCorner(3, 3);
//
//
//	cameraTranslation = currentPose.topRightCorner(3, 1);
//	cameraRotation = currentPose.topLeftCorner(3, 3);
//	cameraQuaternion = Eigen::Quaternionf(cameraRotation);
////	Eigen::Vector3f ea = cameraRotation.eulerAngles(0, 1, 2);
////	ea[1] = -ea[1];
////	cameraRotation = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitX())
////	*Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY())
////	*Eigen::AngleAxisf(ea[2], Eigen::Vector3f::UnitZ());
//	for (int row = 0; row < 4; row++)
//		for (int col = 0; col < 4; col++)
//				cameraTransformation[row][col] = currentPose(row, col);
////	for (int row = 0; row < 3; row++)
////			for (int col = 0; col < 3; col++)
////				cameraTransformation[row][col] = cameraRotation(row, col);
//
////	std::cout << "Camera transformation matrix: " << std::endl;
////	for (int row = 0; row < 4; row++)
////	{
////		for (int col = 0; col < 4; col++)
////			std::cout << cameraTransformation[row][col] << " ";
////		std::cout << std::endl;
////	}
////	std::cout << std::endl;
////	std::cout << std::endl;
////
////	glm::vec4 testPoint(1.f, 2.f, 3.f, 1.f);
////	testPoint = transpose(inverse(cameraTransformation)) * testPoint;
////	std::cout << "Test point is: (" << testPoint.x << " "
////			<< testPoint.y << " "
////			<< testPoint.z << " "
////			<< testPoint.w << ")"
////			<< std::endl;
////	std::cout << std::endl;
////	std::cout << std::endl;

}
