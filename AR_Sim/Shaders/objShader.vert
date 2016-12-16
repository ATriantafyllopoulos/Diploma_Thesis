#version 330

uniform struct Matrices
{
	mat4 projMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;                                                                           
	mat4 normalMatrix;
} matrices;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inCoord;
layout (location = 2) in vec3 inNormal;

smooth out vec3 vNormal;
smooth out vec2 vTexCoord;
smooth out vec3 vWorldPos;

smooth out vec4 vEyeSpacePos;

void main()
{
	mat4 mMV = matrices.viewMatrix*matrices.modelMatrix;  
	//mat4 mMVP = matrices.projMatrix*matrices.viewMatrix*matrices.modelMatrix;
  
	vTexCoord = inCoord;
	
	//vec3 g = vec3(0.478957, -8.010560, 4.166928);
	//g = normalize(g);
	//mat4 gravityR = mat4(1.f, 0.f, 0.f, 0.f,
	//	-g.x, -g.y, -g.z, 0.f,
	//	0.f, g.z, -g.y, 0.f,
	//	0.f, 0.f, 0.f, 1.f);
	//gravityR = inverse(gravityR);
	//vec4 temp4 = gravityR * vec4(inPosition, 1.0);
	//vEyeSpacePos = mMV*temp4;
	
	vec3 g = vec3(0.478957, -8.010560, 4.166928);
	g = normalize(g);
	mat4 gravityR = mat4(1.f, 0.f, 0.f, 0.f,
		-g.x, -g.y, -g.z, 0.f,
		0.f, g.z, -g.y, 0.f,
		0.f, 0.f, 0.f, 1.f);
	gravityR = inverse(gravityR);
	vec4 temp4 = gravityR * vec4(inPosition, 1.0);
	vEyeSpacePos = mMV*temp4;
	
	//vEyeSpacePos = mMV*vec4(inPosition, 1.0);
	vNormal = (matrices.normalMatrix*vec4(inNormal, 1.0)).xyz;
	vWorldPos = (matrices.modelMatrix*vec4(inPosition, 1.0)).xyz;
	vec3 temp;
	//vEyeSpacePos.x = 579.83 * (vEyeSpacePos.x) / vEyeSpacePos.z + 321.55;
	//vEyeSpacePos.y = 586.73 * (vEyeSpacePos.y) / vEyeSpacePos.z + 235.01;
	vEyeSpacePos.x = 528 * (vEyeSpacePos.x) / vEyeSpacePos.z + 319.5;
	vEyeSpacePos.y = 528 * (vEyeSpacePos.y) / vEyeSpacePos.z + 239.5;
	temp.x = -(vEyeSpacePos.x - 320) / 320;
	temp.y = -(vEyeSpacePos.y - 240) / 240;
	float far = 100.0;
	float near = 0.1;
	temp.z = (far + near) / (far - near) * vEyeSpacePos.z + 2.0f * far * near / (far - near) ;
	temp.z = temp.z / vEyeSpacePos.z;
	gl_Position = vec4(temp, 1);
}
