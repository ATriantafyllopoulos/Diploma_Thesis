#version 330


uniform mat4 viewMatrix;                                                                           
uniform float pointRadius;
uniform float pointScale;
layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec4 inColor;

out vec4 fragmentColor;

void main()
{

	vec3 g = vec3(0.478957, -8.010560, 4.166928);
	g = normalize(g);
	mat4 gravityR = mat4(1.f, 0.f, 0.f, 0.f,
		-g.x, -g.y, -g.z, 0.f,
		0.f, g.z, -g.y, 0.f,
		0.f, 0.f, 0.f, 1.f);
	gravityR = inverse(gravityR);
	vec4 temp4 = gravityR * vec4(inPosition, 1.0);
	vec4 vEyeSpacePos = viewMatrix*temp4;
	//vec4 vEyeSpacePos = viewMatrix*vec4(inPosition, 1.0);
	float dist = length(vEyeSpacePos.xyz);
	vec3 temp;
	//vEyeSpacePos.x = 579.83 * (vEyeSpacePos.x) / vEyeSpacePos.z + 321.55;
	//vEyeSpacePos.y = 586.73 * (vEyeSpacePos.y) / vEyeSpacePos.z + 235.01;
	//vEyeSpacePos.x = 528 * (vEyeSpacePos.x) / vEyeSpacePos.z + 319.5;
	//vEyeSpacePos.y = 528 * (vEyeSpacePos.y) / vEyeSpacePos.z + 239.5;
	vEyeSpacePos.x = 517.3 * (vEyeSpacePos.x) / vEyeSpacePos.z + 318.6;
	vEyeSpacePos.y = 516.5 * (vEyeSpacePos.y) / vEyeSpacePos.z + 255.3;
	temp.x = -(vEyeSpacePos.x - 320) / 320;
	temp.y = -(vEyeSpacePos.y - 240) / 240;
	float far = 100.0;
	float near = 0.1;
	temp.z = (far + near) / (far - near) * vEyeSpacePos.z + 2.0f * far * near / (far - near);
	temp.z = temp.z / vEyeSpacePos.z;
	gl_Position = vec4(temp, 1);
	
	
    gl_PointSize = pointRadius * (pointScale / dist);
	fragmentColor = inColor;
}
