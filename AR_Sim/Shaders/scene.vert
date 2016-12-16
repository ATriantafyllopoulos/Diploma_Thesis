#version 330
layout(location = 0) in vec3 inPosition;
layout(location = 0) in vec2 inTex;

out vec2 UV;

void main()
{
	gl_Position = vec4(inPosition, 1);
	UV.x = (1 + inTex.x) / 2;
	UV.y = (1 - inTex.y) / 2;
	//UV = (vec2(1,1)-inTex)/2;
}
