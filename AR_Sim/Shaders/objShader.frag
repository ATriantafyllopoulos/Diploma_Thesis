#version 330

smooth in vec2 vTexCoord;
smooth in vec3 vNormal;
smooth in vec4 vEyeSpacePos;
smooth in vec3 vWorldPos;
out vec4 outputColor;

uniform sampler2D gSampler;
uniform vec4 vColor;

#include "dirLight.frag"
uniform DirectionalLight sunLight;

void main()
{
	vec3 vNormalized = normalize(vNormal);
	
	vec4 vTexColor = texture2D(gSampler, vTexCoord);

	vec4 vMixedColor = vTexColor*vColor;

	//outputColor = vMixedColor;

	float fDiffuseIntensity = max(0.0, dot(vNormalized, -sunLight.vDirection));
	float fMult = clamp(sunLight.fAmbient+fDiffuseIntensity, 0.0, 1.0);
	vec4 vDirLightColor = vec4(sunLight.vColor*fMult, 1.0);	
	
	outputColor = vMixedColor * vDirLightColor;
}