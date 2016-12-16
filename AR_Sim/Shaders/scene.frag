#version 330

in vec2 UV;

uniform sampler2D gSampler;

out vec4 outputColor;
void main() 
{
   outputColor = texture2D(gSampler, UV);
}