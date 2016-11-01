#version 330
uniform vec4 vColor;
in vec4 fragmentColor;
out vec4 outputColor;

void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);
    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_PointCoord.st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	//N.x = (gl_PointCoord.s - 0.5) * gl_PointSize + gl_Position.x - 0.5;
	//N.y = -(gl_PointCoord.t - 0.5) * gl_PointSize + gl_Position.y - 0.5;
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);
    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

	outputColor = vColor * diffuse * fragmentColor;
}