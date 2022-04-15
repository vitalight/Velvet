#version 330

layout(location = 0) in vec3 Pos;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 UV;

out VS {
	vec3 worldPos;
	vec3 normal;
	vec2 uv;
	vec4 lightSpaceFragPos;
} vs;

uniform mat4 _Model;
uniform mat4 _View;
uniform mat4 _Projection;
uniform mat4 _WorldToLight;
uniform mat3 _Normalmatrix;

void main()
{
	gl_Position = _Projection * _View * _Model * vec4(Pos, 1.0);

	vs.worldPos = vec3(_Model * vec4(Pos, 1.0));
	vs.normal = _Normalmatrix * Normal;
	vs.uv = UV;
    vs.lightSpaceFragPos = _WorldToLight * vec4(vs.worldPos, 1.0);
}