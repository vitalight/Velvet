#version 330 core

layout(location = 0) in vec3 Pos;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 UV;
layout(location = 3) in vec3 Translation;

uniform mat4 _WorldToLight;
uniform mat4 _Model;
uniform float _ParticleRadius;

void main()
{
	vec3 instancePos = _ParticleRadius * Pos + Translation;
    gl_Position = _WorldToLight * _Model * vec4(instancePos, 1.0);
}