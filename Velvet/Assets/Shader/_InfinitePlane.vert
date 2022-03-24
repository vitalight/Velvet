#version 330

layout(location = 0) in vec4 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 nearPoint;
out vec3 farPoint;

vec3 UnprojectPoint(float x, float y, float z, mat4 view, mat4 projection) {
    mat4 viewInv = inverse(view);
    mat4 projInv = inverse(projection);
    vec4 unprojectedPoint =  viewInv * projInv * vec4(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main()
{
    nearPoint = UnprojectPoint(aPos.x, aPos.y, 0.0, view, projection).xyz; // unprojecting on the near plane
    farPoint = UnprojectPoint(aPos.x, aPos.y, 1.0, view, projection).xyz; // unprojecting on the far plane
	gl_Position = aPos;
}