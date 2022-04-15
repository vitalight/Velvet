#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 _Projection;
uniform mat4 _MVP;
uniform float _ParticleRadius;

out GS {
    vec2 uv;
    vec3 centerEyePos;
} gs;


void build_quad(vec4 position)
{    
    gs.centerEyePos = position.xyz / position.w;

    gs.uv = vec2(-1.0, -1.0);
    gl_Position = _Projection * (position + _ParticleRadius * vec4(-1.0, -1.0, 0.0, 0.0));    // 1:bottom-left
    EmitVertex();   

    gs.uv = vec2(1.0, -1.0);
    gl_Position = _Projection * (position + _ParticleRadius * vec4(1.0, -1.0, 0.0, 0.0));    // 2:bottom-right
    EmitVertex();

    gs.uv = vec2(-1.0, 1.0);
    gl_Position = _Projection * (position + _ParticleRadius * vec4(-1.0, 1.0, 0.0, 0.0));    // 3:top-left
    EmitVertex();

    gs.uv = vec2(1.0, 1.0);
    gl_Position = _Projection * (position + _ParticleRadius * vec4(1.0, 1.0, 0.0, 0.0));    // 4:top-right
    EmitVertex();

    EndPrimitive();
}

void main() {    
    build_quad(gl_in[0].gl_Position);
}  