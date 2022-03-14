#pragma once

#include <vector>

using namespace std;

#define SHADER(A) "#version 330\n" #A

namespace Velvet
{
	namespace DefaultAssets
	{

		const vector<float> quad_vertices = {
			// positions          // texture coords
			 0.5f,  0.5f, 0.0f,   1.0f, 1.0f, // top right
			 0.5f, -0.5f, 0.0f,   1.0f, 0.0f, // bottom right
			-0.5f, -0.5f, 0.0f,   0.0f, 0.0f, // bottom left
			-0.5f,  0.5f, 0.0f,   0.0f, 1.0f  // top left 
		};

		const vector<unsigned int> quad_indices = { // note that we start from 0!
			0, 1, 3, // first triangle
			1, 2, 3 // second triangle
		};

		const char* quad_shader_vertex = SHADER(
			layout(location = 0) in vec3 aPos;
		layout(location = 1) in vec2 aTexCoord;
		out vec2 TexCoord;

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 projection;

		void main()
		{
			gl_Position = projection * view * model * vec4(aPos, 1.0f);
			TexCoord = aTexCoord;
		}
		);

		const char* quad_shader_fragment = SHADER(
			out vec4 FragColor;
		in vec2 TexCoord;
		uniform sampler2D texture1;
		uniform sampler2D texture2;
		void main()
		{
			FragColor = mix(texture(texture1, TexCoord),
				texture(texture2, TexCoord), 0.2);
		}
		);
	}
}