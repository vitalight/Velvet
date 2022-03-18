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

		const vector<float> cube_vertices = {
			// positions          // normals           // texture coords
			-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
			 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
			 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
			 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
			-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
			-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,

			-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
			 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
			-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
			-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

			-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
			-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
			-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
			-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
			-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
			-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

			 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
			 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
			 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
			 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
			 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
			 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

			-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
			 0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
			 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
			 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
			-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
			-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,

			-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
			 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
			-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
			-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f
		};

		const vector<int> cube_attributes = { 3, 3, 2 };

		// TODO: shader from file
		const char* cube_shader_vertex = SHADER(
			layout(location = 0) in vec3 aPos;
			layout(location = 1) in vec3 aNormal;
			layout(location = 2) in vec2 aTexCoords;

			out vec3 FragPos;
			out vec3 Normal;
			out vec2 TexCoords;

			uniform mat4 model;
			uniform mat4 view;
			uniform mat4 projection;

			void main()
			{
				gl_Position = projection * view * model * vec4(aPos, 1.0);
				FragPos = vec3(model * vec4(aPos, 1.0));
				//Normal = aNormal;
				Normal = mat3(transpose(inverse(model))) * aNormal;
				TexCoords = aTexCoords;
			}
		);

		const char* cube_shader_fragment = SHADER(
			out vec4 FragColor;

			struct Material {
				sampler2D diffuse;
				sampler2D specular;
				float shininess;
			};

			struct DirLight {
				vec3 direction;

				vec3 ambient;
				vec3 diffuse;
				vec3 specular;
			};

			struct PointLight {
				vec3 position;

				float constant;
				float linear;
				float quadratic;

				vec3 ambient;
				vec3 diffuse;
				vec3 specular;
			};

			struct SpotLight {
				vec3 position;
				vec3 direction;
				float cutOff;
				float outerCutOff;

				float constant;
				float linear;
				float quadratic;

				vec3 ambient;
				vec3 diffuse;
				vec3 specular;
			};

			//#define NR_POINT_LIGHTS 4

			in vec3 FragPos;
			in vec3 Normal;
			in vec2 TexCoords;

			uniform vec3 viewPos;
			uniform DirLight dirLight;
			uniform PointLight pointLights[4];
			uniform SpotLight spotLight;
			uniform Material material;

			// function prototypes
			vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir);
			vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
			vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

			void main()
			{
				// properties
				vec3 norm = normalize(Normal);
				vec3 viewDir = normalize(viewPos - FragPos);

				// == =====================================================
				// Our lighting is set up in 3 phases: directional, point lights and an optional flashlight
				// For each phase, a calculate function is defined that calculates the corresponding color
				// per lamp. In the main() function we take all the calculated colors and sum them up for
				// this fragment's final color.
				// == =====================================================
				// phase 1: directional lighting
				vec3 result = CalcDirLight(dirLight, norm, viewDir);
				// phase 2: point lights
				for (int i = 0; i < 4; i++)
					result += CalcPointLight(pointLights[i], norm, FragPos, viewDir);
				// phase 3: spot light
				result += CalcSpotLight(spotLight, norm, FragPos, viewDir);

				FragColor = vec4(result, 1.0);
			}

			// calculates the color when using a directional light.
			vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir)
			{
				vec3 lightDir = normalize(-light.direction);
				// diffuse shading
				float diff = max(dot(normal, lightDir), 0.0);
				// specular shading
				vec3 reflectDir = reflect(-lightDir, normal);
				float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
				// combine results
				vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));
				vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
				vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));
				return (ambient + diffuse + specular);
			}

			// calculates the color when using a point light.
			vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
			{
				vec3 lightDir = normalize(light.position - fragPos);
				// diffuse shading
				float diff = max(dot(normal, lightDir), 0.0);
				// specular shading
				vec3 reflectDir = reflect(-lightDir, normal);
				float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
				// attenuation
				float distance = length(light.position - fragPos);
				float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
				// combine results
				vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));
				vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
				vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));
				ambient *= attenuation;
				diffuse *= attenuation;
				specular *= attenuation;
				return (ambient + diffuse + specular);
			}

			// calculates the color when using a spot light.
			vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
			{
				vec3 lightDir = normalize(light.position - fragPos);
				// diffuse shading
				float diff = max(dot(normal, lightDir), 0.0);
				// specular shading
				vec3 reflectDir = reflect(-lightDir, normal);
				float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
				// attenuation
				float distance = length(light.position - fragPos);
				float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
				// spotlight intensity
				float theta = dot(lightDir, normalize(-light.direction));
				float epsilon = light.cutOff - light.outerCutOff;
				float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
				// combine results
				vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));
				vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
				vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));
				ambient *= attenuation * intensity;
				diffuse *= attenuation * intensity;
				specular *= attenuation * intensity;
				return (ambient + diffuse + specular);
			}
		);

		const char* light_shader_vertex = SHADER(
			layout(location = 0) in vec3 aPos;
			uniform mat4 model;
			uniform mat4 view;
			uniform mat4 projection;
			void main()
			{
				gl_Position = projection * view * model * vec4(aPos, 1.0);
			}
		);

		const char* light_shader_fragment = SHADER(
			out vec4 FragColor;
			void main()
			{
				FragColor = vec4(1.0); // set all 4 vector values to 1.0
			}
		);

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