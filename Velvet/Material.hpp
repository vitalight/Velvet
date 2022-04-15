#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

using namespace std;

namespace Velvet
{
	class Material
	{
	public:
		unordered_map<string, unsigned int> textures;

		Material() {}

		Material(string& vertexCode, string& fragmentCode, string& geometryCode)
		{
			const char* vShaderCode = vertexCode.c_str();
			const char* fShaderCode = fragmentCode.c_str();
			const char* gShaderCode = geometryCode.c_str();
			m_shaderID = CompileShader(vShaderCode, fShaderCode, gShaderCode);
		}

		Material(const Material&) = delete;

		~Material()
		{
			glDeleteProgram(m_shaderID);
		}

		unsigned int CompileShader(const char* vShaderCode, const char* fShaderCode, const char* gShaderCode) const
		{
			unsigned int vertex, fragment, geometry;
			unsigned int shader = glCreateProgram();
			// vertex shader
			vertex = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertex, 1, &vShaderCode, NULL);
			glCompileShader(vertex);
			CheckCompileErrors(vertex, "VERTEX");
			glAttachShader(shader, vertex);
			// fragment Shader
			fragment = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragment, 1, &fShaderCode, NULL);
			glCompileShader(fragment);
			CheckCompileErrors(fragment, "FRAGMENT");
			glAttachShader(shader, fragment);
			// geometry Shader
			if (strlen(gShaderCode) > 0)
			{
				geometry = glCreateShader(GL_GEOMETRY_SHADER);
				glShaderSource(geometry, 1, &gShaderCode, NULL);
				glCompileShader(geometry);
				CheckCompileErrors(fragment, "GEOMETRY");
				glAttachShader(shader, geometry);
			}
			// shader Program
			glLinkProgram(shader);
			CheckCompileErrors(shader, "PROGRAM");
			// delete the shaders as they're linked into our program now and no longer necessary
			glDeleteShader(vertex);
			glDeleteShader(fragment);
			if (strlen(gShaderCode) > 0) glDeleteShader(geometry);
			return shader;
		}

		unsigned int shaderID() const
		{
			return m_shaderID;
		}

		void Use() const
		{
			glUseProgram(m_shaderID);
		}

		GLint GetLocation(const string& name) const
		{
			return glGetUniformLocation(m_shaderID, name.c_str());
		}

		// utility uniform functions
		// ------------------------------------------------------------------------
		void SetTexture(const std::string&name, unsigned int texture)
		{
			textures[name] = texture;
		}
		// ------------------------------------------------------------------------
		void SetBool(const std::string& name, bool value) const
		{
			Use();
			glUniform1i(GetLocation(name), (int)value);
		}
		// ------------------------------------------------------------------------
		void SetInt(const std::string& name, int value) const
		{
			Use();
			glUniform1i(GetLocation(name), value);
		}
		// ------------------------------------------------------------------------
		void SetUInt(const std::string& name, unsigned int value) const
		{
			Use();
			glUniform1ui(GetLocation(name), value);
		}
		// ------------------------------------------------------------------------
		void SetFloat(const std::string& name, float value) const
		{
			Use();
			auto err = glGetError();
			if (err != GL_NO_ERROR)
			{
				// possible reason: opengl buffer overflow (e.g. DrawArrays with too much item count)
				fmt::print("Error(Material::SetFloat): Code #{} in material({})\n", err, this->name);
			}
			glUniform1f(GetLocation(name), value);
		}
		// ------------------------------------------------------------------------
		void SetVec2(const std::string& name, const glm::vec2& value) const
		{
			Use();
			glUniform2fv(GetLocation(name), 1, &value[0]);
		}
		void SetVec2(const std::string& name, float x, float y) const
		{
			Use();
			glUniform2f(GetLocation(name), x, y);
		}
		// ------------------------------------------------------------------------
		void SetVec3(const std::string& name, const glm::vec3& value) const
		{
			Use();
			glUniform3fv(GetLocation(name), 1, &value[0]);
		}
		void SetVec3(const std::string& name, float x, float y, float z) const
		{
			Use();
			glUniform3f(GetLocation(name), x, y, z);
		}
		// ------------------------------------------------------------------------
		void SetVec4(const std::string& name, const glm::vec4& value) const
		{
			Use();
			glUniform4fv(GetLocation(name), 1, &value[0]);
		}
		void SetVec4(const std::string& name, float x, float y, float z, float w)
		{
			Use();
			glUniform4f(GetLocation(name), x, y, z, w);
		}
		// ------------------------------------------------------------------------
		void SetMat2(const std::string& name, const glm::mat2& mat) const
		{
			Use();
			glUniformMatrix2fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
		}
		// ------------------------------------------------------------------------
		void SetMat3(const std::string& name, const glm::mat3& mat) const
		{
			Use();
			glUniformMatrix3fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
		}
		// ------------------------------------------------------------------------
		void SetMat4(const std::string& name, const glm::mat4& mat) const
		{
			Use();
			glUniformMatrix4fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
		}

		string name = "";
		float specular = 0.2f;
		float smoothness = 100.0f;
		bool doubleSided = false;
		bool noWireframe = false;
	private:
		unsigned int m_shaderID = -1;

		void CheckCompileErrors(unsigned int shader, std::string type) const
		{
			int success;
			char infoLog[1024];
			if (type != "PROGRAM")
			{
				glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
				if (!success)
				{
					glGetShaderInfoLog(shader, 1024, NULL, infoLog);
					std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
					exit(-1);
				}
			}
			else
			{
				glGetProgramiv(shader, GL_LINK_STATUS, &success);
				if (!success)
				{
					glGetProgramInfoLog(shader, 1024, NULL, infoLog);
					std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
					exit(-1);
				}
			}
		}
	};
}