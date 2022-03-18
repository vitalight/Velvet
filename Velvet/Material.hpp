#pragma once

#include <vector>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

using namespace std;

namespace Velvet
{
	class Material
	{
	public:
		unsigned int texture1 = 0;
		unsigned int texture2 = 0;

		Material() {}

		Material(const char* vertexShaderSource, const char* fragmentShaderSource)
		{
			unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
			{
				glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
				glCompileShader(vertexShader);
				int success;
				char infoLog[512];
				glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
				if (!success)
				{
					glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
					fmt::print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{}\n", infoLog);
					exit(-1);
				}
			}

			unsigned int fragmentShader;
			{
				fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
				glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
				glCompileShader(fragmentShader);
				int success;
				char infoLog[512];
				glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
				if (!success)
				{
					glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
					fmt::print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n{}\n", infoLog);
					exit(-1);
				}
			}

			m_shaderID = glCreateProgram();
			{
				glAttachShader(m_shaderID, vertexShader);
				glAttachShader(m_shaderID, fragmentShader);
				glLinkProgram(m_shaderID);
				int success;
				char infoLog[512];
				glGetProgramiv(m_shaderID, GL_LINK_STATUS, &success);
				if (!success)
				{
					glGetProgramInfoLog(m_shaderID, 512, NULL, infoLog);
					fmt::print("ERROR:SHADER::PROGRAM::LINKING_FAILED\n{}\n", infoLog);
					exit(-1);
				}
			}
			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);
		}

		unsigned int shaderID() const
		{
			return m_shaderID;
		}

		void Use()
		{
			glUseProgram(m_shaderID);
		}

		GLint GetLocation(const string& name) const
		{
			return glGetUniformLocation(m_shaderID, name.c_str());
		}

		// utility uniform functions
		// ------------------------------------------------------------------------
		void SetBool(const std::string& name, bool value) const
		{
			glUniform1i(GetLocation(name), (int)value);
		}
		// ------------------------------------------------------------------------
		void SetInt(const std::string& name, int value) const
		{
			glUniform1i(GetLocation(name), value);
		}
		// ------------------------------------------------------------------------
		void SetFloat(const std::string& name, float value) const
		{
			glUniform1f(GetLocation(name), value);
		}
		// ------------------------------------------------------------------------
		void SetVec2(const std::string& name, const glm::vec2& value) const
		{
			glUniform2fv(GetLocation(name), 1, &value[0]);
		}
		void SetVec2(const std::string& name, float x, float y) const
		{
			glUniform2f(GetLocation(name), x, y);
		}
		// ------------------------------------------------------------------------
		void SetVec3(const std::string& name, const glm::vec3& value) const
		{
			glUniform3fv(GetLocation(name), 1, &value[0]);
		}
		void SetVec3(const std::string& name, float x, float y, float z) const
		{
			glUniform3f(GetLocation(name), x, y, z);
		}
		// ------------------------------------------------------------------------
		void SetVec4(const std::string& name, const glm::vec4& value) const
		{
			glUniform4fv(GetLocation(name), 1, &value[0]);
		}
		void SetVec4(const std::string& name, float x, float y, float z, float w)
		{
			glUniform4f(GetLocation(name), x, y, z, w);
		}
		// ------------------------------------------------------------------------
		void SetMat2(const std::string& name, const glm::mat2& mat) const
		{
			glUniformMatrix2fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
		}
		// ------------------------------------------------------------------------
		void SetMat3(const std::string& name, const glm::mat3& mat) const
		{
			glUniformMatrix3fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
		}
		// ------------------------------------------------------------------------
		void SetMat4(const std::string& name, const glm::mat4& mat) const
		{
			glUniformMatrix4fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
		}

	private:
		unsigned int m_shaderID = 0;
	};

}