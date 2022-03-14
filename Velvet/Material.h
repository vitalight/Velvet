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

		void SetBool(const string& name, bool value) const
		{
			glUniform1i(glGetUniformLocation(m_shaderID, name.c_str()), (int)value);
		}

		void SetInt(const string& name, int value) const
		{
			glUniform1i(glGetUniformLocation(m_shaderID, name.c_str()), value);
		}

		void SetFloat(const string& name, float value) const
		{
			glUniform1f(glGetUniformLocation(m_shaderID, name.c_str()), value);
		}

		void SetMat4(const string& name, glm::mat4 value) const
		{
			glUniformMatrix4fv(glGetUniformLocation(m_shaderID, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
		}

	private:
		unsigned int m_shaderID = 0;
	};

}