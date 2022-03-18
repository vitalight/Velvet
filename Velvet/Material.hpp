#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

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

		// constructor generates the shader on the fly
		// ------------------------------------------------------------------------
		Material(const string vertexPath, const string fragmentPath)
		{
			string vertexCode = ReadFromFile(vertexPath);
			string fragmentCode = ReadFromFile(fragmentPath);
			const char* vShaderCode = vertexCode.c_str();
			const char* fShaderCode = fragmentCode.c_str();

			// 2. compile shaders
			CompileShader(vShaderCode, fShaderCode);

			return;
		}

		Material(const string path)
			: Material(path + ".vert", path + ".frag")
		{
		}

		static string ReadFromFile(const string path)
		{
			// 1. retrieve the vertex/fragment source code from filePath
			std::string code;
			std::ifstream file;
			// ensure ifstream objects can throw exceptions:
			file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			try
			{
				// open files
				file.open(path);
				std::stringstream vShaderStream;
				// read file's buffer contents into streams
				vShaderStream << file.rdbuf();
				// close file handlers
				file.close();
				// convert stream into string
				code = vShaderStream.str();
			}
			catch (std::ifstream::failure& e)
			{
				std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
			}
			return code;
		}

		void CompileShader(const char* vShaderCode, const char* fShaderCode)
		{
			unsigned int vertex, fragment;
			// vertex shader
			vertex = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertex, 1, &vShaderCode, NULL);
			glCompileShader(vertex);
			CheckCompileErrors(vertex, "VERTEX");
			// fragment Shader
			fragment = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragment, 1, &fShaderCode, NULL);
			glCompileShader(fragment);
			CheckCompileErrors(fragment, "FRAGMENT");
			// shader Program
			m_shaderID = glCreateProgram();
			glAttachShader(m_shaderID, vertex);
			glAttachShader(m_shaderID, fragment);
			glLinkProgram(m_shaderID);
			CheckCompileErrors(m_shaderID, "PROGRAM");
			// delete the shaders as they're linked into our program now and no longer necessary
			glDeleteShader(vertex);
			glDeleteShader(fragment);
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

		void CheckCompileErrors(unsigned int shader, std::string type)
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