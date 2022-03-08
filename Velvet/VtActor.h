#pragma once

#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#define SHADER(A) "#version 330\n" #A

namespace Velvet
{
	using namespace std;

	class Mesh
	{
	public:
		Mesh() {}

		Mesh(vector<float> vertices, vector<unsigned int> indices)
			: m_vertices(vertices), m_indices(indices)
		{
			// 1. bind Vertex Array Object
			glGenVertexArrays(1, &m_VAO);
			glBindVertexArray(m_VAO);

			unsigned int VBO;
			glGenBuffers(1, &VBO);

			// 2. copy our vertices array in a buffer for OpenGL to use
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

			// 3. copy our index array in a element buffer for OpenGL to use
			unsigned int EBO;
			glGenBuffers(1, &EBO);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

			// 4. then set the vertex attributes pointers
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
		}

		unsigned int GetVAO()
		{
			return m_VAO;
		}

	private:
		vector<float> m_vertices;
		vector<unsigned int> m_indices;
		unsigned int m_VAO = 0;
	};

	class Material
	{
	public:
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
					fmt::print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{}\n", infoLog);
				}
			}

			m_shaderProgram = glCreateProgram();
			{
				glAttachShader(m_shaderProgram, vertexShader);
				glAttachShader(m_shaderProgram, fragmentShader);
				glLinkProgram(m_shaderProgram);
				int success;
				char infoLog[512];
				glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
				if (!success)
				{
					glGetProgramInfoLog(m_shaderProgram, 512, NULL, infoLog);
				}
			}
			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);
		}

		unsigned int GetShaderProgram()
		{
			return m_shaderProgram;
		}
	private:
		unsigned int m_shaderProgram = 0;
	};

	class Component
	{
	public:
		virtual void Start() {}

		virtual void Update() { }

		virtual void OnDestroy() {}

		string name = "BaseComponent";
	};

	class MeshRenderer : public Component
	{
	public:
		MeshRenderer() {}

		MeshRenderer(Mesh mesh, Material material)
			: m_mesh(mesh), m_material(material) {}

		void Update() override
		{
			// draw triangles
			glUseProgram(m_material.GetShaderProgram());
			
			// update the uniform color
			float timeValue = glfwGetTime();
			float greenValue = sin(timeValue) / 2.0f + 0.5f;
			int vertexColorLocation = glGetUniformLocation(m_material.GetShaderProgram(), "ourColor");
			glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
			
			glBindVertexArray(m_mesh.GetVAO());
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			//glBindVertexArray(0);
		}

	private:
		Mesh m_mesh;
		Material m_material;
	};

	class VtActor
	{
	public:
		static shared_ptr<VtActor> FixedTriangle();

		void Start();

		void Update();

		void OnDestroy();

		void AddComponent(shared_ptr<Component> component);

	private:
		vector<shared_ptr<Component>> m_components;
	};
}