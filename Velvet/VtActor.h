#pragma once

#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VtComponent.h"

#define SHADER(A) "#version 330\n" #A

namespace Velvet
{
	using namespace std;

	class VtActor
	{
	public:
		VtActor();

		VtActor(string name) : m_name(name) {};

		static shared_ptr<VtActor> FixedTriangle();

		static shared_ptr<VtActor> FixedQuad();

		void Start();

		void Update();

		void OnDestroy();

		void AddComponent(shared_ptr<VtComponent> component);

		template <typename T>
		T* GetComponent()
		{
			T* result = nullptr;
			for (auto c : m_components)
			{
				result = dynamic_cast<T*>(c.get());
				if (result)
					return result;
			}
			return result;
		}

	private:
		string m_name;

		vector<shared_ptr<VtComponent>> m_components;
	};

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

		unsigned int VAO() const
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

	private:
		unsigned int m_shaderID = 0;
	};

	class MeshRenderer : public VtComponent
	{
	public:
		MeshRenderer() {}

		MeshRenderer(Mesh mesh, Material material)
			: m_mesh(mesh), m_material(material) {}

		void Update() override
		{
			// draw triangles
			m_material.Use();
			m_material.SetInt("texture1", 0);
			m_material.SetInt("texture2", 1);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m_material.texture1);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, m_material.texture2);

			glm::mat4 trans = glm::mat4(1.0f);
			trans = glm::translate(trans, glm::vec3(0.5f, -0.5f, 0.0f));
			trans = glm::rotate(trans, (float)glfwGetTime(),
				glm::vec3(0.0f, 0.0f, 1.0f));
			unsigned int transformLoc = glGetUniformLocation(m_material.shaderID(), "transform");
			glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(trans));

			glBindVertexArray(m_mesh.VAO());
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			//glBindVertexArray(0);
		}

		Material material() const
		{
			return m_material;
		}

	private:
		Mesh m_mesh;
		Material m_material;
	};

	// Animate material color with time
	class MaterialAnimator : public VtComponent
	{
	public:
		MaterialAnimator(float speed = 1.0f) : m_speed(speed) {};

		void Update() override
		{
			auto renderer = actor->GetComponent<MeshRenderer>();
			if (!renderer)
			{
				fmt::print("MaterialAnimator: Renderer not found\n");
				return;
			}
			auto material = renderer->material();
			material.Use();
			float timeValue = glfwGetTime() * m_speed;
			float greenValue = sin(timeValue) / 2.0f + 0.5f;
			int vertexColorLocation = glGetUniformLocation(material.shaderID(), "ourColor");
			glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
		}
	private:
		float m_speed = 1.0f;
	};
}