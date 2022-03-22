#pragma once

#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include "Material.hpp"

using namespace std;

namespace Velvet
{
	
	class Mesh
	{
	public:
		Mesh() {}

		Mesh(int numVertices, vector<float> vertices)
			: m_numVertices(numVertices), m_vertices(vertices), m_useIndices(false)
		{
			// 1. bind Vertex Array Object
			glGenVertexArrays(1, &m_VAO);
			glBindVertexArray(m_VAO);

			unsigned int VBO;
			glGenBuffers(1, &VBO);

			// 2. copy our vertices array in a buffer for OpenGL to use
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

			// 3. then set the vertex attributes pointers
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
		}

		Mesh(int numVertices, vector<float> vertices, vector<unsigned int> indices)
			: m_numVertices(numVertices), m_vertices(vertices), m_indices(indices), m_useIndices(true)
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

		void SetupAttributes(vector<int> attributeSizes)
		{
			int total = 0;
			for (int i = 0; i < attributeSizes.size(); i++)
			{
				total += attributeSizes[i];
			}
			int current = 0;
			for (int i = 0; i < attributeSizes.size(); i++)
			{
				int size = attributeSizes[i];
				glVertexAttribPointer(i, size, GL_FLOAT, GL_FALSE, total * sizeof(float),
					(void*)(current * sizeof(float)));
				glEnableVertexAttribArray(i);
				current += size;
			}
		}

		unsigned int VAO() const
		{
			return m_VAO;
		}

		bool useIndices() const
		{
			return m_useIndices;
		}

		unsigned int numVertices() const
		{
			return m_numVertices;
		}

	private:
		vector<float> m_vertices;
		vector<unsigned int> m_indices;
		unsigned int m_VAO = 0;
		bool m_useIndices = true;
		unsigned int m_numVertices = 0;
	};
	
}