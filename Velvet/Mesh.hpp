#pragma once

#include <vector>
#include <algorithm> 

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include "Material.hpp"
#include "Resource.hpp"

using namespace std;

namespace Velvet
{
	/// <summary>
	/// A class that allows you to create or modify meshes.
	/// </summary>
	class Mesh
	{
	public:
		// for infinite plane
		Mesh()
		{
			float w = 1.0;
			vector<glm::vec4> vertices = { 
				glm::vec4(1,1,0, w), glm::vec4(-1,-1,0, w), glm::vec4(-1,1,0, w), glm::vec4(1,-1,0, w) };
			vector<glm::vec3> normals = {
				glm::vec3(0, 1, 0), glm::vec3(0, 1, 0) , glm::vec3(0, 1, 0) , glm::vec3(0, 1, 0) };
			vector<unsigned int> indices = { 0, 1, 2, 1, 0, 3 };
			vector<unsigned int> attributeSizes = { 4,3 };
			m_indices = indices;

			// 1. bind Vertex Array Object
			glGenVertexArrays(1, &m_VAO);
			glBindVertexArray(m_VAO);

			unsigned int VBO;
			glGenBuffers(1, &VBO);

			// 2. copy our vertices array in a buffer for OpenGL to use
			size_t size[] = { vertices.size() * sizeof(glm::vec4),
				normals.size() * sizeof(glm::vec3)};

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, size[0] + size[1], NULL, GL_STATIC_DRAW);

			glBufferSubData(GL_ARRAY_BUFFER, 0, size[0], vertices.data());
			glBufferSubData(GL_ARRAY_BUFFER, size[0], size[1], normals.data());

			// 3. copy our index array in a element buffer for OpenGL to use
			if (useIndices())
			{
				unsigned int EBO;
				glGenBuffers(1, &EBO);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
			}

			// 4. then set the vertex attributes pointers
			SetupAttributes(attributeSizes);
		}

		Mesh(vector<unsigned int> attributeSizes, vector<float> vertices, vector<unsigned int> indices = vector<unsigned int>())
		{
			unsigned int stride = 0;
			for (int i = 0; i < attributeSizes.size(); i++)
			{
				stride += attributeSizes[i];
			}
			unsigned int numVertices = vertices.size() / stride;

			for (unsigned int i = 0; i < numVertices; i++)
			{
				unsigned int baseV = stride * i;
				unsigned int baseN = (stride >= 6) ? baseV + 3 : baseV;
				unsigned int baseT = baseN + 3;

				m_vertices.push_back(glm::vec3(vertices[baseV + 0], vertices[baseV + 1], vertices[baseV + 2]));
				if (stride >= 6)
				{
					m_normals.push_back(glm::vec3(vertices[baseN + 0], vertices[baseN + 1], vertices[baseN + 2]));
				}
				m_texCoords.push_back(glm::vec2(vertices[baseT + 0], vertices[baseT + 1]));
			}
			Initialize(m_vertices, m_normals, m_texCoords, indices, attributeSizes);
		}

		Mesh(vector<glm::vec3>& vertices, vector<glm::vec3>& normals, vector<glm::vec2>& texCoords, vector<unsigned int>& indices = vector<unsigned int>())
		{
			Initialize(vertices, normals, texCoords, indices);
		}

		Mesh(const string& filePath)
		{
			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> texCoords;
			vector<unsigned int> indices;
			Resource::LoadMeshFromFile(filePath, vertices, normals, texCoords, indices);

			Initialize(vertices, normals, texCoords, indices);
		}

		unsigned int VAO() const
		{
			if (m_VAO == 0)
			{
				fmt::print("Error(Mesh): Access VAO of 0. Possiblely uninitialized.");
			}
			return m_VAO;
		}

		bool useIndices() const
		{
			return m_indices.size() > 0;
		}

		unsigned int drawCount() const
		{
			if (useIndices())
			{
				return m_indices.size();
			}
			else
			{
				return m_vertices.size();
			}
		}

	private:
		vector<glm::vec3> m_vertices;
		vector<glm::vec3> m_normals;
		vector<glm::vec2> m_texCoords;
		vector<unsigned int> m_indices;

		unsigned int m_VAO = 0;

		void Initialize(const vector<glm::vec3>& vertices, const vector<glm::vec3>& normals, const vector<glm::vec2>& texCoords,
			const vector<unsigned int>& indices, const vector<unsigned int> attributeSizes = { 3,3,2 })
		{
			m_vertices = vertices;
			m_normals = normals;
			m_texCoords = texCoords;
			m_indices = indices;

			// 1. bind Vertex Array Object
			glGenVertexArrays(1, &m_VAO);
			glBindVertexArray(m_VAO);

			unsigned int VBO;
			glGenBuffers(1, &VBO);

			// 2. copy our vertices array in a buffer for OpenGL to use
			size_t size[] = { vertices.size() * sizeof(glm::vec3),
				normals.size() * sizeof(glm::vec3),
				texCoords.size() * sizeof(glm::vec2) };

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, size[0] + size[1] + size[2], NULL, GL_STATIC_DRAW);

			glBufferSubData(GL_ARRAY_BUFFER, 0, size[0], vertices.data());
			glBufferSubData(GL_ARRAY_BUFFER, size[0], size[1], normals.data());
			glBufferSubData(GL_ARRAY_BUFFER, size[0] + size[1], size[2], texCoords.data());

			// 3. copy our index array in a element buffer for OpenGL to use
			if (useIndices())
			{
				unsigned int EBO;
				glGenBuffers(1, &EBO);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
			}

			// 4. then set the vertex attributes pointers
			SetupAttributes(attributeSizes);
		}

		void SetupAttributes(const vector<unsigned int>& attributeSizes) const
		{
			size_t current = 0;
			for (int i = 0; i < attributeSizes.size(); i++)
			{
				int size = attributeSizes[i];
				glVertexAttribPointer(i, size, GL_FLOAT, GL_FALSE, size * sizeof(float),
					(void*)(current));
				//fmt::print("glVertexAttribPointer({}, {}, GL_FLOAT, GL_FALSE, size * sizeof(float), {}\n",
				//	i, size, current);
				glEnableVertexAttribArray(i);
				current += size * sizeof(float) * m_vertices.size();
			}
		}
	};
}