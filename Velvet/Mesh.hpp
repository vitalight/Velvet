#pragma once

#include <vector>
#include <algorithm> 

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

using namespace std;

namespace Velvet
{
	/// <summary>
	/// A class that allows you to create or modify meshes.
	/// </summary>
	class Mesh
	{
	public:

		Mesh(vector<unsigned int> attributeSizes, vector<float> packedVertices, vector<unsigned int> indices = vector<unsigned int>())
		{
			unsigned int stride = 0;
			for (int i = 0; i < attributeSizes.size(); i++)
			{
				stride += attributeSizes[i];
			}
			unsigned int numVertices = (unsigned int)packedVertices.size() / stride;

			for (unsigned int i = 0; i < numVertices; i++)
			{
				unsigned int baseV = stride * i;
				unsigned int baseN = (stride >= 6) ? baseV + 3 : baseV;
				unsigned int baseT = baseN + 3;

				m_vertices.push_back(glm::vec3(packedVertices[baseV + 0], packedVertices[baseV + 1], packedVertices[baseV + 2]));
				if (stride >= 6)
				{
					m_normals.push_back(glm::vec3(packedVertices[baseN + 0], packedVertices[baseN + 1], packedVertices[baseN + 2]));
				}
				m_texCoords.push_back(glm::vec2(packedVertices[baseT + 0], packedVertices[baseT + 1]));
			}
			Initialize(m_vertices, m_normals, m_texCoords, indices, attributeSizes);
		}

		Mesh(const vector<glm::vec3>& vertices, const vector<glm::vec3>& normals = vector<glm::vec3>(),
			const vector<glm::vec2>& texCoords = vector<glm::vec2>(), const vector<unsigned int>& indices = vector<unsigned int>())
		{
			Initialize(vertices, normals, texCoords, indices);
		}

		Mesh(const Mesh&) = delete;

		~Mesh()
		{
			if (m_EBO > 0)
			{
				glDeleteBuffers(1, &m_EBO);
			}
			if (m_VBO > 0)
			{
				glDeleteBuffers(1, &m_VBO);
			}
			if (m_VAO > 0)
			{
				glDeleteVertexArrays(1, &m_VAO);
			}
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
				return (unsigned int)m_indices.size();
			}
			else
			{
				return (unsigned int)m_vertices.size();
			}
		}

	private:
		vector<glm::vec3> m_vertices;
		vector<glm::vec3> m_normals;
		vector<glm::vec2> m_texCoords;
		vector<unsigned int> m_indices;

		unsigned int m_VAO = 0;
		unsigned int m_VBO = 0;
		unsigned int m_EBO = 0;

		void Initialize(const vector<glm::vec3>& vertices, const vector<glm::vec3>& normals, const vector<glm::vec2>& texCoords,
			const vector<unsigned int>& indices, vector<unsigned int> attributeSizes = {})
		{
			m_vertices = vertices;
			m_normals = normals;
			m_texCoords = texCoords;
			m_indices = indices;

			if (attributeSizes.size() == 0)
			{
				if (vertices.size() > 0) attributeSizes.push_back(3u);
				if (normals.size() > 0) attributeSizes.push_back(3u);
				if (texCoords.size() > 0) attributeSizes.push_back(2u);
			}

			// 1. bind Vertex Array Object
			glGenVertexArrays(1, &m_VAO);
			glBindVertexArray(m_VAO);

			glGenBuffers(1, &m_VBO);

			// 2. copy our vertices array in a buffer for OpenGL to use
			size_t size[] = { vertices.size() * sizeof(glm::vec3),
				normals.size() * sizeof(glm::vec3),
				texCoords.size() * sizeof(glm::vec2) };

			glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
			glBufferData(GL_ARRAY_BUFFER, size[0] + size[1] + size[2], NULL, GL_STATIC_DRAW);

			glBufferSubData(GL_ARRAY_BUFFER, 0, size[0], vertices.data());
			glBufferSubData(GL_ARRAY_BUFFER, size[0], size[1], normals.data());
			glBufferSubData(GL_ARRAY_BUFFER, size[0] + size[1], size[2], texCoords.data());

			// 3. copy our index array in a element buffer for OpenGL to use
			if (useIndices())
			{
				glGenBuffers(1, &m_EBO);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
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