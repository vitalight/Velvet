#pragma once

#include <vector>
#include <algorithm> 

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Material.hpp"

using namespace std;

namespace Velvet
{
	/// <summary>
	/// A class that allows you to create or modify meshes.
	/// </summary>
	class Mesh
	{
	public:
		//Mesh() {}

		Mesh(vector<unsigned int> attributeSizes, vector<float> vertices, vector<unsigned int> indices = vector<unsigned int>())
		{
			unsigned int stride = 0;
			for (int i = 0; i < attributeSizes.size(); i++)
			{
				stride += attributeSizes[i];
			}
			unsigned int numVertices = vertices.size() / stride;

			for (int i = 0; i < numVertices; i++)
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
			LoadMeshFromFile(filePath, vertices, normals, texCoords, indices);

			Initialize(vertices, normals, texCoords, indices);
		}

		unsigned int VAO() const
		{
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
			int current = 0;
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

		static void LoadMeshFromFile(const string& filePath, vector<glm::vec3>& vertices, vector<glm::vec3>& normals, vector<glm::vec2>& texCoords, vector<unsigned int>& indices)
		{
			Assimp::Importer importer;
			const aiScene* scene = importer.ReadFile(filePath, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
			// check for errors
			if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
			{
				cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
				return;
			}
			aiMesh* mesh = scene->mMeshes[0];

			int drawCount = mesh->mNumVertices;

			// walk through each of the mesh's vertices
			for (unsigned int i = 0; i < mesh->mNumVertices; i++)
			{
				// positions
				vertices.push_back(AdaptVector(mesh->mVertices[i]));
				// normals
				if (mesh->HasNormals())
				{
					normals.push_back(AdaptVector(mesh->mNormals[i]));
				}
				else
				{
					fmt::print("Normals not found\n");
					exit(-1);
				}
				// texture coordinates
				if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
				{
					texCoords.push_back(AdaptVector(mesh->mTextureCoords[0][i]));
				}
				else
				{
					texCoords.push_back(glm::vec2(0.0f, 0.0f));
				}
			}
			// now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
			for (unsigned int i = 0; i < mesh->mNumFaces; i++)
			{
				aiFace face = mesh->mFaces[i];
				// retrieve all indices of the face and store them in the indices vector
				for (unsigned int j = 0; j < face.mNumIndices; j++)
					indices.push_back(face.mIndices[j]);
			}
		}
		
		static inline glm::vec3 AdaptVector(const aiVector3D& input)
		{
			return glm::vec3(input.x, input.y, input.z);
		}

		static inline glm::vec2 AdaptVector(const aiVector2D& input)
		{
			return glm::vec2(input.x, input.y);
		}
	};
}