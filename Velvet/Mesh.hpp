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
	
	class Mesh
	{
	public:
		Mesh() {}

		Mesh(int numVertices, vector<float> vertices, vector<unsigned int> indices = vector<unsigned int>())
		{
			// HACK
			unsigned int actualNumVertices = 1;
			if (indices.size() > 0)
			{
				for (int i = 0; i < indices.size(); i++)
				{
					actualNumVertices = max(actualNumVertices, indices[i] + 1);
				}
			}
			else
			{
				actualNumVertices = numVertices;
			}

			unsigned int stride = vertices.size() / actualNumVertices;
			for (int i = 0; i < actualNumVertices; i++)
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
			SetupVAO(m_vertices, m_normals, m_texCoords, indices);
		}

		Mesh(vector<glm::vec3> &vertices, vector<glm::vec3> &normals, vector<glm::vec2> &texCoords, vector<unsigned int> &indices = vector<unsigned int>())
		{
			SetupVAO(vertices, normals, texCoords, indices);
		}

		Mesh(const string& modelPath)
		{
			Assimp::Importer importer;
			const aiScene* scene = importer.ReadFile(modelPath, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
			// check for errors
			if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
			{
				cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
				return;
			}
			aiMesh* mesh = scene->mMeshes[0];

			int numVertices = mesh->mNumVertices;
			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> texCoords;
			vector<unsigned int> indices;

			// walk through each of the mesh's vertices
			for (unsigned int i = 0; i < mesh->mNumVertices; i++)
			{
				// positions
				vertices.push_back(AdaptVec(mesh->mVertices[i]));
				// normals
				if (mesh->HasNormals())
				{
					normals.push_back(AdaptVec(mesh->mNormals[i]));
				}
				else
				{
					fmt::print("Normals not found\n");
					exit(-1);
				}
				// texture coordinates
				if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
				{
					texCoords.push_back(AdaptVec(mesh->mTextureCoords[0][i]));
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

			SetupVAO(vertices, normals, texCoords, indices);
		}

		inline glm::vec3 AdaptVec(aiVector3D input)
		{
			return glm::vec3(input.x, input.y, input.z);
		}

		inline glm::vec2 AdaptVec(aiVector2D input)
		{
			return glm::vec2(input.x, input.y);
		}

		void SetupVAO(vector<glm::vec3>& vertices, vector<glm::vec3>& normals, vector<glm::vec2>& texCoords, vector<unsigned int>& indices = vector<unsigned int>())
		{
			m_vertices = vertices;
			m_normals = normals;
			m_texCoords = texCoords;
			m_indices = indices;
			m_numVertices = indices.size() > 0 ? indices.size() : vertices.size();

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
			m_useIndices = (indices.size() > 0);
			if (m_useIndices)
			{
				unsigned int EBO;
				glGenBuffers(1, &EBO);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
			}

			// 4. then set the vertex attributes pointers
			SetupAttributes({ 3,3,2 });
		}

		void SetupAttributes(vector<int> attributeSizes)
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
				current += size * sizeof(float) * m_numVertices;
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
		vector<glm::vec3> m_vertices;
		vector<glm::vec3> m_normals;
		vector<glm::vec2> m_texCoords;
		vector<unsigned int> m_indices;

		unsigned int m_VAO = 0;
		bool m_useIndices = true;
		unsigned int m_numVertices = 0;
	};
	
}