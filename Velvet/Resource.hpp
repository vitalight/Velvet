#pragma once

#include <iostream>
#include <unordered_map>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "External/stb_image.h"
#include "Mesh.hpp"

namespace Velvet
{
	inline unordered_map<string, unsigned int> textureCache;
	inline unordered_map<string, shared_ptr<Mesh>> meshCache;
	inline unordered_map<string, shared_ptr<Material>> matCache;

	inline string defaultTexturePath = "Assets/Texture/";
	inline string defaultMeshPath = "Assets/Model/";
	inline string defaultMaterialPath = "Assets/Shader/";

	class Resource
	{
	public:

		static unsigned int LoadTexture(const string& path)
		{
			if (textureCache.count(path) > 0)
			{
				return textureCache[path];
			}

            unsigned int textureID;
            glGenTextures(1, &textureID);
			textureCache[path] = textureID;

            int width, height, nrComponents;
            unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
			if (data == nullptr)
			{
				data = stbi_load((defaultTexturePath+path).c_str(), &width, &height, &nrComponents, 0);
			}
            if (data)
            {
				GLenum internalFormat = GL_RED;
				GLenum dataFormat = GL_RED;

				if (nrComponents == 3)
				{
					internalFormat = GL_SRGB;
					dataFormat = GL_RGB;
				}
				else if (nrComponents == 4)
				{
					internalFormat = GL_SRGB_ALPHA;
					dataFormat = GL_RGBA;
				}

				glBindTexture(GL_TEXTURE_2D, textureID);
				glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, dataFormat, GL_UNSIGNED_BYTE, data);
                glGenerateMipmap(GL_TEXTURE_2D);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                stbi_image_free(data);
            }
            else
            {
				fmt::print("Error(Resource): Texture failed to load at path({})\n", path);
                stbi_image_free(data);
            }

            return textureID;
		}

		static shared_ptr<Mesh> LoadMesh(const string& path)
		{
			if (meshCache.count(path) > 0)
			{
				return meshCache[path];
			}

			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> texCoords;
			vector<unsigned int> indices;

			Assimp::Importer importer;
			const aiScene* scene = importer.ReadFile(defaultMeshPath + path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
			// check for errors
			if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
			{
				scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

				if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
				{	
					fmt::print("Error(Resource) Fail to load mesh ({})\n", path);
					return shared_ptr<Mesh>();
				}
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
			auto result = shared_ptr<Mesh>(new Mesh(vertices, normals, texCoords, indices));
			meshCache[path] = result;
			return result;
		}
	
		static shared_ptr<Material> LoadMaterial(const string& path)
		{
			if (matCache.count(path))
			{
				return matCache[path];
			}
			string vertexCode = LoadText(defaultMaterialPath + path + ".vert");
			if (vertexCode.length() == 0) vertexCode = LoadText(path + ".vert");
			if (vertexCode.length() == 0) fmt::print("Error(Resource): material.vertex not found ({})\n", path);

			string fragmentCode = LoadText(defaultMaterialPath + path + ".frag");
			if (fragmentCode.length() == 0) fragmentCode = LoadText(path + ".frag");
			if (fragmentCode.length() == 0) fmt::print("Error(Resource): material.fragment not found ({})\n", path);

			auto result = make_shared<Material>(vertexCode, fragmentCode);
			matCache[path] = result;
			return result;
		}

		static string LoadText(const string& path)
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
				e;
				//std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
			}
			return code;
		}

	private:
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