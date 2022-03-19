#include "Actor.hpp"

#include "External/stb_image.h"

#include "DefaultAssets.hpp"
#include "Camera.hpp"
#include "PlayerController.hpp"
#include "Light.hpp"

namespace Velvet
{
	unsigned int LoadTexture(const char* path, bool isPNG)
	{
		unsigned int texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		// set the texture wrapping/filtering options (on currently bound texture)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// load and generate the texture
		int width, height, nrChannels;
		unsigned char* data = stbi_load(path, &width, &height,
			&nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, isPNG? GL_RGBA : GL_RGB,
				GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
		}
		else
		{
			std::cout << "Failed to load texture" << std::endl;
		}
		stbi_image_free(data);
		return texture;
	}

	Velvet::Actor::Actor() {}

	Velvet::Actor::Actor(string name) : name(name) {}
	
	/*
	shared_ptr<Actor> Actor::PrefabTriangle()
	{
		vector<float> vertices = {
			// positions // colors // texture coords
			0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top right
			0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom right
			-0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
			-0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // top left
		};
		vector<unsigned int> indices = { // note that we start from 0!
			0, 1, 2,
		};

		const char* vertexShaderSource = SHADER(
			layout(location = 0) in vec3 aPos;
		layout(location = 1) in vec3 aColor;
		layout(location = 2) in vec2 aTexCoord;
		out vec3 ourColor;
		out vec2 TexCoord;
		void main()
		{
			gl_Position = vec4(aPos, 1.0);
			ourColor = aColor;
			TexCoord = aTexCoord;
		}
		);

		const char* fragmentShaderSource = SHADER(
			out vec4 FragColor;
		in vec3 ourColor;
		in vec2 TexCoord;
		uniform sampler2D ourTexture;
		void main()
		{
			FragColor = texture(ourTexture, TexCoord);
		}
		);

		Mesh mesh(3, vertices, indices);
		{
			// position attribute
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
				(void*)0);
			glEnableVertexAttribArray(0);
			// color attribute
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
				(void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
		}
		Material material(vertexShaderSource, fragmentShaderSource);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));

		shared_ptr<Actor> actor(new Actor("Fixed Triangle"));
		actor->AddComponent(renderer);

		return actor;
	}

	void Compare(string s1, string s2)
	{
		for (int i = 0; i < s1.length() && i < s2.length(); i++)
		{
			if (s1[i] != s2[i])
			{
				fmt::print("s1[{}]({}) != s2[{}]({})\n", i, s1[i], i, s2[i]);
				return;
			}
		}
		fmt::print("s1 == s2\n");
	}

	shared_ptr<Actor> Actor::PrefabQuad()
	{
		Mesh mesh(6, DefaultAssets::quad_vertices, DefaultAssets::quad_indices);
		mesh.SetupAttributes({3, 2});

		Material material("Assets/Shader/Quad");
		{
			material.texture1 = LoadTexture("Assets/container.jpg", false);
			material.texture2 = LoadTexture("Assets/awesomeface.png", true);

			material.Use();
			material.SetInt("texture1", 0);
			material.SetInt("texture2", 1);
		}
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));

		shared_ptr<Actor> actor(new Actor("Prefab Quad"));
		actor->AddComponent(renderer);

		return actor;
	}

	shared_ptr<Actor> Actor::PrefabCube()
	{
		Mesh mesh(36, DefaultAssets::cube_vertices);
		mesh.SetupAttributes(DefaultAssets::cube_attributes);

		Material material("Assets/Shader/Cube");
		{
			material.texture1 = LoadTexture("Assets/container2.png", true);
			material.texture2 = LoadTexture("Assets/container2_specular.png", true);

			material.Use();
			material.SetInt("material.diffuse", 0);
			material.SetInt("material.specular", 1);

			material.SetVec3("material.ambient", 1.0f, 0.5f, 0.31f);
			material.SetVec3("material.specular", 0.5f, 0.5f, 0.5f);
			material.SetFloat("material.shininess", 32.0f);
		
			material.SetVec3("light.ambient", 0.2f, 0.2f, 0.2f);
			material.SetVec3("light.diffuse", 0.5f, 0.5f, 0.5f); // darkened
			material.SetVec3("light.specular", 1.0f, 1.0f, 1.0f);
			
			material.SetFloat("light.constant", 1.0f);
			material.SetFloat("light.linear", 0.09f);
			material.SetFloat("light.quadratic", 0.032f);
		}
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));

		shared_ptr<Actor> actor(new Actor("Prefab Cube"));
		actor->AddComponent(renderer);

		return actor;
	}

	shared_ptr<Actor> Actor::PrefabLight(LightType type)
	{
		Mesh mesh(36, DefaultAssets::cube_vertices);
		mesh.SetupAttributes(DefaultAssets::cube_attributes);

		Material material("Assets/Shader/Light");
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));

		shared_ptr<Light> light(new Light());
		light->type = type;

		shared_ptr<Actor> actor(new Actor("Prefab Light"));
		actor->AddComponent(renderer);
		actor->AddComponent(light);
		return actor;
	}
	*/

	shared_ptr<Actor> Actor::PrefabCamera()
	{
		shared_ptr<Actor> actor(new Actor("Prefab Camera"));
		shared_ptr<Camera> camera(new Camera());
		shared_ptr<PlayerController> controller(new PlayerController());
		actor->AddComponent(camera);
		actor->AddComponent(controller);
		return actor;
	}

	void Actor::Start()
	{
		for (const auto& c : components)
		{
			c->Start();
		}
	}

	void Actor::AddComponent(shared_ptr<Component> component)
	{
		component->actor = this;
		components.push_back(component);
	}

	void Actor::OnDestroy()
	{
		for (const auto& c : components)
		{
			c->OnDestroy();
		}
	}

	void Actor::Update()
	{
		for (const auto& c : components)
		{
			c->Update();
		}
	}
}