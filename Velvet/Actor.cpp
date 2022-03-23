#include "Actor.hpp"

#include "External/stb_image.h"

#include "DefaultAssets.hpp"
#include "Camera.hpp"
#include "PlayerController.hpp"
#include "Light.hpp"
#include "Resource.hpp"

namespace Velvet
{
	Actor::Actor() {}

	Actor::Actor(string name) : name(name) {}
	
	shared_ptr<Actor> Actor::PrefabQuad()
	{
		Mesh mesh(6, DefaultAssets::quad_vertices, DefaultAssets::quad_indices);
		mesh.SetupAttributes({3, 2});

		Material material("Assets/Shader/Quad");
		{
			material.texture1 = Resource::LoadTexture("Assets/Texture/container.jpg");
			material.texture2 = Resource::LoadTexture("Assets/Texture/awesomeface.png");

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

		Material material("Assets/Shader/_Default");
		{
			material.texture1 = Resource::LoadTexture("Assets/Texture/container2.png");
			material.texture2 = Resource::LoadTexture("Assets/Texture/container2_specular.png");

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