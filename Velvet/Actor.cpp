#include "Actor.hpp"

#include "External/stb_image.h"

#include "DefaultAssets.hpp"
#include "Camera.hpp"
#include "PlayerController.hpp"
#include "Light.hpp"
#include "Resource.hpp"

namespace Velvet
{
	shared_ptr<Actor> Actor::PrefabLight(LightType type)
	{
		auto mesh = Resource::LoadMesh("cylinder.obj");

		auto material = Resource::LoadMaterial("Assets/Shader/_Light");
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

	Actor::Actor() {}

	Actor::Actor(string name) : name(name) {}
	
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