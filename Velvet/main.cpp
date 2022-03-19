#include <iostream>

#include "VtGraphics.hpp"
#include "Global.hpp"
#include "Light.hpp"
#include "Camera.hpp"

using namespace Velvet;

glm::vec3 cubePositions[] = {
	glm::vec3(0.0f,  0.0f,  0.0f),
	glm::vec3(2.0f,  5.0f, -15.0f),
	glm::vec3(-1.5f, -2.2f, -2.5f),
	glm::vec3(-3.8f, -2.0f, -12.3f),
	glm::vec3(2.4f, -0.4f, -3.5f),
	glm::vec3(-1.7f,  3.0f, -7.5f),
	glm::vec3(1.3f, -2.0f, -2.5f),
	glm::vec3(1.5f,  2.0f, -2.5f),
	glm::vec3(1.5f,  0.2f, -1.5f),
	glm::vec3(-1.3f,  1.0f, -1.5f)
};

glm::vec3 pointLightPositions[] = {
	glm::vec3(0.7f, 0.2f, 2.0f),
	glm::vec3(2.3f, -3.3f, -4.0f),
	glm::vec3(-4.0f, 2.0f, -12.0f),
	glm::vec3(0.0f, 0.0f, -3.0f)
};

void CreateScene_Model(VtGraphics& graphics)
{
	auto camera = graphics.AddActor(Actor::PrefabCamera());
	camera->transform->position = glm::vec3(1.5, 1.5, 5.0);
	camera->transform->rotation = glm::vec3(-8.5, 9.0, 0);

	shared_ptr<Actor> actor(new Actor("Backpack"));
	Model backpack("Assets/Model/backpack.obj");
	Material material("Assets/Shader/model");
	shared_ptr<MeshRenderer> renderer(new MeshRenderer(backpack, material));
	actor->AddComponent(renderer);
	graphics.AddActor(actor);
}

void CreateScene_Plane(VtGraphics& graphics)
{
	auto camera = graphics.AddActor(Actor::PrefabCamera());
	camera->transform->position = glm::vec3(1.5, 1.5, 5.0);
	camera->transform->rotation = glm::vec3(-8.5, 9.0, 0);

	auto quad = graphics.AddActor(Actor::PrefabQuad());
	quad->transform->position = glm::vec3(0, -0.6f, 0);
	quad->transform->scale = glm::vec3(5, 5, 1);
	quad->transform->rotation = glm::vec3(90, 0, 0);
}

void CreateScene_Tutorial(VtGraphics& graphics)
{
	//=====================================
	// 1. Camera
	//=====================================
	auto camera = graphics.AddActor(Actor::PrefabCamera());
	camera->transform->position = glm::vec3(1.5, 1.5, 5.0);
	camera->transform->rotation = glm::vec3(-8.5, 9.0, 0);

	//=====================================
	// 2. Light
	//=====================================

	//auto light = graphics.AddActor(Actor::PrefabLight(LightType::Directional));
	//light->transform->position = glm::vec3(1.2f, 1.0f, 2.0f);
	//light->transform->scale = glm::vec3(0.2f);
	//shared_ptr<Light> light(new Light());
	//light->type = LightType::SpotLight;
	//camera->AddComponent(light);

	auto light = graphics.AddActor(Actor::PrefabLight(LightType::Directional));
	light->GetComponent<MeshRenderer>()->hidden = true;

	for (int i = 0; i < 4; i++)
	{
	    light = graphics.AddActor(Actor::PrefabLight(LightType::Point));
		light->transform->position = pointLightPositions[i];
		light->transform->scale = glm::vec3(0.2f);
	}

	light = graphics.AddActor(Actor::PrefabLight(LightType::SpotLight));
	light->GetComponent<MeshRenderer>()->hidden = true;

	//=====================================
	// 3. Objects
	//=====================================
	for (int i = 0; i < 10; i++)
	{
		auto cube = graphics.AddActor(Actor::PrefabCube());
		cube->transform->position = cubePositions[i];
		cube->transform->rotation = glm::vec3(1.0f, 0.3f, 0.5f) * (20.0f * i);
	}

}

// TODO: remove warnings

int main()
{
	//=====================================
	// 1. Create graphics
	//=====================================
	VtGraphics graphics;
	graphics.skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);

	//=====================================
	// 2. Instantiate actors
	//=====================================
	
	//CreateScene_Tutorial(graphics);
	//CreateScene_Plane(graphics);
	CreateScene_Model(graphics);
	
	//graphics.postUpdate.push_back([&]() {
	//	});
	
	// TODO: add onDestroy callback
	
	//=====================================
	// 3. Run graphics
	//=====================================
	return graphics.Run();
}