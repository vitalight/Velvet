#include <iostream>

#include "VtGraphics.hpp"
#include "Global.hpp"
#include "Light.hpp"

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

int main()
{
	// 1. Create graphics
	VtGraphics graphics;
	//graphics.skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);

	// 2. Instantiate actors
	auto camera = graphics.AddActor(Actor::PrefabCamera());
	//camera->transform->position = glm::vec3(0.0f, 0.0f, 3.0f);
	camera->transform->position = glm::vec3(1.5, 1.5, 5.0);
	camera->transform->rotation = glm::vec3(-8.5, 9.0, 0);

	for (int i = 0; i < 10; i++)
	{
		auto cube = graphics.AddActor(Actor::PrefabCube());
		cube->transform->position = cubePositions[i];
		cube->transform->rotation = glm::vec3(1.0f, 0.3f, 0.5f) * (20.0f * i);
	}

	//auto quad = graphics.AddActor(Actor::PrefabQuad());
	//quad->transform->position = glm::vec3(0, -0.6f, 0);
	//quad->transform->scale = glm::vec3(5, 5, 1);
	//quad->transform->rotation = glm::vec3(90, 0, 0);
	
	//auto light = graphics.AddActor(Actor::PrefabLight(LightType::Directional));
	//light->transform->position = glm::vec3(1.2f, 1.0f, 2.0f);
	//light->transform->scale = glm::vec3(0.2f);
	shared_ptr<Light> light(new Light());
	light->type = LightType::SpotLight;
	camera->AddComponent(light);

	//graphics.postUpdate.push_back([&]() {
	//	});
	
	// 3. Run graphics
	return graphics.Run();
}