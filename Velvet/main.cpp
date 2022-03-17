#include <iostream>

#include "VtGraphics.hpp"

using namespace Velvet;


int main()
{
	// 1. Create graphics
	VtGraphics graphics;
	graphics.skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);

	// 2. Instantiate actors
	auto camera = graphics.AddActor(Actor::PrefabCamera());
	//camera->transform->position = glm::vec3(0.0f, 0.0f, 3.0f);
	camera->transform->position = glm::vec3(1.26, 1.26, 3.5);
	camera->transform->rotation = glm::vec3(-16, 12.5, 0);

	auto cube = graphics.AddActor(Actor::PrefabCube());

	auto quad = graphics.AddActor(Actor::PrefabQuad());
	quad->transform->position = glm::vec3(0, -0.6f, 0);
	quad->transform->scale = glm::vec3(5, 5, 1);
	quad->transform->rotation = glm::vec3(90, 0, 0);
	
	auto light = graphics.AddActor(Actor::PrefabLight());
	light->transform->position = glm::vec3(1.2f, 1.0f, 2.0f);
	light->transform->scale = glm::vec3(0.2f);

	graphics.postUpdate.push_back([&]() {
		//light->transform->position = glm::vec3(3.5*sin(graphics.elapsedTime), 1.26, 3.5 * cos(graphics.elapsedTime));
		//cube->transform->rotation = glm::vec3(0, 90 * graphics.elapsedTime, 0);
		});
	
	// 3. Run graphics
	return graphics.Run();
}