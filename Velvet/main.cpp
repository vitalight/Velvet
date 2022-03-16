#include <iostream>

#include "VtGraphics.h"

using namespace Velvet;

int main()
{
	// 1. Create graphics
	VtGraphics graphics;
	graphics.skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);

	// 2. Instantiate actors
	auto camera = graphics.AddActor(Actor::PrefabCamera());
	
	auto cube1 = graphics.AddActor(Actor::PrefabCube());
	
	auto light = graphics.AddActor(Actor::PrefabLight());
	light->transform.position = glm::vec3(1.2f, 1.0f, 2.0f);
	light->transform.scale = glm::vec3(0.2f);

	// 3. Run graphics
	return graphics.Run();
}