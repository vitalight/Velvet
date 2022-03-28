#pragma once

#include <string>
#include <functional>

#include "VtGraphics.hpp"

namespace Velvet
{
	class Scene
	{
	public:
		std::string name = "BaseScene";

		virtual void PopulateActors(VtGraphics* graphics)
		{
			PopulateCameraAndLight(graphics);
		}

		virtual void PopulateCameraAndLight(VtGraphics* graphics)
		{
			//=====================================
			// 1. Camera
			//=====================================
			auto camera = graphics->AddActor(Actor::PrefabCamera());
			camera->transform->position = glm::vec3(1.5, 1.5, 5.0);
			camera->transform->rotation = glm::vec3(-8.5, 9.0, 0);

			//=====================================
			// 2. Light
			//=====================================

			auto light = graphics->AddActor(Actor::PrefabLight(LightType::SpotLight));
			//light->transform->position = glm::vec3(-2.0, 4.0, -1.0f);
			light->transform->position = glm::vec3(0, 4.0, -1.0f);
			light->transform->scale = glm::vec3(0.2f);
			auto lightComp = light->GetComponent<Light>();

			graphics->postUpdate.push_back([light, lightComp]() {
				//light->transform->position = glm::vec3(sin(glfwGetTime()), 4.0, cos(glfwGetTime()));
				light->transform->rotation = glm::vec3(20 * sin(glfwGetTime()) - 20, 0, 0);
				if (Global::input->GetKeyDown(GLFW_KEY_UP))
				{
					fmt::print("Outer: {}\n", lightComp->outerCutoff++);
				}
				if (Global::input->GetKeyDown(GLFW_KEY_DOWN))
				{
					fmt::print("Outer: {}\n", lightComp->outerCutoff--);
				}
				if (Global::input->GetKeyDown(GLFW_KEY_RIGHT))
				{
					fmt::print("Inner: {}\n", lightComp->innerCutoff++);
				}
				if (Global::input->GetKeyDown(GLFW_KEY_LEFT))
				{
					fmt::print("Inner: {}\n", lightComp->innerCutoff--);
				}
				});
		}
	};
}