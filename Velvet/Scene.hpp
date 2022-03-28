#pragma once

#include <string>
#include <functional>

#include "GameInstance.hpp"
#include "Input.hpp"
#include "Resource.hpp"
#include "Actor.hpp"

namespace Velvet
{
	class Scene
	{
	public:
		std::string name = "BaseScene";

		virtual void PopulateActors(GameInstance* game) = 0;

		void PopulateCameraAndLight(GameInstance* game)
		{
			//=====================================
			// 1. Camera
			//=====================================
			auto camera = game->AddActor(Actor::PrefabCamera());
			camera->transform->position = glm::vec3(1.5, 1.5, 5.0);
			camera->transform->rotation = glm::vec3(-8.5, 9.0, 0);

			//=====================================
			// 2. Light
			//=====================================

			auto light = game->AddActor(Actor::PrefabLight(LightType::SpotLight));
			//light->transform->position = glm::vec3(-2.0, 4.0, -1.0f);
			light->transform->position = glm::vec3(0, 4.0, -1.0f);
			light->transform->scale = glm::vec3(0.2f);
			auto lightComp = light->GetComponent<Light>();

			game->postUpdate.push_back([light, lightComp, game]() {
				//light->transform->position = glm::vec3(sin(glfwGetTime()), 4.0, cos(glfwGetTime()));
				light->transform->rotation = glm::vec3(10 * sin(game->elapsedTime) - 10, 0, 0);
				light->transform->position = glm::vec3(2.5 * sin(game->elapsedTime), 4.0, 2.5 * cos(game->elapsedTime));
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
	
		void PopulateDebug(GameInstance* game)
		{
			auto quad = game->CreateActor("Debug Quad");
			{
				auto debugMat = Resource::LoadMaterial("_ShadowDebug");
				{
					float near_plane = 1.0f, far_plane = 7.5f;
					debugMat->SetFloat("near_plane", near_plane);
					debugMat->SetFloat("far_plane", far_plane);
					debugMat->SetTexture("depthMap", game->depthFrameBuffer());
				}
				vector<float> quadVertices = {
					// positions        // texture Coords
					-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
					-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
					 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,

					-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
					 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
					 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
				};
				vector<unsigned int> attributes = { 3,2 };
				auto quadMesh = make_shared<Mesh>(attributes, quadVertices);
				shared_ptr<MeshRenderer> renderer(new MeshRenderer(quadMesh, debugMat));
				quad->AddComponent(renderer);
				renderer->hidden = true;

				game->postUpdate.push_back([renderer]() {
					if (Global::input->GetKeyDown(GLFW_KEY_1))
					{
						renderer->hidden = !renderer->hidden;
					}
					});
			}
		}
	};
}