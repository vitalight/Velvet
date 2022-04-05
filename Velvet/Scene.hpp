#pragma once

#include <string>
#include <functional>

#include "GameInstance.hpp"
#include "Input.hpp"
#include "Resource.hpp"
#include "Actor.hpp"
#include "PlayerController.hpp"
#include "MeshRenderer.hpp"
#include "MaterialProperty.hpp"
#include "Collider.hpp"
#include "VtClothObject.hpp"
#include "VtClothObjectGPU.hpp"
#include "ParticleRenderer.hpp"

namespace Velvet
{
	class Scene
	{
	public:
		std::string name = "BaseScene";

		virtual void PopulateActors(GameInstance* game) = 0;

	protected:
		void SpawnCameraAndLight(GameInstance* game)
		{
			//=====================================
			// 1. Camera
			//=====================================
			auto camera = SpawnCamera(game);
			camera->Initialize(glm::vec3(0.35, 3.3, 7.2),
				glm::vec3(1),
				glm::vec3(-21, 2.25, 0));

			//=====================================
			// 2. Light
			//=====================================

			auto light = SpawnLight(game);
			light->Initialize(glm::vec3(2.5f, 5.0f, 2.5f), 
				glm::vec3(0.2f),
				glm::vec3(20, 30, 0));
			auto lightComp = light->GetComponent<Light>();

			SpawnDebug(game);

			//game->postUpdate.push_back([light, lightComp, game]() {
			//	//light->transform->position = glm::vec3(sin(glfwGetTime()), 4.0, cos(glfwGetTime()));
			//	light->transform->rotation = glm::vec3(10 * sin(game->elapsedTime) - 10, 0, 0);
			//	light->transform->position = glm::vec3(2.5 * sin(game->elapsedTime), 4.0, 2.5 * cos(game->elapsedTime));
			//	if (Global::input->GetKeyDown(GLFW_KEY_UP))
			//	{
			//		fmt::print("Outer: {}\n", lightComp->outerCutoff++);
			//	}
			//	if (Global::input->GetKeyDown(GLFW_KEY_DOWN))
			//	{
			//		fmt::print("Outer: {}\n", lightComp->outerCutoff--);
			//	}
			//	if (Global::input->GetKeyDown(GLFW_KEY_RIGHT))
			//	{
			//		fmt::print("Inner: {}\n", lightComp->innerCutoff++);
			//	}
			//	if (Global::input->GetKeyDown(GLFW_KEY_LEFT))
			//	{
			//		fmt::print("Inner: {}\n", lightComp->innerCutoff--);
			//	}
			//	});
		}
	
		void SpawnDebug(GameInstance* game)
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
				renderer->enabled = false;

				game->postUpdate.push_back([renderer]() {
					if (Global::input->GetKeyDown(GLFW_KEY_X))
					{
						renderer->enabled = !renderer->enabled;
					}
					});
			}
		}
	
		shared_ptr<Actor> SpawnCloth(GameInstance* game, int resolution = 16)
		{
			auto cloth = game->CreateActor("Cloth Generated");

			auto material = Resource::LoadMaterial("_Default");
			material->Use();
			material->SetTexture("_ShadowTex", game->depthFrameBuffer());
			material->doubleSided = true;

			MaterialProperty materialProperty;
			materialProperty.preRendering = [](Material* mat) {
				mat->SetVec3("material.tint", glm::vec3(0.0f, 0.5f, 1.0f));
				mat->SetBool("material.useTexture", true);
				mat->SetTexture("material.diffuse", Resource::LoadTexture("fabric.jpg"));
				mat->specular = 0.01f;
			};

			{
				vector<glm::vec3> vertices;
				vector<glm::vec3> normals;
				vector<glm::vec2> uvs;
				vector<unsigned int> indices;
				const float clothSize = 2.0f;

				for (int y = 0; y <= resolution; y++)
				{
					for (int x = 0; x <= resolution; x++)
					{
						vertices.push_back(clothSize * glm::vec3((float)x / (float)resolution - 0.5f, -(float)y / (float)resolution, 0));
						normals.push_back(glm::vec3(0, 0, 1));
						uvs.push_back(glm::vec2((float)x / (float)resolution, (float)y / (float)resolution));
					}
				}

				auto VertexIndexAt = [resolution](int x, int y) {
					return x * (resolution + 1) + y;
				};

				for (int x = 0; x < resolution; x++)
				{
					for (int y = 0; y < resolution; y++)
					{
						indices.push_back(VertexIndexAt(x, y));
						indices.push_back(VertexIndexAt(x + 1, y));
						indices.push_back(VertexIndexAt(x, y + 1));

						indices.push_back(VertexIndexAt(x, y + 1));
						indices.push_back(VertexIndexAt(x + 1, y));
						indices.push_back(VertexIndexAt(x + 1, y + 1));
					}
				}
				auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);

				auto renderer = make_shared<MeshRenderer>(mesh, material, true);
				renderer->SetMaterialProperty(materialProperty);
				auto clothObj = make_shared<VtClothObject>(resolution);
				auto prenderer = make_shared<ParticleRenderer>();

				cloth->AddComponents({ renderer, clothObj, prenderer });
			}
			return cloth;
		}

		shared_ptr<Actor> SpawnSphere(GameInstance* game)
		{
			auto sphere = game->CreateActor("Sphere");
			MaterialProperty materialProperty;
			materialProperty.preRendering = [](Material* mat) {
				mat->SetVec3("material.tint", glm::vec3(1.0));
				mat->SetBool("material.useTexture", false);
			};

			auto material = Resource::LoadMaterial("_Default");
			{
				material->Use();
				material->SetTexture("_ShadowTex", game->depthFrameBuffer());
			}

			auto mesh = Resource::LoadMesh("sphere.obj");
			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			renderer->SetMaterialProperty(materialProperty);
			auto collider = make_shared<Collider>(false);
			sphere->AddComponents({ renderer, collider });
			return sphere;
		}

		shared_ptr<Actor> SpawnLight(GameInstance* game)
		{
			auto actor = game->CreateActor("Prefab Light");
			auto mesh = Resource::LoadMesh("cylinder.obj");
			auto material = Resource::LoadMaterial("Assets/Shader/_Light");
			auto renderer = make_shared<MeshRenderer>(mesh, material);
			auto light = make_shared<Light>();

			actor->AddComponents({ renderer, light });
			return actor;
		}

		shared_ptr<Actor> SpawnCamera(GameInstance* game)
		{
			auto actor = game->CreateActor("Prefab Camera");
			auto camera = make_shared<Camera>();
			auto controller = make_shared<PlayerController>();
			actor->AddComponents({ camera, controller });
			return actor;
		}

		shared_ptr<Actor> SpawnInfinitePlane(GameInstance* game)
		{
			auto infPlane = game->CreateActor("Infinite Plane");

			auto mat = Resource::LoadMaterial("_InfinitePlane");
			mat->Use();
			mat->SetTexture("_ShadowTex", game->depthFrameBuffer());
			mat->noWireframe = true;
			// Plane: ax + by + cz + d = 0
			mat->SetVec4("_Plane", glm::vec4(0, 1, 0, 0));

			const vector<glm::vec3> vertices = {
				glm::vec3(1,1,0), glm::vec3(-1,-1,0), glm::vec3(-1,1,0), glm::vec3(1,-1,0) };
			const vector<unsigned int> indices = { 2,1,0, 3, 0, 1 };

			auto mesh = make_shared<Mesh>(vertices, vector<glm::vec3>(), vector<glm::vec2>(), indices);
			auto renderer = make_shared<MeshRenderer>(mesh, mat);
			auto collider = make_shared<Collider>(true);
			infPlane->AddComponents({ renderer, collider });
			return infPlane;
		}
	
		shared_ptr<Actor> SpawnColoredCube(GameInstance* game, glm::vec3 color = glm::vec3(1.0f))
		{
			auto cube = game->CreateActor("Cube");
			auto material = Resource::LoadMaterial("_Default");
			material->Use();
			material->SetTexture("_ShadowTex", game->depthFrameBuffer());

			MaterialProperty materialProperty;
			materialProperty.preRendering = [color](Material* mat) {
				mat->SetVec3("material.tint", color);
				mat->SetBool("material.useTexture", false);
			};

			auto mesh = Resource::LoadMesh("cube.obj");
			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			renderer->SetMaterialProperty(materialProperty);

			cube->AddComponent(renderer);
			return cube;
		}
	};
}