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
#include "GeometryRenderer.hpp"

//#define SOLVER_CPU

namespace Velvet
{
	class Scene
	{
	public:
		std::string name = "BaseScene";

		virtual void PopulateActors(GameInstance* game) = 0;

		void ClearCallbacks() { onEnter.Clear(); onExit.Clear(); }

		VtCallback<void()> onEnter;
		VtCallback<void()> onExit;

	protected:
		template <class T>
		void ModifyParameter(T* ptr, T value)
		{
			onEnter.Register([this, ptr, value]() {
				T prev = *ptr;
				*ptr = value;
				onExit.Register([ptr, prev, value]() {
					//fmt::print("Revert ptr[{}] from {} to value {}\n", (int)ptr, value, prev);
					*ptr = prev;
					});
				});
		}

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
			//	light->transform->rotation = glm::vec3(10 * sin(Timer::elapsedTime()) - 10, 0, 0);
			//	light->transform->position = glm::vec3(2.5 * sin(Timer::elapsedTime()), 4.0, 2.5 * cos(Timer::elapsedTime()));
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

				game->postUpdate.Register([renderer]() {
					if (Global::input->GetKeyDown(GLFW_KEY_X))
					{
						renderer->enabled = !renderer->enabled;
					}
					});
			}
		}
	
		shared_ptr<Mesh> GenerateClothMesh(int resolution)
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
			return mesh;
		}

		shared_ptr<Mesh> GenerateClothMeshIrregular(int resolution)
		{
			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> uvs;
			vector<unsigned int> indices;
			const float clothSize = 2.0f;
			float noiseSize = 1.0f / resolution * 0.4;

			auto IsBoundary = [resolution](int x, int y) {
				return x == 0 || y == 0 || x == resolution || y == resolution;
			};

			auto VertexIndexAt = [resolution](int x, int y) {
				return x * (resolution + 1) + y;
			};

			auto Angle = [](glm::vec3 left, glm::vec3 mid, glm::vec3 right) {
				auto line1 = left - mid;
				auto line2 = right - mid;
				return acos(glm::dot(line1, line2));
			};

			for (int y = 0; y <= resolution; y++)
			{
				for (int x = 0; x <= resolution; x++)
				{
					glm::vec2 noise = IsBoundary(x,y)? glm::vec2(0) : noiseSize * glm::vec2(Helper::Random(), Helper::Random());
					glm::vec2 uv = noise + glm::vec2((float)x / (float)resolution, (float)y / (float)resolution);
					auto vertex = glm::vec3(uv.x - 0.5f, -uv.y, 0);

					vertices.push_back(clothSize * (vertex));
					normals.push_back(glm::vec3(0, 0, 1));
					uvs.push_back(uv);
				}
			}

			for (int y = 0; y < resolution; y++)
			{
				for (int x = 0; x < resolution; x++)
				{
					if (x < resolution && y < resolution)
					{
						auto pos1 = vertices[VertexIndexAt(x, y)];
						auto pos2 = vertices[VertexIndexAt(x+1, y)];
						auto pos3 = vertices[VertexIndexAt(x, y+1)];
						auto pos4 = vertices[VertexIndexAt(x+1, y+1)];

						auto angle1 = Angle(pos3, pos1, pos2);
						auto angle2 = Angle(pos1, pos2, pos4);
						auto angle3 = Angle(pos1, pos3, pos4);
						auto angle4 = Angle(pos3, pos4, pos2);

						if (angle1 + angle4 > angle2 + angle3)
						{
							indices.push_back(VertexIndexAt(x, y));
							indices.push_back(VertexIndexAt(x + 1, y+1));
							indices.push_back(VertexIndexAt(x, y + 1));

							indices.push_back(VertexIndexAt(x, y));
							indices.push_back(VertexIndexAt(x + 1, y));
							indices.push_back(VertexIndexAt(x + 1, y + 1));
						}
						else
						{
							indices.push_back(VertexIndexAt(x, y));
							indices.push_back(VertexIndexAt(x + 1, y));
							indices.push_back(VertexIndexAt(x, y + 1));

							indices.push_back(VertexIndexAt(x, y + 1));
							indices.push_back(VertexIndexAt(x + 1, y));
							indices.push_back(VertexIndexAt(x + 1, y + 1));
						}
					}
				}
			}
			auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
			return mesh;
		}


		shared_ptr<Actor> SpawnCloth(GameInstance* game, int resolution = 16)
		{
			auto cloth = game->CreateActor("Cloth Generated");

			auto material = Resource::LoadMaterial("_Default");
			material->Use();
			material->doubleSided = true;

			MaterialProperty materialProperty;
			materialProperty.preRendering = [](Material* mat) {
				mat->SetVec3("material.tint", glm::vec3(0.0f, 0.5f, 1.0f));
				mat->SetBool("material.useTexture", true);
				mat->SetTexture("material.diffuse", Resource::LoadTexture("fabric.jpg"));
				mat->specular = 0.01f;
			};

			//auto mesh = GenerateClothMesh(resolution);
			auto mesh = GenerateClothMeshIrregular(resolution);

			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			renderer->SetMaterialProperty(materialProperty);

#ifdef SOLVER_CPU
			auto clothObj = make_shared<VtClothObject>(resolution);
#else
			auto clothObj = make_shared<VtClothObjectGPU>(resolution);
#endif

			//auto prenderer = make_shared<ParticleRenderer>();
			auto prenderer = make_shared<GeometryRenderer>();

			cloth->AddComponents({ renderer, clothObj, prenderer });

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
			auto material = Resource::LoadMaterial("Assets/Shader/UnlitWhite");
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

			auto mat = Resource::LoadMaterial("InfinitePlane");
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