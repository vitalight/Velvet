#include <iostream>

#include "GameInstance.hpp"
#include "Global.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include "DefaultAssets.hpp"
#include "Resource.hpp"
#include "RenderPipeline.hpp"
#include "Input.hpp"
#include "Scene.hpp"
#include "VtEngine.hpp"

using namespace Velvet;

typedef shared_ptr<Scene> ScenePtr;

class SceneBlinnPhong : public Scene
{
public:
	SceneBlinnPhong() { name = "Basic / Rotating Light"; }

	void PopulateActors(GameInstance* game) override
	{
		Scene::PopulateCameraAndLight(game);
		Scene::PopulateDebug(game);
		game->skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);

		//=====================================
		// 3. Objects
		//=====================================
		auto material = Resource::LoadMaterial("_Default");
		{
			material->Use();

			material->SetTexture("material->diffuse", Resource::LoadTexture("wood.png"));
			material->SetTexture("_ShadowTex", game->depthFrameBuffer());
		}
		auto shadowMaterial = Resource::LoadMaterial("_ShadowDepth");

		shared_ptr<Actor> plane(new Actor("Plane"));
		{
			vector<float> planeVertices = {
				// positions            // normals         // texcoords
				 10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
				-10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,
				-10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,

				-10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,
				 10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
				 10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,  10.0f, 10.0f
			};
			auto mesh = make_shared<Mesh>(vector<unsigned int>{ 3, 3, 2 }, planeVertices);

			shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
			plane->AddComponent(renderer);
			plane->transform->position = glm::vec3(0, -0.1f, 0);
		}
		game->AddActor(plane);

		shared_ptr<Actor> cube(new Actor("Cube"));
		{
			auto mesh = make_shared<Mesh>(DefaultAssets::cube_attributes, DefaultAssets::cube_vertices);
			shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
			cube->AddComponent(renderer);
		}
		game->AddActor(cube);

	}
};

class SceneShadow : public Scene
{
public:
	SceneShadow() { name = "Basic / Premitive Rendering"; }

	void PopulateActors(GameInstance* game) override
	{
		Scene::PopulateCameraAndLight(game);
		Scene::PopulateDebug(game);

		//=====================================
		// 3. Objects
		//=====================================

		auto material = Resource::LoadMaterial("_Default");
		{
			material->Use();

			material->SetTexture("material->diffuse", Resource::LoadTexture("wood.png"));
			material->SetTexture("_ShadowTex", game->depthFrameBuffer());
		}

		auto shadowMaterial = Resource::LoadMaterial("_ShadowDepth");

		auto cube1 = game->CreateActor("Cube1");
		{
			auto mesh = Resource::LoadMesh("sphere.obj");
			shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
			cube1->AddComponent(renderer);
			cube1->transform->position = glm::vec3(0.6f, 2.0f, 0.0);
			cube1->transform->scale = glm::vec3(0.5f);
		}

		auto cube2 = game->CreateActor("Cube2");
		{
			auto mesh = make_shared<Mesh>(DefaultAssets::cube_attributes, DefaultAssets::cube_vertices);
			shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
			cube2->AddComponent(renderer);
			cube2->transform->position = glm::vec3(2.0f, 0.5, 1.0);
			cube2->transform->scale = glm::vec3(0.5f);
		}

		auto cube3 = game->CreateActor("Cube3");
		{
			auto mesh = make_shared<Mesh>(DefaultAssets::cube_attributes, DefaultAssets::cube_vertices);
			shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
			cube3->AddComponent(renderer);
			cube3->transform->position = glm::vec3(-1.0f, 0.5, 2.0);
			cube3->transform->scale = glm::vec3(0.25f);
			cube3->transform->rotation = glm::vec3(60, 0, 60);
		}

		auto infPlane = game->CreateActor("InfPlane");
		{
			auto mat = Resource::LoadMaterial("_InfinitePlane");
			{
				mat->Use();
				mat->SetTexture("_ShadowTex", game->depthFrameBuffer());
				// Plane: ax + by + cz + d = 0
				mat->SetVec4("_Plane", glm::vec4(0, 1, 0, 0));
			}
			auto mesh = make_shared<Mesh>();
			shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, mat));
			infPlane->AddComponent(renderer);
		}

	}
};

int main()
{
	//=====================================
	// 1. Create graphics
	//=====================================
	//shared_ptr<GameInstance> graphics(new GameInstance());
	//graphics.skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);
	auto engine = make_shared<VtEngine>();

	//=====================================
	// 2. Instantiate actors
	//=====================================
	
	vector<ScenePtr> scenes = {
		ScenePtr(new SceneBlinnPhong()),
		ScenePtr(new SceneShadow()),
	};
	engine->SetScenes(scenes);

	//=====================================
	// 3. Run graphics
	//=====================================
	return engine->Run();
}