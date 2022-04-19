#include <iostream>

#include "VtEngine.hpp"
#include "GameInstance.hpp"
#include "Resource.hpp"
#include "Scene.hpp"
#include "Helper.hpp"

using namespace Velvet;

class ScenePremitiveRendering : public Scene
{
public:
	ScenePremitiveRendering() { name = "Basic / Premitive Rendering"; }

	void PopulateActors(GameInstance* game) override
	{
		Scene::SpawnCameraAndLight(game);

		//=====================================
		// 3. Objects
		//=====================================

		auto material = Resource::LoadMaterial("_Default");
		{
			material->Use();

			material->SetTexture("material.diffuse", Resource::LoadTexture("wood.png"));
			material->SetBool("material.useTexture", true);
		}

		auto cube1 = game->CreateActor("Sphere");
		{
			auto mesh = Resource::LoadMesh("sphere.obj");
			shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, true));
			cube1->AddComponent(renderer);
			cube1->transform->position = glm::vec3(0.6f, 2.0f, 0.0);
			cube1->transform->scale = glm::vec3(0.5f);
		}

		auto cube2 = game->CreateActor("Cube2");
		{
			auto mesh = Resource::LoadMesh("cube.obj");
			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			cube2->AddComponent(renderer);
			cube2->transform->position = glm::vec3(2.0f, 0.5, 1.0);
		}

		auto cube3 = game->CreateActor("Cube3");
		{
			auto mesh = Resource::LoadMesh("cube.obj");
			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			cube3->AddComponent(renderer);
			cube3->Initialize(glm::vec3(-1.0f, 0.5, 2.0),
				glm::vec3(0.5f),
				glm::vec3(60, 0, 60));
		}

		SpawnInfinitePlane(game);
	}
};

class SceneColoredCubes : public Scene
{
public:
	SceneColoredCubes() { name = "Basic / Colored Cubes"; }

	void PopulateActors(GameInstance* game)  override
	{
		Scene::SpawnCameraAndLight(game);

		Scene::SpawnInfinitePlane(game);

		{
			auto whiteCube = SpawnColoredCube(game, glm::vec3(1.0, 1.0, 1.0));
			whiteCube->Initialize(glm::vec3(0, 0.25, 0),
				glm::vec3(2, 0.5f, 2));
		}

		vector<glm::vec3> colors = {
			glm::vec3(0.0f, 0.5f, 1.0f),
			glm::vec3(0.797f, 0.354f, 0.000f),
			glm::vec3(0.000f, 0.349f, 0.173f),
			glm::vec3(0.875f, 0.782f, 0.051f),
			glm::vec3(0.01f, 0.170f, 0.453f),
			glm::vec3(0.673f, 0.111f, 0.000f),
			glm::vec3(0.612f, 0.194f, 0.394f)
		};

		vector<shared_ptr<Actor>> cubes;
		static vector<glm::vec3> velocities;
		for (int i = 0; i < 50; i++)
		{
			glm::vec3 color = colors[(int)Helper::Random(0.0f, (float)colors.size())];
			auto cube = SpawnColoredCube(game, color);
			cube->Initialize(glm::vec3(Helper::Random(-3.0f, 3.0f), Helper::Random(0.3f, 0.5f), Helper::Random(-3.0f, 3.0f)), 
				glm::vec3(0.3f));
			cubes.push_back(cube);
			velocities.push_back(glm::vec3(0.0));
		}

		game->animationUpdate.Register([cubes, game]() {
			for (int i = 0; i < cubes.size(); i++)
			{
				auto cube = cubes[i];
				//cube->transform->position += Helper::RandomUnitVector() * Timer::deltaTime() * 5.0f;
				velocities[i] = Helper::Lerp(velocities[i], Helper::RandomUnitVector() * 1.0f, Timer::fixedDeltaTime());
				cube->transform->rotation += Helper::RandomUnitVector() * Timer::fixedDeltaTime() * 50.0f;
				cube->transform->position += velocities[i] * Timer::fixedDeltaTime() * 5.0f;

				if (cube->transform->position.y < 0.07f)
				{
					cube->transform->position.y = 0.07f;
				}
				if (glm::length(cube->transform->position) > 3.0f)
				{
					cube->transform->position = cube->transform->position / glm::length(cube->transform->position) * 3.0f;
				}
			}
			});
	}
};


class SceneClothAttach : public Scene
{
public:
	SceneClothAttach() { name = "Cloth / Attach"; }

	void PopulateActors(GameInstance* game)  override
	{
		SpawnCameraAndLight(game);
		SpawnInfinitePlane(game);

		auto sphere = SpawnSphere(game);
		float radius = 0.5f;
		sphere->Initialize(glm::vec3(0, radius, 0), glm::vec3(radius));

		int clothResolution = 40;
		auto cloth = SpawnCloth(game, clothResolution);
		cloth->Initialize(glm::vec3(0.0f, 1.5f, 1.0f), glm::vec3(1.0), glm::vec3(90, 0, 0));

		auto clothObj = cloth->GetComponent<VtClothObjectGPU>();
		if (clothObj) clothObj->SetAttachedIndices({ 0, clothResolution, (clothResolution + 1) * (clothResolution + 1) - 1, (clothResolution + 1) * (clothResolution) });

	}
};

class SceneClothCollision : public Scene
{
public:
	SceneClothCollision() { name = "Cloth / SDF Collision"; }

	void PopulateActors(GameInstance* game)  override
	{
		SpawnCameraAndLight(game);

		SpawnInfinitePlane(game);

		auto sphere = SpawnSphere(game);
		float radius = 0.6f;
		sphere->Initialize(glm::vec3(0, radius, -1), glm::vec3(radius));
		game->animationUpdate.Register([sphere, game, radius]() {
			float time = Timer::fixedDeltaTime() * Timer::physicsFrameCount();
			sphere->transform->position = glm::vec3(0, radius, -cos(time * 2));
			});

		int clothResolution = 16;
		auto cloth = SpawnCloth(game, clothResolution, 2);
		cloth->Initialize(glm::vec3(0, 2.5f, 0), glm::vec3(1.0));
#ifdef SOLVER_CPU
		auto clothObj = cloth->GetComponent<VtClothObject>();
#else
		auto clothObj = cloth->GetComponent<VtClothObjectGPU>();
#endif
		if (clothObj) clothObj->SetAttachedIndices({ 0, clothResolution });
	}
};

class SceneClothSelfCollision : public Scene
{
public:
	SceneClothSelfCollision() { name = "Cloth / Self Collision"; }

	void PopulateActors(GameInstance* game)  override
	{
		SpawnCameraAndLight(game);
		SpawnInfinitePlane(game);

		ModifyParameter(&Global::simParams.numSubsteps, 3);
		ModifyParameter(&Global::simParams.numSubsteps, 8);
		ModifyParameter(&Global::simParams.friction, 0.3f);

		int clothResolution = 60;
		auto cloth = SpawnCloth(game, clothResolution, 1);
		cloth->Initialize(glm::vec3(0.0f, 1.5f, 1.0f), glm::vec3(1.0), glm::vec3(-15, 10, 10));

		auto clothObj = cloth->GetComponent<VtClothObjectGPU>();
	}
};

class SceneClothFriction : public Scene
{
public:
	SceneClothFriction() { name = "Cloth / Friction"; }

	void PopulateActors(GameInstance* game)  override
	{
		SpawnCameraAndLight(game);
		SpawnInfinitePlane(game);

		ModifyParameter(&Global::simParams.friction, 0.6f);
		ModifyParameter(&Global::simParams.numSubsteps, 5);
		ModifyParameter(&Global::simParams.numIterations, 5);

		auto sphere = SpawnSphere(game);
		float radius = 0.5f;
		sphere->Initialize(glm::vec3(0, radius, 0), glm::vec3(radius));

		game->animationUpdate.Register([sphere, game, radius]() {
			float time = Timer::physicsFrameCount() * Timer::fixedDeltaTime() - 0.5f;
			if (time > 0)
			{
				sphere->transform->position = glm::vec3(sin(time), radius, 0);
			}

			if ((int)time % 4 > 1)
			{
				sphere->transform->rotation = glm::vec3(0, -time * 180, 0);
			}
			else
			{
				sphere->transform->rotation = glm::vec3(0, time * 180, 0);
			}
			});

		int clothResolution = 64;
		auto cloth = SpawnCloth(game, clothResolution, 2);
		cloth->Initialize(glm::vec3(0.0f, 1.5f, 1.0f), glm::vec3(1.0), glm::vec3(90, 0, 0));
	}
};

class SceneClothMultiple : public Scene
{
public:
	SceneClothMultiple() { name = "Cloth / Multiple Object"; }

	void PopulateActors(GameInstance* game)  override
	{
		SpawnCameraAndLight(game);
		SpawnInfinitePlane(game);

		ModifyParameter(&Global::simParams.friction, 0.6f);
		ModifyParameter(&Global::simParams.numSubsteps, 5);
		ModifyParameter(&Global::simParams.numIterations, 5);

		auto cube = SpawnColoredCube(game);
		float radius = 1.0f;
		cube->Initialize(glm::vec3(0, 0.5 * radius, 0), glm::vec3(radius));

		auto solverActor = game->CreateActor("ClothSolver");
		auto solver = make_shared<VtClothSolverGPU>();
		solverActor->AddComponent(solver);

		int clothResolution = 64;
		{
			auto cloth = SpawnCloth(game, clothResolution, 1, solver);
			cloth->Initialize(glm::vec3(0.0f, 1.5f, 1.0f), glm::vec3(1.0), glm::vec3(90, 0, 0));
		}
		{
			auto cloth = SpawnCloth(game, clothResolution, 2, solver);
			cloth->Initialize(glm::vec3(0.0f, 1.8f, 1.0f), glm::vec3(1.0), glm::vec3(90, 0, 0));
		}
		{
			auto cloth = SpawnCloth(game, clothResolution, 3, solver);
			cloth->Initialize(glm::vec3(0.0f, 2.1f, 1.0f), glm::vec3(1.0), glm::vec3(90, 0, 0));
		}
	}
};

class SceneClothHD : public Scene
{
public:
	SceneClothHD() { name = "Cloth / High Resolution"; }

	void PopulateActors(GameInstance* game)  override
	{
		SpawnCameraAndLight(game);
		SpawnInfinitePlane(game);

		ModifyParameter(&Global::simParams.numSubsteps, 10);
		ModifyParameter(&Global::simParams.numIterations, 10);

		auto sphere = SpawnSphere(game);
		float radius = 0.6f;
		sphere->Initialize(glm::vec3(0, radius, 0), glm::vec3(radius));

		int clothResolution = 200;
		auto cloth = SpawnCloth(game, clothResolution, 2);
		cloth->Initialize(glm::vec3(0.0f, 1.5f, 1.0f), glm::vec3(1.0), glm::vec3(90, 0, 0));
	}
};

class SceneClothSwirl : public Scene
{
public:
	SceneClothSwirl() { name = "Cloth / Swirl"; }

	void PopulateActors(GameInstance* game)  override
	{
		SpawnCameraAndLight(game);
		SpawnInfinitePlane(game);

		auto sphere = SpawnSphere(game);
		float radius = 0.1f;
		sphere->GetComponent<Collider>()->enabled = false;
		sphere->Initialize(glm::vec3(0, radius, 0), glm::vec3(radius));

		int clothResolution = 36;
		auto cloth = SpawnCloth(game, clothResolution);
		cloth->Initialize(glm::vec3(0.0f, 1.5f, 1.0f), glm::vec3(1.0), glm::vec3(90, 0, 0));

		auto clothObj = cloth->GetComponent<VtClothObjectGPU>();
		if (clothObj) clothObj->SetAttachedIndices({ 0});

		game->animationUpdate.Register([clothObj, sphere]() {
			float time = Timer::fixedDeltaTime() * Timer::physicsFrameCount() * 3;
			auto pos = glm::vec3(sin(time), cos(time) + 2, 0);
			clothObj->attachSlotPositions()[0] = pos;
			sphere->transform->position = pos;
			});
	}
};

int main()
{
	//=====================================
	// 1. Create graphics
	//=====================================
	auto engine = make_shared<VtEngine>();

	//=====================================
	// 2. Instantiate actors
	//=====================================
	
	vector<shared_ptr<Scene>> scenes = {
		make_shared<SceneClothAttach>(),
		make_shared<SceneClothCollision>(),
		make_shared<SceneClothSelfCollision>(),
		make_shared<SceneClothFriction>(),
		make_shared<SceneClothMultiple>(),
		make_shared<SceneClothHD>(),
		make_shared<SceneClothSwirl>(),
		//make_shared<SceneColoredCubes>(),
		//make_shared<ScenePremitiveRendering>(),
	};
	engine->SetScenes(scenes);

	//=====================================
	// 3. Run graphics
	//=====================================
	return engine->Run();
}