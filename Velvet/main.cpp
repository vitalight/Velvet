#include <iostream>

#include "VtGraphics.hpp"
#include "Global.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include "DefaultAssets.hpp"
#include "Resource.hpp"
#include "RenderPipeline.hpp"
#include "Input.hpp"

using namespace Velvet;

void CreateScene_BlinnPhong(VtGraphics& graphics)
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

	auto light = graphics.AddActor(Actor::PrefabLight(LightType::Directional));
	//light->GetComponent<MeshRenderer>()->hidden = true;
	light->transform->position = glm::vec3(-2.0, 4.0, -1.0f);

	graphics.postUpdate.push_back([light]() {
		light->transform->rotation += glm::vec3(1, 0, 0);
		});

	//=====================================
	// 3. Objects
	//=====================================
	Material material = Resource::LoadMaterial("BlinnPhong");
	{
		material.Use();
		material.SetTexture("floorTexture", Resource::LoadTexture("wood.png"));
		material.SetInt("blinn", 1);
	}

	shared_ptr<Actor> plane(new Actor("Plane"));
	{
		vector<float> planeVertices = {
			// positions            // normals         // texcoords
			 10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
			-10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
			-10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,

			 10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
			-10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,
			 10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,  10.0f, 10.0f
		};
		Mesh mesh({ 3, 3, 2 }, planeVertices);

		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));
		plane->AddComponent(renderer);
		plane->transform->position = glm::vec3(0, -0.1f, 0);
	}
	graphics.AddActor(plane);

	shared_ptr<Actor> cube(new Actor("Cube"));
	{
		Mesh mesh(DefaultAssets::cube_attributes, DefaultAssets::cube_vertices);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));
		cube->AddComponent(renderer);
	}
	graphics.AddActor(cube);

}

void CreateScene_Shadow(VtGraphics& graphics)
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

	auto light = graphics.AddActor(Actor::PrefabLight(LightType::SpotLight));
	//light->transform->position = glm::vec3(-2.0, 4.0, -1.0f);
	light->transform->position = glm::vec3(0, 4.0, -1.0f);
	light->transform->scale = glm::vec3(0.2f);
	auto lightComp = light->GetComponent<Light>();
	lightComp->innerCutoff = 50.0f;
	lightComp->outerCutoff = 60.0f;
	lightComp->ambient = 0.15;
	lightComp->linear = 0;
	lightComp->quadratic = 1.0;
	lightComp->constant = 0.001;

	graphics.postUpdate.push_back([light, lightComp]() {
		//light->transform->position = glm::vec3(sin(glfwGetTime()), 4.0, cos(glfwGetTime()));
		//light->transform->rotation = glm::vec3(20 * sin(glfwGetTime()) - 20, 0, 0);
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

	//=====================================
	// 3. Objects
	//=====================================

	Material material = Resource::LoadMaterial("_Default");
	{
		material.Use();

		material.SetTexture("material.diffuse", Resource::LoadTexture("wood.png"));
		material.SetTexture("_ShadowTex", graphics.depthMapFBO());
	}

	Material shadowMaterial = Resource::LoadMaterial("_ShadowDepth");

	auto plane = graphics.CreateActor("Plane");
	{
		vector<float> planeVertices = {
			// positions            // normals         // texcoords
			 10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
			-10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
			-10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,

			 10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
			-10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,
			 10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,  10.0f, 10.0f
		};
		Mesh mesh({ 3, 3, 2 }, planeVertices);

		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		plane->AddComponent(renderer);
	}

	auto cube1 = graphics.CreateActor("Cube1");
	{
		auto mesh = *Resource::LoadMesh("sphere.obj").get();
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		cube1->AddComponent(renderer);
		cube1->transform->position = glm::vec3(0.0f, 1.5f, 0.0);
		cube1->transform->scale = glm::vec3(0.5f);
	}

	auto cube2 = graphics.CreateActor("Cube2");
	{
		Mesh mesh(DefaultAssets::cube_attributes, DefaultAssets::cube_vertices);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		cube2->AddComponent(renderer);
		cube2->transform->position = glm::vec3(2.0f, 0, 1.0);
		cube2->transform->scale = glm::vec3(0.5f);
	}

	auto cube3 = graphics.CreateActor("Cube3");
	{
		Mesh mesh(DefaultAssets::cube_attributes, DefaultAssets::cube_vertices);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		cube3->AddComponent(renderer);
		cube3->transform->position = glm::vec3(-1.0f, 0, 2.0);
		cube3->transform->scale = glm::vec3(0.25f);
		cube3->transform->rotation = glm::vec3(60, 0, 60);
	}

	auto quad = graphics.CreateActor("Debug Quad");
	{
		Material debugMat = Resource::LoadMaterial("_ShadowDebug");
		{
			float near_plane = 1.0f, far_plane = 7.5f;
			debugMat.SetFloat("near_plane", near_plane);
			debugMat.SetFloat("far_plane", far_plane);
			debugMat.SetTexture("depthMap", graphics.depthMapFBO());
		}
		vector<float> quadVertices = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,

			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};

		Mesh quadMesh({ 3,2 }, quadVertices);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(quadMesh, debugMat));
		quad->AddComponent(renderer);
		renderer->hidden = true;

		graphics.postUpdate.push_back([renderer]() {
			if (Global::input->GetKeyDown(GLFW_KEY_1))
			{
				renderer->hidden = !renderer->hidden;
			}
		});
	}

	auto infPlane = graphics.CreateActor("InfPlane");
	{
		Material mat = Resource::LoadMaterial("_InfinitePlane");
		{
			mat.Use();
			mat.SetTexture("_ShadowTex", graphics.depthMapFBO());
		}
		Mesh mesh;
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, mat));
		infPlane->AddComponent(renderer);
	}
}

int main()
{
	//=====================================
	// 1. Create graphics
	//=====================================
	VtGraphics graphics;
	//graphics.skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);

	//=====================================
	// 2. Instantiate actors
	//=====================================
	
	//CreateScene_BlinnPhong(graphics);
	CreateScene_Shadow(graphics);

	//graphics.postUpdate.push_back([&]() {
	//	});
	
	//=====================================
	// 3. Run graphics
	//=====================================
	return graphics.Run();
}