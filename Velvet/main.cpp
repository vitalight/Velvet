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

glm::vec3 pointLightPositions[] = {
	glm::vec3(0.7f, 0.2f, 2.0f),
	glm::vec3(2.3f, -3.3f, -4.0f),
	glm::vec3(-4.0f, 2.0f, -12.0f),
	glm::vec3(0.0f, 0.0f, -3.0f)
};

void CreateScene_Plane(VtGraphics& graphics)
{
	auto camera = graphics.AddActor(Actor::PrefabCamera());
	camera->transform->position = glm::vec3(1.5, 1.5, 5.0);
	camera->transform->rotation = glm::vec3(-8.5, 9.0, 0);

	auto quad = graphics.AddActor(Actor::PrefabQuad());
	quad->transform->position = glm::vec3(0, -0.6f, 0);
	quad->transform->scale = glm::vec3(5, 5, 1);
	quad->transform->rotation = glm::vec3(90, 0, 0);
}

void CreateScene_Tutorial(VtGraphics& graphics)
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

	//auto light = graphics.AddActor(Actor::PrefabLight(LightType::Directional));
	//light->transform->position = glm::vec3(1.2f, 1.0f, 2.0f);
	//light->transform->scale = glm::vec3(0.2f);
	//shared_ptr<Light> light(new Light());
	//light->type = LightType::SpotLight;
	//camera->AddComponent(light);

	auto light = graphics.AddActor(Actor::PrefabLight(LightType::Directional));
	light->GetComponent<MeshRenderer>()->hidden = true;

	for (int i = 0; i < 4; i++)
	{
	    light = graphics.AddActor(Actor::PrefabLight(LightType::Point));
		light->transform->position = pointLightPositions[i];
		light->transform->scale = glm::vec3(0.2f);
	}

	light = graphics.AddActor(Actor::PrefabLight(LightType::SpotLight));
	light->GetComponent<MeshRenderer>()->hidden = true;

	//=====================================
	// 3. Objects
	//=====================================
	for (int i = 0; i < 10; i++)
	{
		auto cube = graphics.AddActor(Actor::PrefabCube());
		cube->transform->position = cubePositions[i];
		cube->transform->rotation = glm::vec3(1.0f, 0.3f, 0.5f) * (20.0f * i);
	}

}

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
	Material material("Assets/Shader/BlinnPhong");
	{
		material.texture1 = Resource::LoadTexture("Assets/Texture/wood.png");

		material.Use();
		material.SetInt("floorTexture", 0);
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
		Mesh mesh(6, planeVertices);
		mesh.SetupAttributes({ 3, 3, 2 });


		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));
		plane->AddComponent(renderer);
		plane->transform->position = glm::vec3(0, -0.1f, 0);
	}
	graphics.AddActor(plane);

	shared_ptr<Actor> cube(new Actor("Cube"));
	{
		Mesh mesh(36, DefaultAssets::cube_vertices);
		mesh.SetupAttributes(DefaultAssets::cube_attributes);
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

	auto light = graphics.AddActor(Actor::PrefabLight(LightType::Directional));
	//light->GetComponent<MeshRenderer>()->hidden = true;
	light->transform->position = glm::vec3(-2.0, 4.0, -1.0f);
	light->transform->scale = glm::vec3(0.2);

	graphics.postUpdate.push_back([light]() {
		light->transform->rotation += glm::vec3(1, 0, 0);
		});

	//=====================================
	// 3. Objects
	//=====================================
	Material material("Assets/Shader/Shadow");
	{
		material.texture1 = Resource::LoadTexture("Assets/Texture/wood.png");
		material.texture2 = graphics.m_pipeline->depthMapFBO;
		material.Use();
		material.SetInt("diffuseTexture", 0);
		material.SetInt("shadowMap", 1);
	}

	Material shadowMaterial("Assets/Shader/ShadowDepth");

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
		Mesh mesh(6, planeVertices);
		mesh.SetupAttributes({ 3, 3, 2 });

		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		plane->AddComponent(renderer);
		//plane->transform->position = glm::vec3(0, -0.1f, 0);
		graphics.AddActor(plane);
	}

	shared_ptr<Actor> cube1(new Actor("Cube1"));
	{
		Mesh mesh(36, DefaultAssets::cube_vertices);
		mesh.SetupAttributes(DefaultAssets::cube_attributes);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		cube1->AddComponent(renderer);
		cube1->transform->position = glm::vec3(0.0f, 1.5f, 0.0);
		cube1->transform->scale = glm::vec3(0.5f);
		graphics.AddActor(cube1);
	}

	shared_ptr<Actor> cube2(new Actor("Cube2"));
	{
		Mesh mesh(36, DefaultAssets::cube_vertices);
		mesh.SetupAttributes(DefaultAssets::cube_attributes);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		cube2->AddComponent(renderer);
		cube2->transform->position = glm::vec3(2.0f, 0.0f, 1.0);
		cube2->transform->scale = glm::vec3(0.5f);
		graphics.AddActor(cube2);
	}

	shared_ptr<Actor> cube3(new Actor("Cube3"));
	{
		Mesh mesh(36, DefaultAssets::cube_vertices);
		mesh.SetupAttributes(DefaultAssets::cube_attributes);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material, shadowMaterial));
		cube3->AddComponent(renderer);
		cube3->transform->position = glm::vec3(-1.0f, 0.0f, 2.0);
		cube3->transform->scale = glm::vec3(0.25f);
		cube3->transform->rotation = glm::vec3(60, 0, 60);
		graphics.AddActor(cube3);
	}

	shared_ptr<Actor> quad(new Actor("Debug Quad"));
	{
		Material debugMat("Assets/Shader/ShadowDebug");
		{
			float near_plane = 1.0f, far_plane = 7.5f;
			debugMat.SetFloat("near_plane", near_plane);
			debugMat.SetFloat("far_plane", far_plane);
			debugMat.SetInt("depthMap", 0);
			debugMat.texture1 = graphics.m_pipeline->depthMapFBO;
			//debugMat.texture1 = Resource::LoadTexture("Assets/Texture/wood.png");
		}
		vector<float> quadVertices = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		vector<unsigned int> quadIndices = {
			0,1,2,
			1,2,3,
		};
		Mesh quadMesh(6, quadVertices, quadIndices);
		quadMesh.SetupAttributes({ 3, 2 });
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(quadMesh, debugMat));
		quad->AddComponent(renderer);
		renderer->hidden = true;
		graphics.AddActor(quad);

		graphics.postUpdate.push_back([renderer]() {
			if (Global::input->GetKeyDown(GLFW_KEY_1))
			{
				renderer->hidden = !renderer->hidden;
			}
		});
	}
}

int main()
{
	//=====================================
	// 1. Create graphics
	//=====================================
	VtGraphics graphics;
	graphics.skyColor = glm::vec4(0.2f, 0.3f, 0.3f, 1.0f);

	//=====================================
	// 2. Instantiate actors
	//=====================================
	
	//CreateScene_Tutorial(graphics);
	//CreateScene_Plane(graphics);
	//CreateScene_BlinnPhong(graphics);
	CreateScene_Shadow(graphics);

	//graphics.postUpdate.push_back([&]() {
	//	});
	
	// TODO: add onDestroy callback
	
	//=====================================
	// 3. Run graphics
	//=====================================
	return graphics.Run();
}