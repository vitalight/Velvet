#include "GameInstance.hpp"

#include <functional>

#include <fmt/core.h>

#include "Helper.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "RenderPipeline.hpp"
#include "GUI.hpp"
#include "Timer.hpp"
#include "VtEngine.hpp"

using namespace Velvet;

GameInstance::GameInstance(GLFWwindow* window, shared_ptr<GUI> gui)
{
	Global::game = this;

	// setup members
	m_window = window;
	m_gui = gui;
	m_renderPipeline = make_shared<RenderPipeline>();
}

shared_ptr<Actor> GameInstance::AddActor(shared_ptr<Actor> actor)
{
	m_actors.push_back(actor);
	return actor;
}

shared_ptr<Actor> GameInstance::CreateActor(const string& name)
{
	auto actor = shared_ptr<Actor>(new Actor(name));
	return AddActor(actor);
}

int GameInstance::Run()
{
	// Print actors
	{
		fmt::print("Total actors: {}\n", m_actors.size());
		for (auto actor : m_actors)
		{
			fmt::print(" + {}\n", actor->name);
			for (auto component : actor->components)
			{
				fmt::print(" |-- {}\n", component->name);
			}
		}
		fmt::print("\n");
	}

	Initialize();
	MainLoop();
	Finalize();

	return 0;
}

unsigned int GameInstance::depthFrameBuffer()
{
	return m_renderPipeline->depthTex;
}

glm::ivec2 Velvet::GameInstance::windowSize()
{
	glm::ivec2 result;
	glfwGetWindowSize(m_window, &result.x, &result.y);
	return result;
}

void GameInstance::ProcessMouse(GLFWwindow* m_window, double xpos, double ypos)
{
	for (auto callback : onMouseMove)
	{
		callback(xpos, ypos);
	}
}

void GameInstance::ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset)
{
	for (auto callback : onMouseScroll)
	{
		callback(xoffset, yoffset);
	}
}

void GameInstance::ProcessKeyboard(GLFWwindow* m_window)
{
	if (Global::input->GetKey(GLFW_KEY_ESCAPE))
	{
		glfwSetWindowShouldClose(m_window, true);
	}
	if (Global::input->GetKeyDown(GLFW_KEY_L))
	{
		renderWireframe = !renderWireframe;
	}
	if (Global::input->GetKeyDown(GLFW_KEY_P))
	{
		pause = !pause;
	}
	for (int i = 0; i < 9; i++)
	{
		if (Global::input->GetKeyDown(GLFW_KEY_1 + i))
		{
			Global::engine->SwitchScene(i);
		}
	}
}

void GameInstance::Initialize()
{
	for (const auto& go : m_actors)
	{
		go->Start();
	}
}

void GameInstance::MainLoop()
{
	fmt::print("Info(GameInstance): Initialization success. Enter main loop.\n");
	// render loop
	while (!glfwWindowShouldClose(m_window) && !pendingReset)
	{
		// input
		ProcessKeyboard(m_window);

		// rendering commands here
		glClearColor(skyColor.x, skyColor.y, skyColor.z, skyColor.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPolygonMode(GL_FRONT_AND_BACK, renderWireframe ? GL_LINE : GL_FILL);

		Timer::StartTimer("CPU_TIME");
		// timing
		float current = (float)glfwGetTime();
		deltaTime = current - lastUpdateTime;
		lastUpdateTime = current;

		m_gui->OnUpdate();

		//if (!pause)
		{
			frameCount++;
			elapsedTime += deltaTime;

			for (const auto& go : m_actors)
			{
				go->Update();
			}
			for (const auto& callback : postUpdate)
			{
				callback();
			}
		}

		Timer::EndTimer("CPU_TIME");

		m_renderPipeline->Render();
		m_gui->Render();

		// check and call events and swap the buffers
		glfwSwapBuffers(m_window);
		glfwPollEvents();

	}
}

void GameInstance::Finalize()
{
	for (const auto& go : m_actors)
	{
		go->OnDestroy();
	}
}

