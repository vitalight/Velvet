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
#include "Resource.hpp"

using namespace Velvet;

GameInstance::GameInstance(GLFWwindow* window, shared_ptr<GUI> gui)
{
	Global::game = this;

	// setup members
	m_window = window;
	m_gui = gui;
	m_renderPipeline = make_shared<RenderPipeline>();
	m_timer = make_shared<Timer>();

	Timer::StartTimer("GAME_INSTANCE_INIT");
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
	if (0)
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

bool Velvet::GameInstance::windowMinimized()
{
	auto size = windowSize();
	return (size.x < 1 || size.y < 1);
}

void GameInstance::ProcessMouse(GLFWwindow* m_window, double xpos, double ypos)
{
	onMouseMove.Invoke(xpos, ypos);
}

void GameInstance::ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset)
{
	onMouseScroll.Invoke(xoffset, yoffset);
}

void GameInstance::ProcessKeyboard(GLFWwindow* m_window)
{
	Global::input->ToggleOnKeyDown(GLFW_KEY_H, Global::gameState.hideGUI);

	if (Global::input->GetKey(GLFW_KEY_ESCAPE))
	{
		glfwSetWindowShouldClose(m_window, true);
	}
	if (Global::input->GetKeyDown(GLFW_KEY_O))
	{
		Global::gameState.step = true;
		Global::gameState.pause = false;
	}
	for (int i = 0; i < 9; i++)
	{
		if (Global::input->GetKeyDown(GLFW_KEY_1 + i))
		{
			Global::engine->SwitchScene(i);
		}
	}
	if (Global::input->GetKeyDown(GLFW_KEY_R))
	{
		Global::engine->Reset();
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
	double initTime = Timer::EndTimer("GAME_INSTANCE_INIT") * 1000;
	fmt::print("Info(GameInstance): Initialization success within {:.2f} ms. Enter main loop.\n", initTime);
	// render loop
	while (!glfwWindowShouldClose(m_window) && !pendingReset)
	{
		if (windowMinimized())
		{
			glfwPollEvents();
			continue;
		}
		// Input
		ProcessKeyboard(m_window);

		// Init
		glClearColor(skyColor.x, skyColor.y, skyColor.z, skyColor.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPolygonMode(GL_FRONT_AND_BACK, Global::gameState.renderWireframe ? GL_LINE : GL_FILL);

		Timer::StartTimer("CPU_TIME");
		Timer::UpdateDeltaTime();

		// Logic Updates
		if (!Global::gameState.hideGUI) m_gui->OnUpdate();

		if (!Global::gameState.pause)
		{
			Timer::NextFrame();
			if (Timer::NextFixedFrame())
			{
				for (const auto& go : m_actors) go->FixedUpdate();

				animationUpdate.Invoke();

				if (Global::gameState.step)
				{
					Global::gameState.pause = true;
					Global::gameState.step = false;
				}
			}

			for (const auto& go : m_actors) go->Update();
		}

		Global::input->OnUpdate();

		godUpdate.Invoke();

		Timer::EndTimer("CPU_TIME");

		// Render
		m_renderPipeline->Render();
		if (!Global::gameState.hideGUI) m_gui->Render();

		// Check and call events and swap the buffers
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

