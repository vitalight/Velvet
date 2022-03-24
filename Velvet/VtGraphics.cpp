#include "VtGraphics.hpp"

#include <functional>

#include "External/stb_image.h"

#include "VtHelper.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "RenderPipeline.hpp"
#include "Config.hpp"

using namespace Velvet;

VtGraphics::VtGraphics()
{
	Global::graphics = this;

	// setup glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// Multi-sample Anti-aliasing
	glfwWindowHint(GLFW_SAMPLES, 4);

	m_window = glfwCreateWindow(Config::screenWidth, Config::screenHeight, "Velvet", NULL, NULL);

	if (m_window == NULL)
	{
		fmt::print("Failed to create GLFW window\n");
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(m_window);

	glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* m_window, int width, int height) {
		glViewport(0, 0, width, height);
		});

	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(m_window, [](GLFWwindow* m_window, double xpos, double ypos) {
		Global::graphics->ProcessMouse(m_window, xpos, ypos);
		});
	glfwSetScrollCallback(m_window, [](GLFWwindow* m_window, double xoffset, double yoffset) {
		Global::graphics->ProcessScroll(m_window, xoffset, yoffset);
		});

	// setup opengl
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		fmt::print("Failed to initialize GLAD\n");
		return;
	}
	glViewport(0, 0, Config::screenWidth, Config::screenHeight);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	// setup stbi
	stbi_set_flip_vertically_on_load(true);

	// setup members
	m_input = shared_ptr<Input>(new Input(m_window));
	m_renderPipeline = shared_ptr<RenderPipeline>(new RenderPipeline());
}

shared_ptr<Actor> VtGraphics::AddActor(shared_ptr<Actor> actor)
{
	m_actors.push_back(actor);
	return actor;
}

shared_ptr<Actor> VtGraphics::CreateActor(const string& name)
{
	auto actor = shared_ptr<Actor>(new Actor(name));
	return AddActor(actor);
}

int VtGraphics::Run()
{
	fmt::print("Hello Velvet!\n");
	fmt::print("# Total actors: {}\n", m_actors.size());
	for (auto actor : m_actors)
	{
		fmt::print(" + {}\n", actor->name);
		for (auto component : actor->components)
		{
			fmt::print(" |-- {}\n", component->name);
		}
	}

	Initialize();
	MainLoop();
	Finalize();

	return 0;
}

unsigned int VtGraphics::depthMapFBO()
{
	return m_renderPipeline->depthMapFBO;
}

void VtGraphics::ProcessMouse(GLFWwindow* m_window, double xpos, double ypos)
{
	for (auto foo : onMouseMove)
	{
		foo(xpos, ypos);
	}
}

void VtGraphics::ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset)
{
	for (auto foo : onMouseScroll)
	{
		foo(xoffset, yoffset);
	}
}

void VtGraphics::ProcessInput(GLFWwindow* m_window)
{
	if (Global::input->GetKey(GLFW_KEY_ESCAPE))
	{
		glfwSetWindowShouldClose(m_window, true);
	}
	if (Global::input->GetKeyDown(GLFW_KEY_L))
	{
		static bool renderLine = true;
		if (renderLine)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		}
		else
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
		renderLine = !renderLine;
	}
	if (Global::input->GetKeyDown(GLFW_KEY_SPACE))
	{
		m_pause = !m_pause;
		if (!m_pause)
		{
			lastUpdateTime = (float)glfwGetTime();
		}
	}

	if (Global::input->GetKeyDown(GLFW_KEY_ENTER))
	{
		if (glfwGetInputMode(m_window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL)
		{
			glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
		else
		{
			glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
}


void VtGraphics::Initialize()
{
	for (const auto& go : m_actors)
	{
		go->Start();
	}
}

void VtGraphics::MainLoop()
{
	// render loop
	while (!glfwWindowShouldClose(m_window))
	{
		// input
		ProcessInput(m_window);

		// rendering commands here
		if (!m_pause)
		{
			glClearColor(skyColor.x, skyColor.y, skyColor.z, skyColor.w);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// timing
			float current = (float)glfwGetTime();
			deltaTime = current - lastUpdateTime;
			lastUpdateTime = current;
			elapsedTime += deltaTime;

			for (const auto& go : m_actors)
			{
				go->Update();
			}
			for (const auto& callback : postUpdate)
			{
				callback();
			}

			m_renderPipeline->Render();

			// check and call events and swap the buffers
			glfwSwapBuffers(m_window);
		}

		glfwPollEvents();
	}
}

void VtGraphics::Finalize()
{
	if (Global::camera)
	{
		fmt::print("Final camera state[position{}, rotation{}],\n", 
			Global::camera->transform()->position, Global::camera->transform()->rotation);
	}
	for (const auto& go : m_actors)
	{
		go->OnDestroy();
	}
	glfwTerminate();
}

