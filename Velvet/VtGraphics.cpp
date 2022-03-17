#include "VtGraphics.hpp"

#include <functional>

#include "External/stb_image.h"

#include "VtHelper.hpp"
#include "Camera.hpp"

using namespace Velvet;

char keyOnce[GLFW_KEY_LAST + 1];
#define glfwGetKeyOnce(WINDOW, KEY)				\
	(glfwGetKey(WINDOW, KEY) ?				\
	 (keyOnce[KEY] ? false : (keyOnce[KEY] = true)) :	\
	 (keyOnce[KEY] = false))

VtGraphics::VtGraphics()
{
	Global::graphics = this;

	// setup glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(800, 600, "Velvet", NULL, NULL);

	if (window == NULL)
	{
		fmt::print("Failed to create GLFW window\n");
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

	glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
		glViewport(0, 0, width, height);
		});

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, [](GLFWwindow* window, double xpos, double ypos) {
		Global::graphics->ProcessMouse(window, xpos, ypos);
		});
	glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
		Global::graphics->ProcessScroll(window, xoffset, yoffset);
		});

	// setup opengl
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		fmt::print("Failed to initialize GLAD\n");
		return;
	}
	glViewport(0, 0, 800, 600);
	glEnable(GL_DEPTH_TEST);

	// setup stbi
	stbi_set_flip_vertically_on_load(true);
}

shared_ptr<Actor> VtGraphics::AddActor(shared_ptr<Actor> gameObject)
{
	m_actors.push_back(gameObject);
	return gameObject;
}

void VtGraphics::ProcessMouse(GLFWwindow* window, double xpos, double ypos)
{
	for (auto foo : onMouseMove)
	{
		foo(xpos, ypos);
	}
}

void VtGraphics::ProcessScroll(GLFWwindow* window, double xoffset, double yoffset)
{
	for (auto foo : onMouseScroll)
	{
		foo(xoffset, yoffset);
	}
}

void VtGraphics::Initialize()
{
	for (const auto& go : m_actors)
	{
		go->Start();
	}
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

void VtGraphics::ProcessInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}
	if (glfwGetKeyOnce(window, GLFW_KEY_L) == GLFW_PRESS)
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
	if (glfwGetKeyOnce(window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		m_pause = !m_pause;
		if (!m_pause)
		{
			lastUpdateTime = glfwGetTime();
		}
	}

	if (glfwGetKeyOnce(window, GLFW_KEY_ENTER) == GLFW_PRESS)
	{
		static bool showCursor = true;
		if (showCursor)
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		else
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
		showCursor = !showCursor;
	}
}

void VtGraphics::MainLoop()
{
	// render loop
	while (!glfwWindowShouldClose(window))
	{
		// input
		ProcessInput(window);

		// rendering commands here
		if (!m_pause)
		{
			glClearColor(skyColor.x, skyColor.y, skyColor.z, skyColor.w);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// timing
			float current = glfwGetTime();
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

			// check and call events and swap the buffers
			glfwSwapBuffers(window);
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

