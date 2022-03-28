#include "VtGraphics.hpp"

#include <functional>

#include "External/stb_image.h"
#include "fmt/color.h"

#include "Helper.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "RenderPipeline.hpp"
#include "Config.hpp"
#include "GUI.hpp"

using namespace Velvet;

static void PrintGlfwError(int error, const char* description)
{
	fmt::print("Error(Glfw): Code({}), {}\n", error, description);
}

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
	glfwSetErrorCallback(PrintGlfwError);

	// setup opengl
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		fmt::print("Failed to initialize GLAD\n");
		return;
	}
	glViewport(0, 0, Config::screenWidth, Config::screenHeight);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// setup stbi
	stbi_set_flip_vertically_on_load(true);

	// setup members
	m_input = shared_ptr<Input>(new Input(m_window));
	m_renderPipeline = shared_ptr<RenderPipeline>(new RenderPipeline());
	m_gui = shared_ptr<GUI>(new GUI());
	m_gui->Initialize(m_window);
}

Velvet::VtGraphics::~VtGraphics()
{
	m_gui->ShutDown();
	m_gui = nullptr;
	m_input = nullptr;
	m_renderPipeline = nullptr;

	glfwTerminate();
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

void VtGraphics::SetSceneInitializers(const vector<SceneInitializer>& initializers)
{
	sceneInitializers = initializers;
}

int VtGraphics::Run()
{
	fmt::print(fg(fmt::color::green),
		"©°{0:\-^{2}}©´\n"
		"©¦{1: ^{2}}©¦\n"
		"©¸{0:\-^{2}}©¼\n", "", "Hello, Velvet!", 30);
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

	do {
		if (m_pendingReset)
		{
			frameCount = 0;
			deltaTime = 0.0f;
			elapsedTime = 0.0f;
			lastUpdateTime = glfwGetTime();

			m_pendingReset = false;
		}
		// TODO: rename
		sceneInitializers[m_sceneIndex](this);

		Initialize();
		MainLoop();
		Finalize();
	} while(m_pendingReset);

	return 0;
}

void VtGraphics::Reset()
{
	m_pendingReset = true;
}

void VtGraphics::SwitchScene(unsigned int sceneIndex)
{
	m_sceneIndex = sceneIndex;
	m_pendingReset = true;
}

unsigned int VtGraphics::depthMapFBO()
{
	return m_renderPipeline->depthMapFBO;
}

glm::ivec2 Velvet::VtGraphics::windowSize()
{
	glm::ivec2 result;
	glfwGetWindowSize(m_window, &result.x, &result.y);
	return result;
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

void VtGraphics::ProcessKeyboard(GLFWwindow* m_window)
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
	fmt::print("Info(VtGraphics): Initialization success. Enter main loop.\n");
	// render loop
	while (!glfwWindowShouldClose(m_window) && !m_pendingReset)
	{
		// input
		ProcessKeyboard(m_window);

		// rendering commands here
		glClearColor(skyColor.x, skyColor.y, skyColor.z, skyColor.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPolygonMode(GL_FRONT_AND_BACK, renderWireframe ? GL_LINE : GL_FILL);

		// timing
		float current = (float)glfwGetTime();
		deltaTime = current - lastUpdateTime;
		lastUpdateTime = current;

		m_gui->PreUpdate();
		m_gui->OnUpdate();
		
		if (!pause)
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

		m_renderPipeline->Render();
		m_gui->Render();

		// check and call events and swap the buffers
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}
}

void VtGraphics::Finalize()
{
	for (const auto& go : m_actors)
	{
		go->OnDestroy();
	}

	Global::light.clear();
	
	m_actors.clear();
	onMouseScroll.clear();
	onMouseMove.clear();
	postUpdate.clear();
}

