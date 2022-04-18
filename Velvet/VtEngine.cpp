#include "VtEngine.hpp"

#include <algorithm>
#include <fmt/core.h>
#include <fmt/color.h>

#include "Scene.hpp"
#include "GUI.hpp"
#include "GameInstance.hpp"
#include "Input.hpp"

using namespace Velvet;

void PrintGlfwError(int error, const char* description)
{
	fmt::print("Error(Glfw): Code({}), {}\n", error, description);
}

VtEngine::VtEngine()
{
	Global::engine = this;
	// setup glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// Multi-sample Anti-aliasing
	glfwWindowHint(GLFW_SAMPLES, 4);

	m_window = glfwCreateWindow(Global::Config::screenWidth, Global::Config::screenHeight, "Velvet", NULL, NULL);

	if (m_window == NULL)
	{
		fmt::print("Failed to create GLFW window\n");
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(m_window); 
	glfwSwapInterval(0);

	glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* m_window, int width, int height) {
		glViewport(0, 0, width, height);
		});

	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(m_window, [](GLFWwindow* m_window, double xpos, double ypos) {
		Global::game->ProcessMouse(m_window, xpos, ypos);
		});
	glfwSetScrollCallback(m_window, [](GLFWwindow* m_window, double xoffset, double yoffset) {
		Global::game->ProcessScroll(m_window, xoffset, yoffset);
		});
	glfwSetErrorCallback(PrintGlfwError);

	// setup opengl
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		fmt::print("Failed to initialize GLAD\n");
		return;
	}
	glViewport(0, 0, Global::Config::screenWidth, Global::Config::screenHeight);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// setup stbi
	stbi_set_flip_vertically_on_load(true);

	// setup members
	m_gui = make_shared<GUI>(m_window);
	m_input = make_shared<Input>(m_window);
}

VtEngine::~VtEngine()
{
	m_gui->ShutDown();
	glfwTerminate();
}

void VtEngine::SetScenes(const vector<shared_ptr<Scene>>& initializers)
{
	scenes = initializers;
}

int VtEngine::Run()
{
	do 
	{
#pragma warning( push )
#pragma warning( disable : 4129)
		fmt::print(
			"©°{0:\-^{2}}©´\n"
			"©¦{1: ^{2}}©¦\n"
			"©¸{0:\-^{2}}©¼\n", "", "Hello, Velvet!", 30);
#pragma warning( pop ) 

		m_game = make_shared<GameInstance>(m_window, m_gui);
		sceneIndex = m_nextSceneIndex;
		scenes[sceneIndex]->PopulateActors(m_game.get());
		scenes[sceneIndex]->onEnter.Invoke();
		m_game->Run();
		scenes[sceneIndex]->onExit.Invoke();
		scenes[sceneIndex]->ClearCallbacks();

		Resource::ClearCache();
		m_gui->ClearCallback();
	} while (m_game->pendingReset);

	return 0;
}

void VtEngine::Reset()
{
	m_game->pendingReset = true;
}

void VtEngine::SwitchScene(unsigned int _sceneIndex)
{
	m_nextSceneIndex = std::clamp(_sceneIndex, 0u, (unsigned int)scenes.size()-1);
	m_game->pendingReset = true;
}

glm::ivec2 VtEngine::windowSize()
{
	glm::ivec2 result;
	glfwGetWindowSize(m_window, &result.x, &result.y);
	return result;
}
