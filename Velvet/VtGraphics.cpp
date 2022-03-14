#include "VtGraphics.h"

using namespace Velvet;

char keyOnce[GLFW_KEY_LAST + 1];
#define glfwGetKeyOnce(WINDOW, KEY)				\
	(glfwGetKey(WINDOW, KEY) ?				\
	 (keyOnce[KEY] ? false : (keyOnce[KEY] = true)) :	\
	 (keyOnce[KEY] = false))

void VtGraphics::AddActor(shared_ptr<Actor> gameObject)
{
	m_objects.push_back(gameObject);
}

int VtGraphics::Initialize()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_window = glfwCreateWindow(800, 600, "Velvet", NULL, NULL);

	if (m_window == NULL)
	{
		fmt::print("Failed to create GLFW window\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(m_window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		fmt::print("Failed to initialize GLAD\n");
		return -1;
	}

	glViewport(0, 0, 800, 600);
	glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int width, int height) {
		glViewport(0, 0, width, height);
		});

	for (const auto& go : m_objects)
	{
		go->Start();
	}
}

int VtGraphics::Run()
{
	fmt::print("Hello Velvet!\n");

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
	// TODO: ignore continuous calls
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
}

int VtGraphics::MainLoop()
{
	// render loop
	while (!glfwWindowShouldClose(m_window))
	{
		// input
		ProcessInput(m_window);

		// rendering commands here
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		for (const auto& go : m_objects)
		{
			go->Update();
		}

		// check and call events and swap the buffers
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}
	return 0;
}

int VtGraphics::Finalize()
{
	for (const auto& go : m_objects)
	{
		go->OnDestroy();
	}
	glfwTerminate();
	return 0;
}

