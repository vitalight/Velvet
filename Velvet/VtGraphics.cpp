#include "VtGraphics.h"

#include "Camera.h"

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


void VtGraphics::ProcessMouse(GLFWwindow* window, double xpos, double ypos)
{
	static bool firstMouse = true;
	static float lastX = 400, lastY = 300;
	static float yaw = -90.0f, pitch = 0;
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;
	float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;
	yaw += xoffset;
	pitch += yoffset;
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;
	glm::vec3 direction;
	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	Global::mainCamera->front = glm::normalize(direction);
}


void VtGraphics::Initialize()
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
		return;
	}

	glfwMakeContextCurrent(m_window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		fmt::print("Failed to initialize GLAD\n");
		return;
	}

	glViewport(0, 0, 800, 600);
	glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int width, int height) {
		glViewport(0, 0, width, height);
		});
	glEnable(GL_DEPTH_TEST);

	glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(m_window, ProcessMouse);

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
	}

	const auto& camera = Global::mainCamera;

	if (camera)
	{
		const float cameraSpeed = 2.5f * deltaTime; // adjust accordingly
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera->position += cameraSpeed * camera->front;
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera->position -= cameraSpeed * camera->front;
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera->position -= glm::normalize(glm::cross(camera->front, camera->up)) *
			cameraSpeed;
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera->position += glm::normalize(glm::cross(camera->front, camera->up)) *
			cameraSpeed;
	}
	else
	{
		fmt::print("Error: Camera not found.\n");
	}
}

void VtGraphics::MainLoop()
{
	// render loop
	while (!glfwWindowShouldClose(m_window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// input
		ProcessInput(m_window);

		// rendering commands here
		if (!m_pause)
		{
			glClearColor(0.2f, 0.3f, 0.3f, 1.0f); 
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			for (const auto& go : m_objects)
			{
				go->Update();
			}

			// check and call events and swap the buffers
			glfwSwapBuffers(m_window);
		}

		glfwPollEvents();
	}
}

void VtGraphics::Finalize()
{
	for (const auto& go : m_objects)
	{
		go->OnDestroy();
	}
	glfwTerminate();
}

