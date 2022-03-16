#pragma once

#include "Global.h"

#include <functional>

#include "VtGraphics.h"
#include "Component.h"

namespace Velvet
{
	class PlayerController : public Component
	{
		void Start() override
		{
			Global::graphics->onMouseScroll.push_back(OnMouseScroll);

			Global::graphics->onMouseMove.push_back(OnMouseMove);
		}

		void Update() override
		{
			const auto& camera = Global::mainCamera;

			if (camera)
			{
				auto window = Global::graphics->window;
				const float cameraSpeed = 2.5f * Global::graphics->deltaTime; // adjust accordingly
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
				if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
					camera->position += cameraSpeed * camera->up;
				if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
					camera->position -= cameraSpeed * camera->up;
			}
			else
			{
				fmt::print("Error: Camera not found.\n");
			}
		}

		static void OnMouseScroll(double xoffset, double yoffset)
		{
			auto camera = Global::mainCamera;
			camera->zoom -= (float)yoffset;
			if (camera->zoom < 1.0f)
				camera->zoom = 1.0f;
			if (camera->zoom > 45.0f)
				camera->zoom = 45.0f;
		}

		static void OnMouseMove(double xpos, double ypos)
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
	};
}