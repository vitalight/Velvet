#pragma once

#include "Global.hpp"

#include <functional>

#include "VtGraphics.hpp"
#include "Component.hpp"

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
			const auto& camera = Global::camera;

			if (camera)
			{
				const auto& trans = camera->transform();
				auto window = Global::graphics->window;
				const float speedScalar = 3.0f; // adjust accordingly

				static glm::vec3 currentSpeed(0);
				glm::vec3 targetSpeed(0);

				if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
					targetSpeed += camera->front();
				else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
					targetSpeed -= camera->front();

				if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
					targetSpeed -= glm::normalize(glm::cross(camera->front(), camera->up()));
				else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
					targetSpeed += glm::normalize(glm::cross(camera->front(), camera->up()));

				if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
					targetSpeed += camera->up();
				else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
					targetSpeed -= camera->up();

				currentSpeed = Helper::Lerp(currentSpeed, targetSpeed, Global::graphics->deltaTime * 10);
				trans->position += currentSpeed * speedScalar * Global::graphics->deltaTime;
			}
			else
			{
				fmt::print("Error: Camera not found.\n");
			}
		}

		static void OnMouseScroll(double xoffset, double yoffset)
		{
			auto camera = Global::camera;
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
			static bool shouldRotate = false;

			auto window = Global::graphics->window;
			int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
			if (state == GLFW_PRESS)
			{
				shouldRotate = true;
			}
			else
			{
				shouldRotate = false;
				lastX = xpos;
				lastY = ypos;
			}

			if (shouldRotate)
			{
				auto rot = Global::camera->transform()->rotation;
				float yaw = -rot.y, pitch = rot.x;

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
				float sensitivity = 0.15f;
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

				Global::camera->transform()->rotation = glm::vec3(pitch, -yaw, 0);
				//fmt::print("CameraRotation: {}\n", Global::mainCamera->transform()->rotation);
				//Global::mainCamera->front = glm::normalize(direction);
			}
		}
	};
}