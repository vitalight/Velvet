#pragma once

#include "Global.hpp"

#include <functional>

#include "VtGraphics.hpp"
#include "Component.hpp"
#include "Config.hpp"
#include "Light.hpp"
#include "Input.hpp"

namespace Velvet
{
	class PlayerController : public Component
	{
	public:
		PlayerController()
		{
			name = __func__;
		}

		void Start() override
		{
			Global::graphics->onMouseScroll.push_back(OnMouseScroll);

			Global::graphics->onMouseMove.push_back(OnMouseMove);
		}

		void Update() override
		{
			auto m_window = Global::graphics->m_window;

			const auto& camera = Global::camera;

			if (camera)
			{
				const auto& trans = camera->transform();
				const float speedScalar = Config::cameraTranslateSpeed; // adjust accordingly

				static glm::vec3 currentSpeed(0);
				glm::vec3 targetSpeed(0);

				if (Global::input->GetKey(GLFW_KEY_W))
					targetSpeed += camera->front();
				else if (Global::input->GetKey(GLFW_KEY_S))
					targetSpeed -= camera->front();

				if (Global::input->GetKey(GLFW_KEY_A))
					targetSpeed -= glm::normalize(glm::cross(camera->front(), camera->up()));
				else if (Global::input->GetKey(GLFW_KEY_D))
					targetSpeed += glm::normalize(glm::cross(camera->front(), camera->up()));

				if (Global::input->GetKey(GLFW_KEY_Q))
					targetSpeed += camera->up();
				else if (Global::input->GetKey(GLFW_KEY_E))
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

			auto m_window = Global::graphics->m_window;
			int state = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT);
			if (state == GLFW_PRESS)
			{
				shouldRotate = true;
			}
			else
			{
				shouldRotate = false;
				lastX = (float)xpos;
				lastY = (float)ypos;
			}

			if (shouldRotate)
			{
				auto rot = Global::camera->transform()->rotation;
				float yaw = -rot.y, pitch = rot.x;

				if (firstMouse)
				{
					lastX = (float)xpos;
					lastY = (float)ypos;
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