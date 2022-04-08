#pragma once

#include "Global.hpp"

#include <functional>

#include "GameInstance.hpp"
#include "Component.hpp"
#include "Light.hpp"
#include "Input.hpp"
#include "GUI.hpp"
#include "Camera.hpp"

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
			Global::game->onMouseMove.Register(OnMouseMove);
			Global::game->godUpdate.Register(GodUpdate);
		}

		static void GodUpdate()
		{
			const auto& camera = Global::camera;

			if (camera)
			{
				const auto& trans = camera->transform();
				const float speedScalar = Global::Config::cameraTranslateSpeed; // adjust accordingly

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

				currentSpeed = Helper::Lerp(currentSpeed, targetSpeed, Timer::deltaTime() * 10);
				trans->position += currentSpeed * speedScalar * Timer::deltaTime();
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
			static float lastX = Global::Config::screenWidth / 2, lastY = Global::Config::screenHeight / 2;

			bool shouldRotate = Global::input->GetMouse(GLFW_MOUSE_BUTTON_RIGHT);

			if (shouldRotate)
			{
				auto rot = Global::camera->transform()->rotation;
				float yaw = -rot.y, pitch = rot.x;

				float xoffset = (float)xpos - lastX;
				float yoffset = lastY - (float)ypos;
				xoffset *= Global::Config::cameraRotateSensitivity;
				yoffset *= Global::Config::cameraRotateSensitivity;
				yaw += xoffset;
				pitch = clamp(pitch + yoffset, -89.0f, 89.0f);

				Global::camera->transform()->rotation = glm::vec3(pitch, -yaw, 0);
			}
			lastX = (float)xpos;
			lastY = (float)ypos;
		}
	};
}