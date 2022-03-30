#pragma once

#include <vector>

#include <glm/glm.hpp>
#include <imgui.h>


namespace Velvet
{
	class GameInstance;
	class Camera;
	class Light;
	class Input;
	class VtEngine;

	namespace Global
	{
		inline VtEngine* engine;
		inline GameInstance* game;
		inline Camera* camera;
		inline Input* input;

		inline std::vector<Light*> lights;

		namespace Config
		{
			// Controls how fast the camera moves
			const float cameraTranslateSpeed = 5.0f;
			const float cameraRotateSensitivity = 0.15f;

			const unsigned int screenWidth = 1600;
			const unsigned int screenHeight = 900;

			const unsigned int shadowWidth = 1024;
			const unsigned int shadowHeight = 1024;
		}

		namespace Sim
		{
			#define IMGUI_LEFT_LABEL(func, label, ...) (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

			inline int numSubsteps = 3;
			inline int numIterations = 3;

			inline float compliance = 0;
			inline glm::vec3 gravity = glm::vec3(0, -9.8, 0);
			inline float damping = 0.1f;
			inline float collisionMargin = 0.1;

			inline void OnGUI()
			{
				IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Substeps", &numSubsteps, 1, 20);
				IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Iterations", &numIterations, 1, 20);
				ImGui::Separator();
				IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Compliance", &compliance, 0, 0.1, "%.3f", ImGuiSliderFlags_Logarithmic);
				IMGUI_LEFT_LABEL(ImGui::SliderFloat3, "Gravity", (float*)&gravity, -50, 50);
				IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Collision Margin", &collisionMargin, 0, 1);
			}
		}
	}
}