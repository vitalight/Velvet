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

		namespace Sim
		{
			#define IMGUI_LEFT_LABEL(func, label, ...) (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

			inline float stiffness = 1.0f;
			inline glm::vec3 gravity = glm::vec3(0, -9.8, 0);
			inline float damping = 0.1f;

			inline int numSubsteps = 2;
			inline int numIterations = 4;

			inline void OnGUI()
			{
				IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Substeps", &numSubsteps, 1, 10);
				IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Iterations", &numIterations, 1, 20);
				IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Stiffness", &stiffness, 0, 50);
				IMGUI_LEFT_LABEL(ImGui::SliderFloat3, "Gravity", (float*)&gravity, -50, 50);
			}
		}
	}
}