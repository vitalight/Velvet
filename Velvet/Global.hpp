#pragma once

#include <vector>
#include <glm/glm.hpp>

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
			inline float stiffness = 15.0f;
			inline glm::vec3 gravity = glm::vec3(0, -1, 0);
		}
	}
}