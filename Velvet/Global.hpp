#pragma once

#include <vector>

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
	}
}