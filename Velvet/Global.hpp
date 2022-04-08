#pragma once

#include <vector>

#include "Common.hpp"

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

		inline VtGameState gameState;
		inline VtSimParams simParams;

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
	}
}