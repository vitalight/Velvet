#pragma once

#include <vector>

namespace Velvet
{
	class VtGraphics;
	class Camera;
	class Light;
	class Input;

	namespace Global
	{
		inline VtGraphics* graphics;
		inline Camera* camera;
		inline Input* input;

		inline std::vector<Light*> light;
	}
}