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
		inline std::vector<Light*> light;
		inline Input* input;
	}
}