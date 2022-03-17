#pragma once

namespace Velvet
{
	extern class VtGraphics;
	extern class Camera;
	extern class Light;
	extern class Input;

	namespace Global
	{
		inline VtGraphics* graphics;
		inline Camera* camera;
		inline Light* light;
		inline Input* input;
	}
}