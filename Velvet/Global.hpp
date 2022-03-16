#pragma once

namespace Velvet
{
	extern class VtGraphics;
	extern class Camera;
	extern class Light;

	namespace Global
	{
		inline VtGraphics* graphics;
		inline Camera* mainCamera;
		inline Light* light;
	}
}