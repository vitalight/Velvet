#pragma once

namespace Velvet
{
	namespace Config
	{
		// Controls how fast the camera moves
		const float cameraTranslateSpeed = 5.0f;

		const unsigned int screenWidth = 1280;
		const unsigned int screenHeight = 720;
		const float screenAspect = float(screenWidth) / float(screenHeight);

		const unsigned int shadowWidth = 1024;
		const unsigned int shadowHeight = 1024;
	}
}