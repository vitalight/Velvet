#pragma once

namespace Velvet
{
	namespace Config
	{
		// Controls how fast the camera moves
		const float cameraTranslateSpeed = 5.0f;
		const float cameraRotateSensitivity = 0.15f;

		const unsigned int screenWidth = 1600;
		const unsigned int screenHeight = 900;
		const float screenAspect = float(screenWidth) / float(screenHeight);

		const unsigned int shadowWidth = 1024;
		const unsigned int shadowHeight = 1024;
	}
}