#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Global.hpp"
#include "GameInstance.hpp"

namespace Velvet
{
	class Input
	{
	public:
		Input(GLFWwindow* window);

		// Returns true while the user holds down the key.
		bool GetKey(int key);

		// Returns true during the frame the user starts pressing down the key.
		bool GetKeyDown(int key);

		// Returns true during the frame the user releases the key.
		bool GetKeyUp(int key);

		bool GetMouse(int button);

		bool GetMouseDown();

	private:
		GLFWwindow* m_window; 
		char m_keyOnce[GLFW_KEY_LAST + 1];
	};
}