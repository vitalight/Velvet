#include "Input.hpp"

using namespace Velvet;

Input::Input(GLFWwindow* window)
{
	m_window = window;
	Global::input = this;
	memset(m_keyOnce, 0, sizeof(m_keyOnce));
}

// Returns true while the user holds down the key.

bool Input::GetKey(int key)
{
	return (glfwGetKey(m_window, key) == GLFW_PRESS);
}

// Returns true during the frame the user starts pressing down the key.
bool Input::GetKeyDown(int key)
{
	//return GetKey(key);
	return	(GetKey(key) ?
		(m_keyOnce[key] ? false : (m_keyOnce[key] = true)) :
		(m_keyOnce[key] = false));
}

// Returns true during the frame the user releases the key.

bool Input::GetKeyUp(int key)
{
	return true;
}

bool Input::GetMouseDown()
{
	return true;
}
