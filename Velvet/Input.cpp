#include "Input.hpp"

using namespace Velvet;

Input::Input(GLFWwindow* window)
{
	m_window = window;
	Global::input = this;
	memset(m_keyOnce, 0, sizeof(m_keyOnce));
	memset(m_keyNow, 0, sizeof(m_keyNow));
}

void Input::OnUpdate()
{
	swap(m_keyOnce, m_keyNow);
}

// Returns true while the user holds down the key.

bool Input::GetKey(int key)
{
	return (glfwGetKey(m_window, key) == GLFW_PRESS);
}

// Returns true during the frame the user starts pressing down the key.
bool Input::GetKeyDown(int key)
{
	m_keyNow[key] = GetKey(key);
	return !m_keyOnce[key] && m_keyNow[key];
}

void Input::ToggleOnKeyDown(int key, bool& variable)
{
	if (GetKeyDown(key))
	{
		variable = !variable;
	}
}

// Returns true during the frame the user releases the key.
bool Input::GetKeyUp(int key)
{
	m_keyNow[key] = GetKey(key);
	return m_keyOnce[key] && !m_keyNow[key];
}

bool Input::GetMouse(int button)
{
	int state = glfwGetMouseButton(m_window, button);
	return state == GLFW_PRESS;
}

bool Input::GetMouseDown(int button)
{
	m_keyNow[button] = GetMouse(button);
	return !m_keyOnce[button] && m_keyNow[button];
}

bool Input::GetMouseUp(int button)
{
	m_keyNow[button] = GetMouse(button);
	return m_keyOnce[button] && !m_keyNow[button];
}
