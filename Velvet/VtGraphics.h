#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include "Actor.h"

namespace Velvet
{
	using namespace std;

	class VtGraphics
	{
	public:
		VtGraphics();

		void AddActor(shared_ptr<Actor> actor);

		void Initialize();

		int Run();

		vector<function<void(double, double)>> onMouseScroll;

		vector<function<void(double, double)>> onMouseMove;

		GLFWwindow* window = nullptr;
		float deltaTime = 0.0f;
		float lastFrame = 0.0f;

	private:
		void ProcessMouse(GLFWwindow* window, double xpos, double ypos);
		
		void ProcessScroll(GLFWwindow* window, double xoffset, double yoffset);

		void ProcessInput(GLFWwindow* window);

		void MainLoop();

		void Finalize();

		vector<shared_ptr<Actor>> m_objects;
		bool m_pause = false;

	};
}