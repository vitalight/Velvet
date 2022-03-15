#pragma once

#include <iostream>
#include <vector>
#include <string>

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
		void AddActor(shared_ptr<Actor> actor);

		void Initialize();

		int Run();

	private:
		static void ProcessMouse(GLFWwindow* window, double xpos, double ypos);

		void ProcessInput(GLFWwindow* window);

		void MainLoop();

		void Finalize();

	private:
		GLFWwindow* m_window;
		vector<shared_ptr<Actor>> m_objects;
		bool m_pause = false;

		float deltaTime = 0.0f;
		float lastFrame = 0.0f;
	};
}