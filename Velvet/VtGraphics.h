#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include "VtActor.h"

namespace Velvet
{
	using namespace std;

	class VtGraphics
	{
	public:
		void AddActor(shared_ptr<VtActor> actor);

		int Initialize();

		int Run();

	private:
		void ProcessInput(GLFWwindow* window);

		int MainLoop();

		int Finalize();

	private:
		GLFWwindow* m_window;
		vector<shared_ptr<VtActor>> m_objects;
	};
}