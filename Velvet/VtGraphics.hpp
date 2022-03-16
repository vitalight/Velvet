#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include "Actor.hpp"

namespace Velvet
{
	using namespace std;

	extern class Light;

	class VtGraphics
	{
	public:
		VtGraphics();

		shared_ptr<Actor> AddActor(shared_ptr<Actor> actor);

		void Initialize();

		int Run();

		vector<function<void(double, double)>> onMouseScroll;
		vector<function<void(double, double)>> onMouseMove;
		vector<function<void()>> postUpdate;

		GLFWwindow* window = nullptr;
		float deltaTime = 0.0f;
		float elapsedTime = 0.0f;
		float lastUpdateTime = 0.0f;
		glm::vec4 skyColor = glm::vec4(0.0f);

	private:
		void ProcessMouse(GLFWwindow* window, double xpos, double ypos);
		
		void ProcessScroll(GLFWwindow* window, double xoffset, double yoffset);

		void ProcessInput(GLFWwindow* window);

		void MainLoop();

		void Finalize();

		vector<shared_ptr<Actor>> m_actors;
		bool m_pause = false;
	};
}