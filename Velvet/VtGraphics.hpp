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

	class Light;
	class Input;
	class RenderPipeline;

	class VtGraphics
	{
	public:
		VtGraphics();

		shared_ptr<Actor> AddActor(shared_ptr<Actor> actor);

		shared_ptr<Actor> CreateActor(const string& name);

		int Run();

		template <typename T>
		enable_if_t<is_base_of<Component, T>::value, vector<T*>> FindComponents()
		{
			vector<T*> result;
			for (auto actor : m_actors)
			{
				auto component = actor->GetComponent<T>();
				if (component != nullptr)
				{
					result.push_back(component);
				}
			}
			return result;
		}

	public:
		unsigned int depthMapFBO();

		vector<function<void(double, double)>> onMouseScroll;
		vector<function<void(double, double)>> onMouseMove;
		vector<function<void()>> postUpdate;

		float deltaTime = 0.0f;
		float elapsedTime = 0.0f;
		float lastUpdateTime = 0.0f;
		glm::vec4 skyColor = glm::vec4(0.0f);

	private:
		void ProcessMouse(GLFWwindow* m_window, double xpos, double ypos);
		
		void ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset);

		void ProcessInput(GLFWwindow* m_window);

		void Initialize();

		void MainLoop();

		void Finalize();

	private:
		vector<shared_ptr<Actor>> m_actors;
		shared_ptr<Input> m_input;
		bool m_pause = false;
		shared_ptr<RenderPipeline> m_renderPipeline;
		GLFWwindow* m_window = nullptr;
	};
}