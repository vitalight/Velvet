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
	class GUI;

	class VtGraphics
	{
	public:
		VtGraphics();

		~VtGraphics();

		VtGraphics(const VtGraphics&) = delete;

		shared_ptr<Actor> AddActor(shared_ptr<Actor> actor);

		shared_ptr<Actor> CreateActor(const string& name);

		void CreateScene(function<void(VtGraphics*)> scene);

		int Run();

		void Reset();

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

		glm::ivec2 windowSize();

		vector<function<void(double, double)>> onMouseScroll;
		vector<function<void(double, double)>> onMouseMove;
		vector<function<void()>> postUpdate;

		int frameCount = 0;
		float deltaTime = 0.0f;
		float elapsedTime = 0.0f;
		float lastUpdateTime = 0.0f;
		bool renderWireframe = false;
		bool pause = false;
		glm::vec4 skyColor = glm::vec4(0.0f);

	private:
		void ProcessMouse(GLFWwindow* m_window, double xpos, double ypos);
		
		void ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset);

		void ProcessKeyboard(GLFWwindow* m_window);

		void Initialize();

		void MainLoop();

		void Finalize();

	private:
		bool m_pendingReset = false;
		vector<shared_ptr<Actor>> m_actors;
		shared_ptr<Input> m_input;
		shared_ptr<RenderPipeline> m_renderPipeline;
		GLFWwindow* m_window = nullptr;
		shared_ptr<GUI> m_gui;
		function<void(VtGraphics*)> m_scene;
	};
}