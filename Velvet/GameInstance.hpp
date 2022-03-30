#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Component.hpp"

namespace Velvet
{
	using namespace std;

	class Light;
	class Input;
	class RenderPipeline;
	class GUI;
	class Scene;
	class Actor;

	class GameInstance
	{
	public:
		GameInstance(GLFWwindow* window, shared_ptr<GUI> gui);
		GameInstance(const GameInstance&) = delete;

		shared_ptr<Actor> AddActor(shared_ptr<Actor> actor);
		shared_ptr<Actor> CreateActor(const string& name);

		int Run();

		void ProcessMouse(GLFWwindow* m_window, double xpos, double ypos);
		void ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset);
		void ProcessKeyboard(GLFWwindow* m_window);

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
		unsigned int depthFrameBuffer();
		glm::ivec2 windowSize();

		vector<function<void(double, double)>> onMouseScroll;
		vector<function<void(double, double)>> onMouseMove;
		vector<function<void()>> postUpdate;
		vector<function<void()>> godUpdate; // update when main logic is paused (for debugging purpose)
		vector<function<void()>> onFinalize;

		int frameCount = 0;
		float elapsedTime = 0.0f;
		float lastUpdateTime = (float)glfwGetTime();
		float deltaTime = 0.0f;
		const float fixedDeltaTime = 1.0f / 60.0f;

		bool pause = false;
		bool step = false;
		bool renderWireframe = false;
		bool pendingReset = false;
		glm::vec4 skyColor = glm::vec4(0.0f);

	private:
		void Initialize();
		void MainLoop();
		void Finalize();

	private:
		GLFWwindow* m_window = nullptr;
		shared_ptr<GUI> m_gui;

		vector<shared_ptr<Actor>> m_actors;
		shared_ptr<RenderPipeline> m_renderPipeline;
	};
}