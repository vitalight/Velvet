#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

using namespace std;

namespace Velvet
{
	class Scene;
	class GUI;
	class GameInstance;
	class Input;

	class VtEngine
	{
	public:
		VtEngine();
		~VtEngine();

		int Run();

		void Reset();
		void SwitchScene(unsigned int sceneIndex);
		void SetScenes(const vector<shared_ptr<Scene>>& scenes);

		glm::ivec2 windowSize();

		vector<shared_ptr<Scene>> scenes;
		unsigned int sceneIndex = 0;
	private:
		unsigned int m_nextSceneIndex = 0;
		GLFWwindow* m_window = nullptr;
		shared_ptr<GUI> m_gui;
		shared_ptr<GameInstance> m_game;
		shared_ptr<Input> m_input;
	};
}