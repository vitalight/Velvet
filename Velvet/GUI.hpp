#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <functional>

#include "Timer.hpp"

using namespace std;

#define IMGUI_LEFT_LABEL(func, label, ...) (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

namespace Velvet
{
	struct PerformanceStat
	{
		float deltaTime = 0;
		int frameRate = 0;
		int frameCount = 0;
		int physicsFrameCount = 0;

		float graphValues[100] = {};
		int graphIndex = 0;
		float graphAverage = 0.0f;

		double cpuTime = 0;
		double gpuTime = 0;

		void Update()
		{
			if (Global::gameState.pause)
			{
				return;
			}
			static float timer1 = 0;
			static float timer2 = 0.0;
			const float timer1_interval = 0.2f;
			const float timer2_interval = 1.0f / 30.0f;

			const auto& game = Global::game;
			float elapsedTime = Timer::elapsedTime();
			float deltaTimeMiliseconds = Timer::deltaTime() * 1000;

			frameCount = Timer::frameCount();
			physicsFrameCount = Timer::physicsFrameCount();

			// Some variables should not be update each frame
			timer1 += Timer::deltaTime();
			timer2 += Timer::deltaTime();

			if (timer2 > timer2_interval)
			{
				timer2 = 0;

				graphValues[graphIndex] = deltaTimeMiliseconds;
				graphIndex = (graphIndex + 1) % IM_ARRAYSIZE(graphValues);
			}

			if (timer1 > timer1_interval)
			{
				timer1 = 0;

				deltaTime = deltaTimeMiliseconds;
				frameRate = elapsedTime > 0 ? (int)(frameCount / elapsedTime) : 0;
				cpuTime = Timer::GetTimer("CPU_TIME") * 1000;
				gpuTime = Timer::GetTimer("GPU_TIME") * 1000;

				for (int n = 0; n < IM_ARRAYSIZE(graphValues); n++)
					graphAverage += graphValues[n];
				graphAverage /= (float)IM_ARRAYSIZE(graphValues);
			}
		}
	};

	class VtApp;

	class GUI
	{
	public:
		static void RegisterDebug(function<void()> callback);

		static void RegisterDebugOnce(function<void()> callback);

		static void RegisterDebugOnce(const string& debugMessage);

	public:
		GUI(GLFWwindow* window);

		void OnUpdate();

		void Render();

		void ShutDown();

		void ClearCallback();

	private:

		void CustomizeStyle();

		void ShowSceneWindow();

		void ShowOptionWindow();

		void ShowStatWindow();

		const ImGuiWindowFlags k_windowFlags = ImGuiWindowFlags_AlwaysAutoResize |
			ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
			ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
		const float k_windowWidth = 250.0f;

		vector<function<void()>> m_showDebugInfo;
		vector<function<void()>> m_showDebugInfoOnce;

		GLFWwindow* m_window = nullptr;
		int m_canvasWidth = 0;
		int m_canvasHeight = 0;
		string m_deviceName;
	};
}