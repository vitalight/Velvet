#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <functional>

using namespace std;

namespace Velvet
{
	struct PerformanceStat
	{
		float deltaTime = 0;
		int frameRate = 0;
		int frameCount = 0;

		float graphValues[100] = {};
		int graphIndex = 0;
		float graphAverage = 0.0f;

		double cpuTime = 0;
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

	private:

		void CustomizeStyle();

		void ShowSceneWindow();

		void ShowOptionWindow();

		void ShowStatWindow();

		void ComputeStatData(PerformanceStat& stat);

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