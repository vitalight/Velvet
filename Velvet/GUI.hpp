#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <functional>

#include "Timer.hpp"
#include "Common.hpp"

using namespace std;

#define IMGUI_LEFT_LABEL(func, label, ...) (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

namespace Velvet
{
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

		VtCallback<void()> m_showDebugInfo;
		VtCallback<void()> m_showDebugInfoOnce;

		GLFWwindow* m_window = nullptr;
		int m_canvasWidth = 0;
		int m_canvasHeight = 0;
		string m_deviceName;
	};
}