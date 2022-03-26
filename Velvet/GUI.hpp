#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define IMGUI_LEFT_LABEL(func, label, ...) (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

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
	};

	inline GUI* g_Gui;

	class GUI
	{
	public:
		static void RegisterDebug(function<void()> callback)
		{
			g_Gui->m_showDebugInfo.push_back(callback);
		}

		static void RegisterDebugOnce(function<void()> callback)
		{
			g_Gui->m_showDebugInfoOnce.push_back(callback);
		}

		static void RegisterDebugOnce(const string& debugMessage)
		{
			//vprintf(debugMessage, args);
			g_Gui->m_showDebugInfoOnce.push_back([debugMessage]() {
				ImGui::Text(debugMessage.c_str());
				});
		}

	public:
		void Initialize(GLFWwindow* window)
		{
			g_Gui = this;
			m_window = window;
			// Setup Dear ImGui context
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.IniFilename = NULL;
			io.Fonts->AddFontFromFileTTF("Assets/DroidSans.ttf", 18);
			//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
			//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

			// Setup Dear ImGui style
			ImGui::StyleColorsDark();

			auto style = &ImGui::GetStyle();
			style->SelectableTextAlign = ImVec2(0, 0.5);
			style->WindowPadding = ImVec2(10, 12);
			style->WindowRounding = 6;
			style->GrabRounding = 8;
			style->FrameRounding = 6;
			style->WindowTitleAlign = ImVec2(0.5, 0.5);

			style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 0.6f);
			style->Colors[ImGuiCol_TitleBg] = style->Colors[ImGuiCol_WindowBg];
			style->Colors[ImGuiCol_TitleBgActive] = style->Colors[ImGuiCol_TitleBg];
			style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.325, 0.325, 0.325, 1);
			style->Colors[ImGuiCol_FrameBg] = ImVec4(0.114, 0.114, 0.114, 1);
			style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.2, 0.2, 0.2, 1);
			style->Colors[ImGuiCol_Button] = ImVec4(0.46, 0.46, 0.46, 0.46);
			style->Colors[ImGuiCol_CheckMark] = ImVec4(0.851, 0.851, 0.851, 1);
			//ImGui::StyleColorsClassic();

			// Setup Platform/Renderer backends
			const char* glsl_version = "#version 330";
			ImGui_ImplGlfw_InitForOpenGL(m_window, true);
			ImGui_ImplOpenGL3_Init(glsl_version);

			m_deviceName = string((char*)glGetString(GL_RENDERER));
			m_deviceName = m_deviceName.substr(0, m_deviceName.find("/"));
		}

		void PreUpdate()
		{
			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
		}

		void OnUpdate()
		{
			//static bool show_demo_window = true;
			//ImGui::ShowDemoWindow(&show_demo_window);

			glfwGetWindowSize(m_window, &m_canvasWidth, &m_canvasHeight);

			ShowSceneWindow();
			ShowOptionWindow();
			ShowStatWindow();
		}

		void Render()
		{
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}

		void ShutDown()
		{
			ImGui_ImplOpenGL3_Shutdown();
			ImGui_ImplGlfw_Shutdown();
			ImGui::DestroyContext();
		}

	private:

		void ShowSceneWindow()
		{
			ImGui::SetNextWindowSize(ImVec2(k_windowWidth, (m_canvasHeight - 60) * 0.4));
			ImGui::SetNextWindowPos(ImVec2(20, 20));
			ImGui::Begin("Scene", NULL, k_windowFlags);

			static int selected = 0;
			for (int i = 0; i < 10; i++)
			{
				auto label = fmt::format("Scene {}", (char)('A' + i));
				if (ImGui::Selectable(label.c_str(), selected == i, 0, ImVec2(0, 28)))
				{
					selected = i;
				}
			}

			ImGui::End();
		}

		void ShowOptionWindow()
		{
			ImGui::SetNextWindowSize(ImVec2(k_windowWidth, (m_canvasHeight - 60) * 0.6));
			ImGui::SetNextWindowPos(ImVec2(20, 40 + (m_canvasHeight - 60) * 0.4));
			ImGui::Begin("Options", NULL, k_windowFlags);

			ImGui::PushItemWidth(-FLT_MIN);

			if (ImGui::Button("Reset", ImVec2(-FLT_MIN, 0)))
			{
				Global::graphics->Reset();
			}
			ImGui::Dummy(ImVec2(0.0f, 10.0f));

			if (ImGui::CollapsingHeader("Global", ImGuiTreeNodeFlags_DefaultOpen))
			{
				static bool radio = false;
				ImGui::Checkbox("Pause Physics", &Global::graphics->pause);
				ImGui::Checkbox("Wireframe", &Global::graphics->renderWireframe);
				ImGui::Checkbox("Draw Points", &radio);
				ImGui::Dummy(ImVec2(0.0f, 10.0f));
			}

			//if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
			//{
			//	IMGUI_LEFT_LABEL(ImGui::SliderFloat3, "LightPos", (float*)&(Global::light[0]->transform()->position), -10, 10, "%.2f");
			//	static float value = 0.0;
			//	IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Timestep", &value, 0, 1);
			//}

			ImGui::End();
		}

		void ShowStatWindow()
		{
			ImGui::SetNextWindowSize(ImVec2(k_windowWidth * 1.1, 0));
			ImGui::SetNextWindowPos(ImVec2(m_canvasWidth - k_windowWidth * 1.1 - 20, 20));
			ImGui::Begin("Statistics", NULL, k_windowFlags);

			static PerformanceStat stat;
			ComputeStatData(stat);

			ImGui::Text("Device:  %s", m_deviceName.c_str());
			ImGui::Text("Frame:  %d", stat.frameCount);
			ImGui::Text("Avg FrameRate:  %d FPS", stat.frameRate);

			ImGui::Dummy(ImVec2(0, 5));
			auto overlay = fmt::format("{:.2f} ms (Avg: {:.2f} ms)", stat.deltaTime, stat.graphAverage);
			ImGui::PlotLines("##", stat.graphValues, IM_ARRAYSIZE(stat.graphValues), stat.graphIndex, overlay.c_str(),
				0, stat.graphAverage * 2.0, ImVec2(k_windowWidth + 5, 80.0f));
			ImGui::Dummy(ImVec2(0, 5));

			if (m_showDebugInfo.size() + m_showDebugInfoOnce.size() > 0)
			{
				if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen))
				{
					//ImGui::Text("DebugInfo: (1,1,1)");
					for (auto callback : m_showDebugInfo)
					{
						callback();
					}
					for (auto callback : m_showDebugInfoOnce)
					{
						callback();
					}
					m_showDebugInfoOnce.clear();
				}
			}

			ImGui::End();
		}

		void ComputeStatData(PerformanceStat& stat)
		{
			static float timer1 = 0;
			static float timer2 = 0.0;
			const float timer1_interval = 0.2f;
			const float timer2_interval = 1.0f / 30.0f;

			const auto& graphics = Global::graphics;
			float frameCount = graphics->frameCount;
			float elapsedTime = graphics->elapsedTime;
			float deltaTimeMiliseconds = graphics->deltaTime * 1000;

			stat.frameCount = frameCount;
			timer1 += graphics->deltaTime;
			timer2 += graphics->deltaTime;

			if (timer2 > timer2_interval)
			{
				timer2 = 0;

				stat.graphValues[stat.graphIndex] = deltaTimeMiliseconds;
				stat.graphIndex = (stat.graphIndex + 1) % IM_ARRAYSIZE(stat.graphValues);
			}

			if (timer1 > timer1_interval)
			{
				timer1 = 0;

				stat.deltaTime = deltaTimeMiliseconds;
				stat.frameRate = (int)(frameCount / elapsedTime);

				for (int n = 0; n < IM_ARRAYSIZE(stat.graphValues); n++)
					stat.graphAverage += stat.graphValues[n];
				stat.graphAverage /= (float)IM_ARRAYSIZE(stat.graphValues);
			}
		}

		const ImGuiWindowFlags k_windowFlags = ImGuiWindowFlags_AlwaysAutoResize |
			ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
			ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
		const int k_windowWidth = 250;

		vector<function<void()>> m_showDebugInfo;
		vector<function<void()>> m_showDebugInfoOnce;

		GLFWwindow* m_window;
		int m_canvasWidth;
		int m_canvasHeight;
		string m_deviceName;
	};
}