#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define IMGUI_LEFT_LABEL(func, label, ...) (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

namespace Velvet
{
	class GUI
	{
	public:
		void Initialize(GLFWwindow* window)
		{
			m_window = window;
			// Setup Dear ImGui context
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.IniFilename = NULL;
			io.Fonts->AddFontFromFileTTF("Assets/DroidSans.ttf", 19);
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
		}

		void PreUpdate()
		{
			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
		}

		void OnUpdate1()
		{
			static bool show_demo_window = true;
			static bool show_another_window = true;
			// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
			if (show_demo_window)
				ImGui::ShowDemoWindow(&show_demo_window);

			// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
			{
				static float f = 0.0f;
				static int counter = 0;

				ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

				ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
				ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
				ImGui::Checkbox("Another Window", &show_another_window);

				ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f

				if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
					counter++;
				ImGui::SameLine();
				ImGui::Text("counter = %d", counter);

				ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
				ImGui::End();
			}

			// 3. Show another simple window.
			if (show_another_window)
			{
				ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
				ImGui::Text("Hello from another window!");
				if (ImGui::Button("Close Me"))
					show_another_window = false;
				ImGui::End();
			}
		}

		void OnUpdate()
		{
			//static bool show_demo_window = true;
			//ImGui::ShowDemoWindow(&show_demo_window);

			ImGuiWindowFlags window_flags =  ImGuiWindowFlags_AlwaysAutoResize |
				ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
				ImGuiWindowFlags_NoResize  | ImGuiWindowFlags_NoCollapse
				;
			static bool displayGUI = true;

			int canvasWidth;
			int canvasHeight;
			glfwGetWindowSize(m_window, &canvasWidth, &canvasHeight);
			int windowWidth = 250;

			{
				ImGui::SetNextWindowSize(ImVec2(windowWidth, (canvasHeight - 60) * 0.4));
				ImGui::SetNextWindowPos(ImVec2(20, 20));
				ImGui::Begin("Scene", NULL, window_flags);
				//ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 151, 61, 255));
				//ImGui::Indent(10);

				//ImGui::SetWindowFontScale(2.0f);
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

			{
				ImGui::SetNextWindowSize(ImVec2(windowWidth, (canvasHeight - 60) * 0.6));
				ImGui::SetNextWindowPos(ImVec2(20, 40 + (canvasHeight - 60) * 0.4));
				ImGui::Begin("Options", NULL, window_flags);
				ImGui::PushItemWidth(-FLT_MIN);

				ImGui::Button("Reset", ImVec2(-FLT_MIN, 0));
				ImGui::Dummy(ImVec2(0.0f, 10.0f));
				if (ImGui::CollapsingHeader("Global", ImGuiTreeNodeFlags_DefaultOpen))
				{
					static bool radio = false;
					ImGui::Checkbox("Pause", &Global::graphics->pause);
					ImGui::Checkbox("Wireframe", &Global::graphics->renderWireframe);
					ImGui::Checkbox("Draw Points", &radio);
					ImGui::Dummy(ImVec2(0.0f, 10.0f));
				}

				if (ImGui::CollapsingHeader("Sim", ImGuiTreeNodeFlags_DefaultOpen))
				{
					//ImGui::SliderFloat3("Lightpos", (float*)&(Global::light[0]->transform()->position), -10, 10, "%.2f");
					IMGUI_LEFT_LABEL(ImGui::SliderFloat3, "Lightpos", (float*)&(Global::light[0]->transform()->position), -10, 10, "%.2f");
					static float value = 0.0;
					//ImGui::SliderFloat("Timestep", &value, 0, 1);
					IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Timestep", &value, 0, 1);
				}

				ImGui::End();
			}

			{
				static float nextUpdateTime = Global::graphics->elapsedTime;
				static float deltaTime;
				static int frameRate;
				if (Global::graphics->elapsedTime > nextUpdateTime)
				{
					nextUpdateTime = Global::graphics->elapsedTime + 0.2;
					deltaTime = Global::graphics->deltaTime * 1000;
					frameRate = (int)(Global::graphics->frameCount / Global::graphics->elapsedTime);
				}
				ImGui::SetNextWindowSize(ImVec2(windowWidth * 1.1, 0));
				ImGui::SetNextWindowPos(ImVec2(canvasWidth - windowWidth * 1.1 -20, 20));
				ImGui::Begin("Statistics", NULL, window_flags);
				//ImGui::Indent(10);
				ImGui::Text("Frame:  %d", Global::graphics->frameCount);
				ImGui::Text("Avg FrameRate:  %d FPS", frameRate);
				{
					static float values[100] = {};
					static int values_offset = 0;
					static double refresh_time = 0.0;
					if (refresh_time == 0.0)
						refresh_time = ImGui::GetTime();
					while (refresh_time < ImGui::GetTime()) // Create data at fixed 60 Hz rate for the demo
					{
						values[values_offset] = Global::graphics->deltaTime * 1000;
						values_offset = (values_offset + 1) % IM_ARRAYSIZE(values);
						refresh_time += 1.0f / 30.0f;
					}
					float average = 0.0f;
					for (int n = 0; n < IM_ARRAYSIZE(values); n++)
						average += values[n];
					average /= (float)IM_ARRAYSIZE(values);
					auto overlay = fmt::format("{:.2f} ms (Avg: {:.2f} ms)", deltaTime, average);
					ImGui::PlotLines("##", values, IM_ARRAYSIZE(values), values_offset, overlay.c_str(), 
						0, average * 2.0, ImVec2(windowWidth+5, 80.0f));
				}

				ImGui::Separator();
				string deviceName((char*)glGetString(GL_RENDERER));
				deviceName = deviceName.substr(0, deviceName.find("/"));
				ImGui::Text("Device:  %s", deviceName.c_str());
				ImGui::End();
			}
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
		GLFWwindow* m_window;
	};
}