#include "GUI.hpp"

#include "Scene.hpp"
#include "VtEngine.hpp"

using namespace Velvet;

#define SHORTCUT_BOOL(key, variable) if (Global::input->GetKeyDown(key)) variable = !variable

inline GUI* g_Gui;
const float k_leftWindowWidth = 250.0f;
const float k_rightWindowWidth = 330.0f;

struct SolverTiming
{
	int count = 0;

	vector<string> labels = {
		"SetParams",
		//"Initialize",
		"Predict",
		"SolveStretch",
		"SolveAttach",
		"CollideSDFs",
		"CollideParticles",
		"Finalize",
		"UpdateNormals",

		"HashParticle",
		"HashSort",
		"HashBuildCell",
		"HashCache",

		"Total",
	};

	unordered_map<string, double> label2time;
	unordered_map<string, double> label2avgTime;

	ImVec4 color_high = ImVec4(1.000f, 0.244f, 0.000f, 1.000f);
	ImVec4 color_mid = ImVec4(1.000f, 0.602f, 0.000f, 1.000f);
	ImVec4 color_low = ImVec4(1.000f, 0.889f, 0.000f, 1.000f);
	
	void DisplayKernelTiming(const string name, bool autoColor = true)
	{
		double total = label2avgTime["Total"];
		bool shouldPop = false;
		float percentage = total > 0 ? (label2avgTime[name] / total * 100) : 0;
		if (autoColor && percentage > 10)
		{
			ImVec4 textColor = color_low;
			if (percentage > 30)
				textColor = color_high;
			else if (percentage > 10)
				textColor = color_mid;
			ImGui::PushStyleColor(ImGuiCol_Text, textColor);
			shouldPop = true;
		}

		ImGui::TableNextColumn(); 
		ImGui::Text(name.c_str()); 
		ImGui::TableNextColumn(); 
		ImGui::Text("%.2f ms", label2time[name]); 
		ImGui::TableNextColumn(); 
		ImGui::Text("%.2f ms", label2avgTime[name] / count); 
		ImGui::TableNextColumn(); 
		ImGui::Text("%.2f %%", percentage);

		if (shouldPop)
		{
			ImGui::PopStyleColor();
		}
	}

	void Update()
	{
		if (Timer::PeriodicUpdate("GUI_SOLVER", 0.2f))
		{
			if (Timer::frameCount() < 2)
			{
				count = 0;
				for (const auto& label : labels)
				{
					label2avgTime[label] = 0;
				}
				label2avgTime["KernelSum"] = 0;
			}

			label2time["KernelSum"] = 0;
			for (const auto& label : labels)
			{
				label2time[label] = Timer::GetTimerGPU("Solver_" + label);
				label2avgTime[label] += label2time[label];

				if (label != "Total" && label != "Initialize") label2time["KernelSum"] += label2time[label];
			}

			label2avgTime["KernelSum"] += label2time["KernelSum"];
			count++;
		}
	}

	void OnGUI()
	{
		if (!ImGui::CollapsingHeader("Solver timing", ImGuiTreeNodeFlags_DefaultOpen))
		{
			return;
		}

		float averageGPUTime = label2avgTime["KernelSum"] / count;
		int averageFPS = (int)(1000.0f / averageGPUTime);
		ImGui::Text("Average GPU time: %.2f ms (%d fps)", averageGPUTime, averageFPS);

		static bool hasPrinted = false;
		int printAtFrame = 300;
		if (!hasPrinted && Timer::physicsFrameCount() == printAtFrame)
		{
			hasPrinted = true;
			fmt::print("Info(GUI): Average GPU time at frame({}) is: {:.3f} ms\n", printAtFrame, label2avgTime["KernelSum"] / count);
		}

		if (ImGui::BeginTable("timing", 4))//,  ImGuiTableFlags_BordersOuter))
		{
			//ImGui::PushItemWidth(20);            
			ImGui::TableSetupColumn("Kernel");
			ImGui::TableSetupColumn("Time (ms)");
			ImGui::TableSetupColumn("Avg (ms)");
			ImGui::TableSetupColumn("%");
			ImGui::TableHeadersRow();

			for (int i = 0; i < labels.size() -1; i++)
			{
				auto& label = labels[i];
				DisplayKernelTiming(label);
			}
			DisplayKernelTiming(string("KernelSum"), false);
			DisplayKernelTiming(labels[labels.size() - 1], false);

			ImGui::EndTable();
		}
	}
};

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
		if (Global::gameState.pause) return;

		const auto& game = Global::game;
		float elapsedTime = Timer::elapsedTime();
		float deltaTimeMiliseconds = Timer::deltaTime() * 1000;

		frameCount = Timer::frameCount();
		physicsFrameCount = Timer::physicsFrameCount();

		if (Timer::PeriodicUpdate("GUI_FAST", 0.03f))
		{
			graphValues[graphIndex] = deltaTimeMiliseconds;
			graphIndex = (graphIndex + 1) % IM_ARRAYSIZE(graphValues);
		}

		if (Timer::PeriodicUpdate("GUI_SLOW", 0.2f))
		{
			deltaTime = deltaTimeMiliseconds;
			frameRate = elapsedTime > 0 ? (int)(frameCount / elapsedTime) : 0;
			cpuTime = Timer::GetTimer("CPU_TIME") * 1000;
			gpuTime = Timer::GetTimer("GPU_TIME") * 1000;

			for (int n = 0; n < IM_ARRAYSIZE(graphValues); n++)
				graphAverage += graphValues[n];
			graphAverage /= (float)IM_ARRAYSIZE(graphValues);
		}
	}

	void OnGUI()
	{
		if (ImGui::BeginTable("stat", 2, ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableNextColumn(); ImGui::Text("Frame: ");
			ImGui::TableNextColumn(); ImGui::Text("%d", frameCount);
			ImGui::TableNextColumn(); ImGui::Text("Physics Frame: ");
			ImGui::TableNextColumn(); ImGui::Text("%d", physicsFrameCount);
			ImGui::TableNextColumn(); ImGui::Text("Avg FrameRate: ");
			ImGui::TableNextColumn(); ImGui::Text("%d FPS", frameRate);
			ImGui::TableNextColumn(); ImGui::Text("CPU time: ");
			ImGui::TableNextColumn(); ImGui::Text("%.2f ms", cpuTime);
			ImGui::TableNextColumn(); ImGui::Text("GPU time: ");
			ImGui::TableNextColumn(); ImGui::Text("%.2f ms", gpuTime);
			ImGui::TableNextColumn(); ImGui::Text("Num Particles: ");
			ImGui::TableNextColumn(); ImGui::Text("%d", Global::simParams.numParticles);
			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0, 5));
		ImGui::PushItemWidth(-FLT_MIN);
		auto overlay = fmt::format("{:.2f} ms ({:.2f} FPS)", deltaTime, 1000.0 / deltaTime);
		ImGui::PlotLines("##", graphValues, IM_ARRAYSIZE(graphValues), graphIndex, overlay.c_str(),
			0, graphAverage * 2.0f, ImVec2(0, 80.0f));
		ImGui::Dummy(ImVec2(0, 5));

	}
};


void GUI::RegisterDebug(function<void()> callback)
{
	g_Gui->m_showDebugInfo.Register(callback);
}

void GUI::RegisterDebugOnce(function<void()> callback)
{
	g_Gui->m_showDebugInfoOnce.Register(callback);
}

void GUI::RegisterDebugOnce(const string& debugMessage)
{
	//vprintf(debugMessage, args);
	g_Gui->m_showDebugInfoOnce.Register([debugMessage]() {
		ImGui::Text(debugMessage.c_str());
		});
}

GUI::GUI(GLFWwindow* window)
{
	g_Gui = this;
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
	CustomizeStyle();

	// Setup Platform/Renderer backends
	const char* glsl_version = "#version 330";
	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	m_deviceName = string((char*)glGetString(GL_RENDERER));
	m_deviceName = m_deviceName.substr(0, m_deviceName.find("/"));
}

void GUI::OnUpdate()
{
	//static bool show_demo_window = true;
	//ImGui::ShowDemoWindow(&show_demo_window);
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	glfwGetWindowSize(m_window, &m_canvasWidth, &m_canvasHeight);

	ShowSceneWindow();
	ShowOptionWindow();
	ShowStatWindow();
}

void GUI::Render()
{
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::ClearCallback()
{
	m_showDebugInfo.Clear();
	m_showDebugInfoOnce.Clear();
}

void GUI::ShutDown()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void GUI::CustomizeStyle()
{
	ImGui::StyleColorsDark();

	auto style = &ImGui::GetStyle();
	style->SelectableTextAlign = ImVec2(0, 0.5);
	style->WindowPadding = ImVec2(10, 12);
	style->WindowRounding = 6;
	style->GrabRounding = 8;
	style->FrameRounding = 6;
	style->WindowTitleAlign = ImVec2(0.5, 0.5);

	style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 0.7f);
	style->Colors[ImGuiCol_TitleBg] = style->Colors[ImGuiCol_WindowBg];
	style->Colors[ImGuiCol_TitleBgActive] = style->Colors[ImGuiCol_TitleBg];
	style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.325f, 0.325f, 0.325f, 1.0f);
	style->Colors[ImGuiCol_FrameBg] = ImVec4(0.114f, 0.114f, 0.114f, 1.0f);
	style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
	style->Colors[ImGuiCol_Button] = ImVec4(0.46f, 0.46f, 0.46f, 0.46f);
	style->Colors[ImGuiCol_CheckMark] = ImVec4(0.851f, 0.851f, 0.851f, 1.0f);

	style->Colors[ImGuiCol_TableBorderLight] = ImVec4(1.0f, 1.0f, 1.0f, 0.3f);
	style->Colors[ImGuiCol_TableBorderStrong] = ImVec4(1.0f, 1.0f, 1.0f, 0.6f);
	//ImGui::StyleColorsClassic();ImGuiCol_TableBorderLight
}

void GUI::ShowSceneWindow()
{
	ImGui::SetNextWindowSize(ImVec2(k_leftWindowWidth, (m_canvasHeight - 60.0f) * 0.4f));
	ImGui::SetNextWindowPos(ImVec2(20, 20));
	ImGui::Begin("Scene", NULL, k_windowFlags);

	const auto& scenes = Global::engine->scenes;

	for (unsigned int i = 0; i < scenes.size(); i++)
	{
		auto scene = scenes[i];
		auto label = scene->name;
		if (ImGui::Selectable(label.c_str(), Global::engine->sceneIndex == i, 0, ImVec2(0, 28)))
		{
			Global::engine->SwitchScene(i);
		}
	}

	ImGui::End();
}

void GUI::ShowOptionWindow()
{
	ImGui::SetNextWindowSize(ImVec2(k_leftWindowWidth, (m_canvasHeight - 60.0f) * 0.6f));
	ImGui::SetNextWindowPos(ImVec2(20, 40 + (m_canvasHeight - 60.0f) * 0.4f));
	ImGui::Begin("Options", NULL, k_windowFlags);

	ImGui::PushItemWidth(-FLT_MIN);

	if (ImGui::Button("Reset (R)", ImVec2(-FLT_MIN, 0)))
	{
		Global::engine->Reset();
	}
	ImGui::Dummy(ImVec2(0.0f, 10.0f));

	if (ImGui::CollapsingHeader("Global", ImGuiTreeNodeFlags_DefaultOpen))
	{
		static bool radio = false;
		ImGui::Checkbox("Pause (P, O)", &Global::gameState.pause);
		Global::input->ToggleOnKeyDown(GLFW_KEY_P, Global::gameState.pause);
		ImGui::Checkbox("Draw Particles (K)", &Global::gameState.drawParticles);
		Global::input->ToggleOnKeyDown(GLFW_KEY_K, Global::gameState.drawParticles);
		ImGui::Checkbox("Draw Wireframe (L)", &Global::gameState.renderWireframe);
		Global::input->ToggleOnKeyDown(GLFW_KEY_L, Global::gameState.renderWireframe);
		ImGui::Dummy(ImVec2(0.0f, 10.0f));
	}

	if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Global::simParams.OnGUI();
	}

	ImGui::End();
}

void GUI::ShowStatWindow()
{
	ImGui::SetNextWindowSize(ImVec2(k_rightWindowWidth * 1.1f, 0));
	ImGui::SetNextWindowPos(ImVec2(m_canvasWidth - k_rightWindowWidth * 1.1f - 20, 20.0f));
	ImGui::Begin("Statistics", NULL, k_windowFlags);
	ImGui::Text("Device:  %s", m_deviceName.c_str());

	static PerformanceStat stat;
	stat.Update();
	stat.OnGUI();

	static SolverTiming solverTiming;
	solverTiming.Update();
	solverTiming.OnGUI();

	if (!m_showDebugInfo.empty() || !m_showDebugInfoOnce.empty())
	{
		if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen))
		{
			m_showDebugInfo.Invoke();
			m_showDebugInfoOnce.Invoke();

			if (!Global::gameState.pause) m_showDebugInfoOnce.Clear();
		}
	}

	ImGui::End();
}