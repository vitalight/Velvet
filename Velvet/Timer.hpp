#pragma once

#include <iostream>
#include <unordered_map>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/printf.h>
#include <cuda_runtime.h>

//#include "Global.hpp"

using namespace std;

namespace Velvet
{

	class Timer
	{
	public:

		Timer()
		{
			s_timer = this;

			m_lastUpdateTime = (float)CurrentTime();
			m_fixedUpdateTimer = (float)CurrentTime();
		}

		~Timer()
		{
			for (const auto& label2events : cudaEvents)
			{
				for (auto& e : label2events.second)
				{
					cudaEventDestroy(e);
				}
			}
		}

		static void StartTimer(const string& label)
		{
			s_timer->times[label] = CurrentTime();
		}

		// Returns elapsed time from StartTimer in seconds.
		// When called multiple time during one frame, result gets accumulated.
		static double EndTimer(const string& label, int frame = -1)
		{
			double time = CurrentTime() - s_timer->times[label];
			if (frame == -1)
			{
				frame = s_timer->m_frameCount;
			}

			if (s_timer->times.count(label))
			{
				if (frame > s_timer->frames[label])
				{
					s_timer->history[label] = time;
				}
				else
				{
					s_timer->history[label] += time;
				}
				s_timer->frames[label] = frame;
				return s_timer->history[label];
			}
			else
			{
				fmt::print("Warning(Timer): EndTimer with undefined label[{}].\n", label);
				return -1;
			}
		}

		// returns time in seconds
		static double GetTimer(const string& label)
		{
			if (s_timer->history.count(label))
			{
				return s_timer->history[label];
			}
			else
			{
				return 0;
			}
		}

		static double CurrentTime()
		{
			return glfwGetTime();
		}
	public:
		static void StartTimerGPU(const string& label)
		{
			int frame = s_timer->m_frameCount;

			if (s_timer->frames.count(label) && s_timer->frames[label] != frame)
			{
				GetTimerGPU(label);
			}
			s_timer->frames[label] = frame;

			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			auto& events = s_timer->cudaEvents[label];
			events.push_back(start);
			events.push_back(end);
			cudaEventRecord(start);
		}

		static void EndTimerGPU(const string& label)
		{
			const auto& events = s_timer->cudaEvents[label];
			auto stop = events[events.size() - 1];
			cudaEventRecord(stop);
		}

		// return time in mili seconds
		static double GetTimerGPU(const string& label)
		{
			auto& events = s_timer->cudaEvents[label];
			if (events.size() > 0)
			{
				auto lastEvent = events[events.size() - 1];
				cudaEventSynchronize(lastEvent);

				float totalTime = 0.0f;
				for (int i = 0; i < events.size(); i += 2)
				{
					float time;
					cudaEventElapsedTime(&time, events[i], events[i + 1]);
					totalTime += time;
					cudaEventDestroy(events[i]);
					cudaEventDestroy(events[i + 1]);
				}
				events.clear();
				s_timer->history[label] = totalTime;
			}

			if (s_timer->history.count(label))
			{
				return s_timer->history[label];
			}
			else
			{
				return 0;
			}
		}
	public:
		static void UpdateDeltaTime()
		{
			float current = (float)CurrentTime();
			s_timer->m_deltaTime = min(current - s_timer->m_lastUpdateTime, 0.2f);
			s_timer->m_lastUpdateTime = current;

			//fmt::print("dt: {}\n", s_timer->m_deltaTime);
		}

		static void NextFrame()
		{
			s_timer->m_frameCount++;
			s_timer->m_elapsedTime += s_timer->m_deltaTime;
		}

		// Return true when fixed update should be executed
		static bool NextFixedFrame()
		{
			s_timer->m_fixedUpdateTimer += s_timer->m_deltaTime;

			if (s_timer->m_fixedUpdateTimer > s_timer->m_fixedDeltaTime)
			{
				s_timer->m_fixedUpdateTimer = 0;
				s_timer->m_physicsFrameCount++;
				return true;
			}
			return false;
		}

		static bool PeriodicUpdate(const string& label, float interval, bool allowRepetition = true)
		{
			auto& l2t = s_timer->label2accumulatedTime;
			if (!l2t.count(label))
			{
				l2t[label] = 0;
			}

			if (l2t[label] < s_timer->m_elapsedTime)
			{
				l2t[label] = allowRepetition ? l2t[label] + interval : s_timer->m_elapsedTime + interval;
				return true;
			}
			return false;
		}

		static auto frameCount()
		{
			return s_timer->m_frameCount;
		}

		static auto physicsFrameCount()
		{
			return s_timer->m_physicsFrameCount;
		}

		static auto elapsedTime()
		{
			return s_timer->m_elapsedTime;
		}

		static auto deltaTime()
		{
			return s_timer->m_deltaTime;
		}

		static auto fixedDeltaTime()
		{
			return s_timer->m_fixedDeltaTime;
		}
	private:
		static Timer* s_timer;

		unordered_map<string, double> times;
		unordered_map<string, double> history;
		unordered_map<string, int> frames;
		unordered_map<string, vector<cudaEvent_t>> cudaEvents;
		unordered_map<string, float> label2accumulatedTime;

		int m_frameCount = 0;
		int m_physicsFrameCount = 0;
		float m_elapsedTime = 0.0f;
		float m_deltaTime = 0.0f;
		const float m_fixedDeltaTime = 1.0f / 60.0f;

		float m_lastUpdateTime = 0.0f;
		float m_fixedUpdateTimer = 0.0f;
	};


	class ScopedTimerGPU
	{
	public:
		ScopedTimerGPU(const string&& _label)
		{
			//if (!Global::gameState.detailTimer) return;
			label = _label;
			Timer::StartTimerGPU(_label);
		}

		~ScopedTimerGPU()
		{
			//if (!Global::gameState.detailTimer) return;
			Timer::EndTimerGPU(label);
		}

	private:
		string label;
	};
}