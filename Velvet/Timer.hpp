#pragma once

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>

#include <fmt/printf.h>

#include "Global.hpp"
#include "GameInstance.hpp"

using namespace std;

namespace Velvet
{
	class Timer
	{
	public:

		Timer()
		{
			s_timer = this;
		}

		static void StartTimer(string label)
		{
			s_timer->times[label] = CurrentTime();
		}

		static double EndTimer(string label, int frame = -1)
		{
			double time = CurrentTime() - s_timer->times[label];
			if (frame == -1)
			{
				frame = s_timer->m_frameCount;
			}

			if (s_timer->times.count(label))
			{
				if (s_timer->frames.count(label) && frame > s_timer->frames[label])
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
		static double GetTimer(string label)
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
			using namespace std::chrono;
			using SecondsFP = duration<double>;
			return duration_cast<SecondsFP>(high_resolution_clock::now().time_since_epoch()).count();
		}

	public:
		static void UpdateDeltaTime()
		{
			float current = (float)glfwGetTime();
			s_timer->m_deltaTime = min(current - s_timer->m_lastUpdateTime, 0.2f);
			s_timer->m_lastUpdateTime = current;
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
		inline static Timer* s_timer;

		unordered_map<string, double> times;
		unordered_map<string, double> history;
		unordered_map<string, int> frames;

		int m_frameCount = 0;
		int m_physicsFrameCount = 0;
		float m_elapsedTime = 0.0f;
		float m_deltaTime = 0.0f;
		const float m_fixedDeltaTime = 1.0f / 60.0f;

		float m_lastUpdateTime = (float)glfwGetTime();
		float m_fixedUpdateTimer = (float)glfwGetTime();
	};
}