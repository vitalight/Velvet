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
		static void StartTimer(string label)
		{
			times[label] = CurrentTime();
		}

		static double EndTimer(string label, int frame = -1)
		{
			double time = CurrentTime() - times[label];
			if (frame == -1)
			{
				frame = Global::game->frameCount;
			}

			if (times.count(label))
			{
				if (frames.count(label) && frame > frames[label])
				{
					history[label] = time;
				}
				else
				{
					history[label] += time;
				}
				frames[label] = frame;
				return history[label];
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
			if (history.count(label))
			{
				return history[label];
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

		inline static unordered_map<string, double> times;
		inline static unordered_map<string, double> history;
		inline static unordered_map<string, int> frames;
	};
}