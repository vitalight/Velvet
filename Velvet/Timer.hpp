#pragma once

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>

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

		static double EndTimer(string label)
		{
			if (times.count(label))
			{
				history[label] = CurrentTime() - times[label];
				return history[label];
			}
			else
			{
				fmt::print("Warning(Timer): EndTimer with undefined label[{}].\n", label);
				return -1;
			}
		}

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
	};
}