#pragma once

#include <string>

namespace Velvet
{
	using namespace std;

	class Actor;

	class Component
	{
	public:
		virtual void Start() {}

		virtual void Update() { }

		virtual void OnDestroy() {}

		string name = "BaseComponent";

		Actor* actor;
	};
}