#pragma once

#include <string>

namespace Velvet
{
	using namespace std;

	class VtActor;

	class VtComponent
	{
	public:
		virtual void Start() {}

		virtual void Update() { }

		virtual void OnDestroy() {}

		string name = "BaseComponent";

		VtActor* actor;
	};
}