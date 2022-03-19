#pragma once

#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include "MeshRenderer.hpp"
#include "Light.hpp"

namespace Velvet
{
	using namespace std;

	class Actor
	{
	public:
		Actor();

		Actor(string _name);

		static shared_ptr<Actor> PrefabTriangle() { return shared_ptr<Actor>(); }

		static shared_ptr<Actor> PrefabCube() { return shared_ptr<Actor>(); }

		static shared_ptr<Actor> PrefabQuad() { return shared_ptr<Actor>(); }

		static shared_ptr<Actor> PrefabCamera();

		static shared_ptr<Actor> PrefabLight(LightType type = LightType::Point) { return shared_ptr<Actor>(); }

		void Start();

		void Update();

		void OnDestroy();

		void AddComponent(shared_ptr<Component> component);

		template <typename T>
		enable_if_t<is_base_of<Component, T>::value, T*> GetComponent()
		{
			T* result = nullptr;
			for (auto c : components)
			{
				result = dynamic_cast<T*>(c.get());
				if (result)
					return result;
			}
			return result;
		}

		shared_ptr<Transform> transform = make_shared<Transform>(Transform(this));
		vector<shared_ptr<Component>> components;

		string name;

	};

}