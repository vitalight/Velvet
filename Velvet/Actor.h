#pragma once

#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "MeshRenderer.h"

namespace Velvet
{
	using namespace std;

	class Actor
	{
	public:
		Actor();

		Actor(string name);

		static shared_ptr<Actor> PrefabTriangle();

		static shared_ptr<Actor> PrefabCube();

		static shared_ptr<Actor> PrefabQuad();

		void Start();

		void Update();

		void OnDestroy();

		void AddComponent(shared_ptr<Component> component);

		template <typename T>
		T* GetComponent()
		{
			T* result = nullptr;
			for (auto c : m_components)
			{
				result = dynamic_cast<T*>(c.get());
				if (result)
					return result;
			}
			return result;
		}

	private:
		string m_name;

		vector<shared_ptr<Component>> m_components;
	};

}