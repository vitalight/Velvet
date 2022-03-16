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

	class Transform
	{
	public:

		glm::mat4 matrix()
		{
			glm::mat4 result = glm::mat4(1.0f);
			//glm::rotate(result, 
			result = glm::translate(result, position);
			result = glm::scale(result, scale);
			return result;
		}

		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 rotation = glm::vec3(0.0f);
		glm::vec3 scale = glm::vec3(1.0f);
	};

	class Actor
	{
	public:
		Actor();

		Actor(string _name);

		static shared_ptr<Actor> PrefabTriangle();

		static shared_ptr<Actor> PrefabCube();

		static shared_ptr<Actor> PrefabQuad();

		static shared_ptr<Actor> PrefabCamera();

		static shared_ptr<Actor> PrefabLight();

		void Start();

		void Update();

		void OnDestroy();

		void AddComponent(shared_ptr<Component> component);

		template <typename T>
		T* GetComponent()
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

		Transform transform;
		vector<shared_ptr<Component>> components;

		string name;

	};

}