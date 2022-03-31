#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Helper.hpp"

namespace Velvet
{
	class Actor;

	class Transform
	{
	public:

		Transform(Actor* actor)
		{
			m_actor = actor;
		}

		glm::mat4 matrix()
		{
			glm::mat4 result = glm::mat4(1.0f);
			result = glm::translate(result, position);
			result = Helper::RotateWithDegree(result, rotation);
			result = glm::scale(result, scale);
			return result;
		}

		Actor* actor()
		{
			return m_actor;
		}

		void Reset()
		{
			position = glm::vec3(0.0f);
			rotation = glm::vec3(0.0f);
			scale = glm::vec3(1.0f);
		}

		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 rotation = glm::vec3(0.0f);
		glm::vec3 scale = glm::vec3(1.0f);
		Actor* m_actor;
	};
}