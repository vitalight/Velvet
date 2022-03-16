#pragma once

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Component.hpp"
#include "Global.hpp"
#include "Actor.hpp"

using namespace std;

namespace Velvet
{
	class Camera : public Component
	{
	public:
		float zoom = 45.0f;

		Camera()
		{
			name = __func__;
			Global::mainCamera = this;
		}

		glm::vec3 position() const
		{
			//return position;
			return actor->transform->position;
		}

		glm::vec3 front() const
		{
			return Helper::RotateWithDegree(kFront, actor->transform->rotation);
		}

		glm::vec3 up() const
		{
			return Helper::RotateWithDegree(kUp, actor->transform->rotation);
		}

		glm::mat4 view() const
		{
			auto trans = actor->transform;
			auto rotation = trans->rotation;
			auto result = glm::lookAt(position(), position() + front(), up());
			return result;
		}

	private:
		//glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f);
		glm::vec3 kFront = glm::vec3(0.0f, 0.0f, -1.0f);
		glm::vec3 kUp = glm::vec3(0.0f, 1.0f, 0.0f);
	};
}