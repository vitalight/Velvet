#pragma once

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Component.h"
#include "Global.h"

using namespace std;

namespace Velvet
{
	class Camera : public Component
	{
	public:
		Camera()
		{
			name = __func__;
			Global::mainCamera = this;
		}

		// TODO: use actor position
		glm::mat4 view() const
		{
			return glm::lookAt(position, position + front, up);
		}

	//private:
		glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f);
		glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
		glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
		float zoom = 45;
	};
}