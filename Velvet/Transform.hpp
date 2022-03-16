#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace Velvet
{
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
}