#include "VtHelper.hpp"
#include <glm\ext\matrix_transform.hpp>

namespace Velvet
{
	namespace Helper
	{
		glm::mat4 RotateWithDegree(glm::mat4 result, const glm::vec3& rotation)
		{
			result = glm::rotate(result, glm::radians(rotation.y), glm::vec3(0, 1, 0));
			result = glm::rotate(result, glm::radians(rotation.z), glm::vec3(0, 0, 1));
			result = glm::rotate(result, glm::radians(rotation.x), glm::vec3(1, 0, 0));
			return result;
		}

		glm::vec3 RotateWithDegree(glm::vec3 result, const glm::vec3& rotation)
		{
			glm::mat4 rotationMatrix(1);
			rotationMatrix = RotateWithDegree(rotationMatrix, rotation);
			result = rotationMatrix * glm::vec4(result, 0.0f);
			return glm::normalize(result);
		}
	}
}