#pragma once

#include "fmt/format.h"
#include <glm/glm.hpp>

template <>
struct fmt::formatter<glm::vec3> : fmt::formatter<std::string> {
	auto format(glm::vec3 p, format_context& ctx) {
		return formatter<std::string>::format(
			fmt::format("[{}, {}, {}]", p.x, p.y, p.z), ctx);
	}
};

namespace Velvet
{
	namespace Helper
	{
		glm::mat4 RotateWithDegree(glm::mat4 result, const glm::vec3& rotation);

		glm::vec3 RotateWithDegree(glm::vec3 result, const glm::vec3& rotation);
	}
}