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
