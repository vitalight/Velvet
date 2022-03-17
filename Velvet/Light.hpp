#pragma once

#include "Component.hpp"
#include "Global.hpp"
#include "VtGraphics.hpp"

namespace Velvet
{
	enum class LightType
	{
		Point,
		Directional,
	};

	class Light : public Component
	{
	public:
		Light()
		{
			Global::light = this;
		}

		glm::vec4 position()
		{
			if (type == LightType::Point)
			{
				return glm::vec4(transform()->position, 1.0);
			}
			else
			{
				return glm::vec4(transform()->position, 0.0);
			}
		}

		LightType type = LightType::Directional;
		glm::vec3 lightColor = glm::vec3(1.0f);
	};
}