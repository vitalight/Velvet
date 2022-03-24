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
		SpotLight,
	};

	class Light : public Component
	{
	public:
		Light()
		{
			Global::light.push_back(this);
			name = __func__;
		}

		glm::vec4 position()
		{
			if (type == LightType::Point || type == LightType::SpotLight)
			{
				return glm::vec4(transform()->position, 1.0);
			}
			else
			{
				return glm::vec4(transform()->position, 0.0);
			}
		}

		LightType type = LightType::Point;
		glm::vec3 color = glm::vec3(1.0f);
		float ambient = 0.4f;
		float innerCutoff = 45.0f;
		float outerCutoff = 60.0f;
		float constant = 1.0f;
		float linear = 0.09f;
		float quadratic = 0.032f;
	};
}