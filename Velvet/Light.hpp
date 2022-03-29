#pragma once

#include "Component.hpp"
#include "Global.hpp"
#include "GameInstance.hpp"

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
		Light(LightType _type = LightType::SpotLight)
		{
			Global::lights.push_back(this);
			name = __func__;
			type = _type;
		}

		~Light()
		{
			auto& lights = Global::lights;
			auto position = std::find(lights.begin(), lights.end(), this);
			if (position != lights.end())
			{
				lights.erase(position);
			}
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
		glm::vec3 color = glm::vec3(1.3f);
		float ambient = 0.15f;
		float innerCutoff = 40.0f;
		float outerCutoff = 50.0f;

		float constant = 1.0f;
		float linear = 0.09f;
		float quadratic = 0.032f;
	};
}