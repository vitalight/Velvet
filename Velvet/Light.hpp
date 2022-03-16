#pragma once

#include "Component.hpp"
#include "Global.hpp"
#include "VtGraphics.hpp"

namespace Velvet
{
	class Light : public Component
	{
	public:
		Light()
		{
			Global::light = this;
		}

		glm::vec3 lightColor = glm::vec3(1.0f);
	};
}