#pragma once

#include "Component.hpp"
#include "Actor.hpp"
#include "Global.hpp"

namespace Velvet
{
	class Collider : public Component
	{
	public:
		// specific for sphere, return repulsion
		virtual glm::vec3 ComputeSDF(glm::vec3 position)
		{
			auto mypos = actor->transform->position;
			float radius = actor->transform->scale.x + Global::Sim::collisionMargin;

			auto diff = position - mypos;
			float distance = glm::length(diff);
			if (distance < radius)
			{
				auto direction = diff / distance;
				return (radius - distance) * direction;
			}
			return glm::vec3(0);
		}
	};
}