#pragma once

#include "Component.hpp"
#include "Actor.hpp"
#include "Global.hpp"

namespace Velvet
{
	class Collider : public Component
	{
	public:
		int m_sphereOrPlane = 0;

		Collider(bool sphereOrPlane)
		{
			name = __func__;
			m_sphereOrPlane = sphereOrPlane;
		}

		virtual glm::vec3 ComputeSDF(glm::vec3 position)
		{
			if (m_sphereOrPlane)
			{
				return ComputePlaneSDF(position);
			}
			else
			{
				return ComputeSphereSDF(position);
			}
		}

		virtual glm::vec3 ComputePlaneSDF(glm::vec3 position)
		{
			if (position.y < Global::Sim::collisionMargin)
			{
				return glm::vec3(0, Global::Sim::collisionMargin - position.y, 0);
			}
			return glm::vec3(0);
		}

		virtual glm::vec3 ComputeSphereSDF(glm::vec3 position)
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