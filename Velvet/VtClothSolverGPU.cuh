#pragma once

#include "Common.cuh"
#include "Common.hpp"

namespace Velvet
{
	struct SDFCollider
	{
		enum class SDFColliderType
		{
			Sphere,
			Plane,
		};

		SDFColliderType type;
		glm::vec3 position;
		glm::vec3 scale;

		__device__ glm::vec3 ComputeSDF(const glm::vec3 targetPosition, const float collisionMargin) const
		{
			if (type == SDFColliderType::Plane)
			{
				float offset = targetPosition.y - (position.y + collisionMargin);
				if (offset < 0)
				{
					return glm::vec3(0, -offset, 0);
				}
			}
			else if (type == SDFColliderType::Sphere)
			{
				float radius = scale.x + collisionMargin;
				auto diff = targetPosition - position;
				float distance = glm::length(diff);
				float offset = distance - radius;
				if (offset < 0)
				{
					glm::vec3 direction = diff / distance;
					return -offset * direction;
				}
			}
			return glm::vec3(0);
		}
	};

	void SetSimulationParams(VtSimParams* hostParams);

	void InitializePositions(glm::vec3* positions, int count, glm::mat4 modelMatrix);

	void EstimatePositions(CONST(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime);

	void SolveStretch(uint numConstraints, CONST(int*) stretchIndices, CONST(float*) stretchLengths,
		CONST(float*) inverseMass, glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount);

	void SolveBending(
		glm::vec3* predicted,
		glm::vec3* positionDeltas,
		int* positionDeltaCount,
		CONST(uint*) bendingIndices,
		CONST(float*) bendingAngles,
		CONST(float*) invMass,
		uint numConstraints,
		float deltaTime);

	void SolveAttachment(
		int numConstraints,
		CONST(float*) invMass,
		CONST(int*) attachIndices,
		CONST(glm::vec3*) attachPositions,
		CONST(float*) attachDistances,
		glm::vec3* predicted,
		glm::vec3* positionDeltas,
		int* positionDeltaCount);

	void ApplyDeltas(glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount);

	void CollideSDF(const uint numColliders, CONST(SDFCollider*) colliders, CONST(glm::vec3*) positions, glm::vec3* predicted);

	void CollideParticles(
		CONST(float*) inverseMass,
		CONST(uint*) neighbors,
		CONST(glm::vec3*) positions,
		glm::vec3* predicted);

	void UpdatePositionsAndVelocities(CONST(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime);

	void ComputeNormal(uint numTriangles, CONST(glm::vec3*) positions, CONST(uint*) indices, glm::vec3* normals);

}
