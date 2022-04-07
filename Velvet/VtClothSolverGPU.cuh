#pragma once

#include "Common.cuh"

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

	void SetSimulationParams(SimulationParams* hostParams);

	void InitializePositions(glm::vec3* positions, int count, glm::mat4 modelMatrix);

	void EstimatePositions(CONST(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime);

	void SolveStretch(uint numConstraints, CONST(int*) stretchIndices, CONST(float*) stretchLengths,
		CONST(float*) inverseMass, glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount);

	void UpdatePositionsAndVelocities(CONST(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime);

	void SolveAttachment(int numConstraints, CONST(int*) attachIndices, CONST(glm::vec3*) attachPositions, glm::vec3* predicted);

	void SolveSDFCollision(const uint numColliders, CONST(SDFCollider*) colliders, CONST(glm::vec3*) positions, glm::vec3* predicted);

	void ComputeNormal(uint numTriangles, CONST(glm::vec3*) positions, CONST(uint*) indices, glm::vec3* normals);

	void SolveParticleCollision(
		CONST(float*) inverseMass,
		CONST(uint*) neighbors,
		glm::vec3* predicted,
		glm::vec3* positionDeltas,
		int* positionDeltaCount);
}
