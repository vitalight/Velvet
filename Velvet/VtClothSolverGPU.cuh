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

		float deltaTime;
		glm::mat4 invCurTransform;
		glm::mat4 lastTransform;

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
	
		__device__ glm::vec3 VelocityAt(const glm::vec3 targetPosition)
		{
			glm::vec4 lastPos = lastTransform * invCurTransform * glm::vec4(targetPosition, 1.0);
			glm::vec3 vel = (targetPosition - glm::vec3(lastPos)) / deltaTime;
			return vel;
		}
	};

	void SetSimulationParams(VtSimParams* hostParams);

	void InitializePositions(glm::vec3* positions, const int start, const int count, const glm::mat4 modelMatrix);

	void PredictPositions(
		glm::vec3* predicted,
		glm::vec3* velocities,
		CONST(glm::vec3*) positions,
		const float deltaTime);

	void SolveStretch(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(int*) stretchIndices,
		CONST(float*) stretchLengths,
		CONST(float*) invMasses,
		const uint numConstraints);
	
	// Bending doesn't work well with Jacobi. Small compliance lead to shaking, large compliance makes no effect.
	// It's recommended to disable this.
	void SolveBending(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(uint*) bendingIndices,
		CONST(float*) bendingAngles,
		CONST(float*) invMass,
		const uint numConstraints,
		const float deltaTime);

	void SolveAttachment(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(float*) invMass,
		CONST(int*) attachIndices,
		CONST(glm::vec3*) attachPositions,
		CONST(float*) attachDistances,
		const int numConstraints);

	void ApplyDeltas(glm::vec3* predicted, glm::vec3* deltas, int* deltaCounts);

	void CollideSDF(
		glm::vec3* predicted,
		CONST(SDFCollider*) colliders,
		CONST(glm::vec3*) positions,
		const uint numColliders,
		const float deltaTime);

	void CollideParticles(
		glm::vec3* deltas,
		int* deltaCounts,
		glm::vec3* predicted,
		CONST(float*) invMasses,
		CONST(uint*) neighbors,
		CONST(glm::vec3*) positions);

	void Finalize(
		glm::vec3* velocities,
		glm::vec3* positions,
		CONST(glm::vec3*) predicted,
		const float deltaTime);

	void ComputeNormal(
		glm::vec3* normals,
		CONST(glm::vec3*) positions,
		CONST(uint*) indices,
		const uint numTriangles);
}
