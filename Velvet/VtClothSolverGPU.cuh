#pragma once

#include "Common.cuh"
#include "Common.hpp"

namespace Velvet
{
	struct SDFCollider
	{
		ColliderType type;

		glm::vec3 position;
		glm::vec3 scale;

		float deltaTime;
		glm::mat3 curTransform;
		glm::mat4 invCurTransform;
		glm::mat4 lastTransform;

		__device__ float sgn(float value) const { return (value > 0) ? 1.0f : (value < 0 ? -1.0f : 0.0f); }

		__device__ glm::vec3 ComputeSDF(const glm::vec3 targetPosition, const float collisionMargin) const
		{
			if (type == ColliderType::Plane)
			{
				float offset = targetPosition.y - (position.y + collisionMargin);
				if (offset < 0)
				{
					return glm::vec3(0, -offset, 0);
				}
			}
			else if (type == ColliderType::Sphere)
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
			else if (type == ColliderType::Cube)
			{
				glm::vec3 correction = glm::vec3(0);
				glm::vec3 localPos = invCurTransform * glm::vec4(targetPosition, 1.0);
				glm::vec3 cubeSize = glm::vec3(0.5f, 0.5f, 0.5f) + collisionMargin / scale;
				glm::vec3 offset = glm::abs(localPos) - cubeSize;

				float maxVal = max(offset.x, max(offset.y, offset.z));
				float minVal = min(offset.x, min(offset.y, offset.z)); 
				float midVal = offset.x  + offset.y + offset.z - maxVal - minVal;
				float scalar = 1.0f;

				if (maxVal < 0)
				{
					// make cube corner round to avoid particle vibration	
					float margin = 0.03f;
					if (midVal > -margin) scalar = 0.2f;
					if (minVal > -margin)
					{
						glm::vec3 mask;
						mask.x = offset.x < 0 ? sgn(localPos.x) : 0;
						mask.y = offset.y < 0 ? sgn(localPos.y) : 0;
						mask.z = offset.z < 0 ? sgn(localPos.z) : 0;

						glm::vec3 vec = offset + glm::vec3(margin);
						float len = glm::length(vec);
						if (len < margin)
							correction = mask * glm::normalize(vec) * (margin - len);
					}
					else if (offset.x == maxVal)
					{
						correction = glm::vec3(copysignf(-offset.x, localPos.x), 0, 0);
					}
					else if (offset.y == maxVal)
					{
						correction = glm::vec3(0, copysignf(-offset.y, localPos.y), 0);
					}
					else if (offset.z == maxVal)
					{
						correction = glm::vec3(0, 0, copysignf(-offset.z, localPos.z));
					}
				}
				return curTransform * scalar * correction;
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
		CONST(int*) attachParticleIDs,
		CONST(int*) attachSlotIDs,
		CONST(glm::vec3*) attachSlotPositions,
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
