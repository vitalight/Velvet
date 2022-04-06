#pragma once

#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "helper_cuda.h"

typedef unsigned int uint;

#define CONST(type) const type const

namespace Velvet
{
	struct SimulationParams
	{
		uint numParticles;
		glm::vec3 gravity;

		float damping;
		float collisionMargin;
		float particleDiameter;
	};

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

	template<class T>
	inline T* VtAllocBuffer(int elementCount)
	{
		T* devPtr = nullptr;
		checkCudaErrors(cudaMallocManaged((void**)&devPtr, elementCount * sizeof(T)));
		cudaDeviceSynchronize(); // this is necessary, otherwise realloc can cause crash
		return devPtr;
	}

	void inline VtFreeBuffer(void* buffer)
	{
		checkCudaErrors(cudaFree(buffer));
	}

	void SetSimulationParams(SimulationParams* hostParams);

	void InitializePositions(glm::vec3* positions, int count, glm::mat4 modelMatrix);

	void EstimatePositions(CONST(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime);

	void SolveStretch(uint numConstraints, CONST(int*) stretchIndices, CONST(float*) stretchLengths,
		CONST(float*) inverseMass, glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount);

	void UpdatePositionsAndVelocities(CONST(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime);

	void SolveAttachment(int numConstraints, CONST(int*) attachIndices, CONST(glm::vec3*) attachPositions, glm::vec3* predicted);

	void SolveSDFCollision(const uint numColliders, CONST(SDFCollider*) colliders, CONST(glm::vec3*) positions, glm::vec3* predicted);

	void ComputeNormal(uint numTriangles, CONST(glm::vec3*) positions, CONST(uint*) indices, glm::vec3* normals);

	void SolveParticleCollision(CONST(float*) inverseMass, glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount);
}
