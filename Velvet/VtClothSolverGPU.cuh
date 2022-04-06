#pragma once

#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "helper_cuda.h"

typedef unsigned int uint;

#define READ_ONLY(type) const type const

namespace Velvet
{
	struct SimulationParams
	{
		uint numParticles;
		glm::vec3 gravity;
		float damping;
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

	void EstimatePositions(READ_ONLY(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime);

	void SolveStretch(uint numConstraints, READ_ONLY(int*) stretchIndices, READ_ONLY(float*) stretchLengths,
		READ_ONLY(float*) inverseMass, READ_ONLY(glm::vec3*) predicted, glm::vec3* positionDeltas, int* positionDeltaCount);

	void UpdatePositionsAndVelocities(READ_ONLY(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime);

	void SolveAttachment(int numConstraints, READ_ONLY(int*) attachIndices, READ_ONLY(glm::vec3*) attachPositions, glm::vec3* predicted);

	void ApplyPositionDeltas(glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount);
}
