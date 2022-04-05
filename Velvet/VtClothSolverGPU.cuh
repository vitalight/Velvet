#pragma once

#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "helper_cuda.h"

typedef unsigned int uint;


namespace Velvet
{
	struct SimulationParams
	{
		uint numParticles;
		glm::vec3 gravity;
		float deltaTime;
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

	void ApplyExternalForces(glm::vec3* positions, glm::vec3* velocities, uint numParticles);

	void SolveStretch(glm::vec3* predicted, int* stretchIndices, float* stretchLengths, float* inverseMass, uint numConstraints);
}
