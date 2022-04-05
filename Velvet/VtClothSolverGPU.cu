#include "VtClothSolverGPU.cuh"

#include <tuple>

#include "helper_cuda.h"

using namespace std;

namespace Velvet
{
	void AllocateArray(void** devPtr, size_t size)
	{
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void FreeArray(void* devPtr)
	{
		checkCudaErrors(cudaFree(devPtr));
	}

	struct TransformFunctor
	{
		const glm::mat4 matrix;
		TransformFunctor(glm::mat4 _matrix) : matrix(_matrix) {}

		__host__ __device__
			glm::vec3 operator()(const glm::vec3 position) const {
			return glm::vec3(matrix * glm::vec4(position, 1));
		}
	};

	void InitializePositions(glm::vec3* positions, int count, glm::mat4 modelMatrix)
	{
		thrust::device_ptr<glm::vec3> d_positions(positions);
		thrust::transform(d_positions, d_positions + count, d_positions, TransformFunctor(modelMatrix));
	}

	struct ApplyExternalForcesFunctor1
	{
		const glm::vec3 force;
		ApplyExternalForcesFunctor1(glm::vec3 _force) : force(_force) {}

		__host__ __device__
			glm::vec3 operator()(const glm::vec3 velocities) const {
			return velocities + force;
		}
	};


	__global__
	void ApplyExternalForces_Impl(glm::vec3* positions, glm::vec3* velocities, uint numParticles)
	{
		uint id = blockIdx.x * blockDim.x + threadIdx.x;

		if (id < numParticles)
		{
			glm::vec3 gravity = glm::vec3(0, -10, 0);
			float dt = 1.0f / 60.0f;
			velocities[id] += gravity * dt;
			positions[id] += velocities[id] * dt;
		}
	}

	const uint blockSize = 256;

	void ComputeGridSize(uint n, uint &numBlocks, uint &numThreads)
	{
		numThreads = min(n, blockSize);
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
	}

	void ApplyExternalForces(glm::vec3* positions, glm::vec3* velocities, uint count)
	{
		uint numBlocks, numThreads;
		ComputeGridSize(count, numBlocks, numThreads);
		ApplyExternalForces_Impl <<< numBlocks, numThreads >>> (positions, velocities, count);
	}
}