#include "VtClothSolverGPU.cuh"

#include <tuple>
#include <fmt/format.h>

#include "helper_cuda.h"

using namespace std;

namespace Velvet
{
	__device__ __constant__ SimulationParams d_params;


	void SetSimulationParams(SimulationParams* hostParams)
	{
		checkCudaErrors(cudaMemcpyToSymbol(d_params, hostParams, sizeof(SimulationParams)));
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

	__global__
	void ApplyExternalForces_Impl(glm::vec3* positions, glm::vec3* velocities)
	{
		uint id = blockIdx.x * blockDim.x + threadIdx.x;

		if (id < d_params.numParticles)
		{
			glm::vec3 gravity = glm::vec3(0, -10, 0);
			velocities[id] += d_params.gravity * d_params.deltaTime;
			positions[id] += velocities[id] * d_params.deltaTime;
		}
	}

	const uint blockSize = 256;

	void ComputeGridSize(uint n, uint &numBlocks, uint &numThreads)
	{
		if (n == 0)
		{
			fmt::print("Error(Solver): numParticles is 0\n");
			numBlocks = 0;
			numThreads = 0;
			return;
		}
		numThreads = min(n, blockSize);
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
	}

	void ApplyExternalForces(glm::vec3* positions, glm::vec3* velocities, uint numParticles)
	{
		uint numBlocks, numThreads;
		ComputeGridSize(numParticles, numBlocks, numThreads);
		ApplyExternalForces_Impl <<< numBlocks, numThreads >>> (positions, velocities);
	}
}