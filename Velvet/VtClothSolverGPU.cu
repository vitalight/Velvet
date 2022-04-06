#include "VtClothSolverGPU.cuh"

#include <tuple>
#include <fmt/format.h>

#include "helper_cuda.h"
#include <cuda_runtime.h>

using namespace std;

#define EPSILON 1e-6f
#define GET_CUDA_ID(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= maxID) return

namespace Velvet
{
	__device__ __constant__ SimulationParams d_params;
	SimulationParams h_params;

	void SetSimulationParams(SimulationParams* hostParams)
	{
		checkCudaErrors(cudaMemcpyToSymbol(d_params, hostParams, sizeof(SimulationParams)));
		h_params = *hostParams;
	}

	const uint blockSize = 256;

	void ComputeGridSize(uint n, uint& numBlocks, uint& numThreads)
	{
		if (n == 0)
		{
			//fmt::print("Error(Solver): numParticles is 0\n");
			numBlocks = 0;
			numThreads = 0;
			return;
		}
		numThreads = min(n, blockSize);
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
	}

	struct InitializePositionsFunctor
	{
		const glm::mat4 matrix;
		InitializePositionsFunctor(glm::mat4 _matrix) : matrix(_matrix) {}

		__host__ __device__
			glm::vec3 operator()(const glm::vec3 position) const {
			return glm::vec3(matrix * glm::vec4(position, 1));
		}
	};

	void InitializePositions(glm::vec3* positions, int count, glm::mat4 modelMatrix)
	{
		thrust::device_ptr<glm::vec3> d_positions(positions);
		thrust::transform(d_positions, d_positions + count, d_positions, InitializePositionsFunctor(modelMatrix));
	}

	__global__ void EstimatePositions_Impl(READ_ONLY(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		glm::vec3 gravity = glm::vec3(0, -10, 0);
		velocities[id] += d_params.gravity * deltaTime;
		predicted[id] = positions[id] + velocities[id] * deltaTime;
	}

	void EstimatePositions(READ_ONLY(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime)
	{
		uint numBlocks, numThreads;
		ComputeGridSize(h_params.numParticles, numBlocks, numThreads);
		EstimatePositions_Impl <<< numBlocks, numThreads >>> (positions, predicted, velocities, deltaTime);
	}

	__device__ void AtomicAdd(glm::vec3* address, int index, glm::vec3 val)
	{
		atomicAdd(&(address[index].x), val.x);
		atomicAdd(&(address[index].y), val.y);
		atomicAdd(&(address[index].z), val.z);
	}

	__global__ void SolveStretch_Impl(glm::vec3* predicted, int* stretchIndices, float* stretchLengths, float* inverseMass, uint numConstraints)
	{
		GET_CUDA_ID(id, numConstraints);

		int idx1 = stretchIndices[2 * id];
		int idx2 = stretchIndices[2 * id + 1];
		float expectedDistance = stretchLengths[id];

		glm::vec3 diff = predicted[idx1] - predicted[idx2];
		float distance = glm::length(diff);
		float w1 = inverseMass[idx1];
		float w2 = inverseMass[idx2];

		if (distance > expectedDistance && w1 + w2 > 0)
		{
			auto gradient = diff / (distance + EPSILON);
			// compliance is zero, therefore XPBD=PBD
			auto denom = w1 + w2;
			auto lambda = (distance - expectedDistance) / denom;
			auto correction1 = -w1 * lambda * gradient;
			auto correction2 = w2 * lambda * gradient;
			AtomicAdd(predicted, idx1, correction1);
			AtomicAdd(predicted, idx2, correction2);
		}
	}

	void SolveStretch(glm::vec3* predicted, int* stretchIndices, float* stretchLengths, float* inverseMass, uint numConstraints)
	{
		uint numBlocks, numThreads;
		ComputeGridSize(numConstraints, numBlocks, numThreads);
		SolveStretch_Impl <<< numBlocks, numThreads >>> (predicted, stretchIndices, stretchLengths, inverseMass, numConstraints);
	}

	__global__ void UpdatePositionsAndVelocities_Impl(READ_ONLY(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime)
	{
		// TODO: encapsulate macro
		uint id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= d_params.numParticles) return;

		velocities[id] = (predicted[id] - positions[id]) / deltaTime;// * (1 - d_params.damping * deltaTime);
		positions[id] = predicted[id];
	}

	void UpdatePositionsAndVelocities(READ_ONLY(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime)
	{
		uint numBlocks, numThreads;
		ComputeGridSize(h_params.numParticles, numBlocks, numThreads);
		UpdatePositionsAndVelocities_Impl <<< numBlocks, numThreads >>> (predicted, velocities, positions, deltaTime);
	}

	__global__ void SolveAttachment_Impl(int numConstraints, READ_ONLY(int*) attachIndices, READ_ONLY(glm::vec3*) attachPositions, glm::vec3* predicted)
	{
		GET_CUDA_ID(id, numConstraints);

		predicted[attachIndices[id]] = attachPositions[id];
	}

	void SolveAttachment(int numConstraints, READ_ONLY(int*) attachIndices, READ_ONLY(glm::vec3*) attachPositions, glm::vec3* predicted)
	{
		uint numBlocks, numThreads;
		ComputeGridSize(numConstraints, numBlocks, numThreads);
		SolveAttachment_Impl <<<numBlocks, numThreads >>> (numConstraints, attachIndices, attachPositions, predicted);
	}
}