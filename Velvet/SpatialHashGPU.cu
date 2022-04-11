#include "SpatialHashGPU.cuh"
#include "Timer.hpp"

using namespace Velvet;

__device__ __constant__ float d_hashCellSpacing;
__device__ __constant__ int d_hashTableSize;
__device__ const float distanceTolerance = 2.0f;

__device__ inline int ComputeIntCoord(float value)
{
	return (int)floor(value / d_hashCellSpacing);
}

__device__ inline int HashCoords(int x, int y, int z)
{
	int h = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);	// fantasy function
	return abs(h % d_hashTableSize);
}

__device__ inline int HashPosition(float3 position)
{
	int x = ComputeIntCoord(position.x);
	int y = ComputeIntCoord(position.y);
	int z = ComputeIntCoord(position.z);

	int h = HashCoords(x, y, z);
	return h;
}

// TODO(low): make all parameters conform (output, input, constants)
__global__ void ComputeParticleHash(
	uint* particleHash,
	uint* particleIndex,
	CONST(float3*) positions,
	uint numObjects)
{
	GET_CUDA_ID(id, numObjects);
	particleHash[id] = HashPosition(positions[id]);
	particleIndex[id] = id;
}

__global__ void FindCellStart(
	uint* cellStart,
	uint* cellEnd,
	CONST(uint*) particleHash,
	uint numObjects)
{
	extern __shared__ uint sharedHash[];

	GET_CUDA_ID_NO_RETURN(id, numObjects);

	uint hash = particleHash[id];
	sharedHash[threadIdx.x + 1] = hash;
	if (id > 0 && threadIdx.x == 0)
	{
		sharedHash[0] = particleHash[id - 1];
	}
	__syncthreads();

	if (id >= numObjects) return;

	if (id == 0 || hash != sharedHash[threadIdx.x])
	{
		cellStart[hash] = id;

		if (id > 0)
		{
			cellEnd[sharedHash[threadIdx.x]] = id;
		}
	}

	if (id == numObjects - 1)
	{
		cellEnd[hash] = id + 1;
	}
}


__global__ void CacheNeighbors_BF(
	uint* neighbors,
	CONST(uint*) particleIndex,
	CONST(uint*) cellStart,
	CONST(glm::vec3*) positions,
	const uint numObjects,
	const uint maxNumNeihgbors)
{
	GET_CUDA_ID(id, numObjects);
	int neighborIndex = id * maxNumNeihgbors;
	for (int neighbor = 0; neighbor < numObjects; neighbor++)
	{
		float distance = glm::length(positions[id] - positions[neighbor]);
		if (neighbor != id && distance < d_hashCellSpacing)
		{
			neighbors[neighborIndex++] = neighbor;
		}
	}
}

__global__ void CacheNeighbors(
	uint* neighbors,
	CONST(uint*) particleIndex,
	CONST(uint*) cellStart,
	CONST(uint*) cellEnd,
	CONST(float3*) positions,
	const uint numObjects,
	const uint maxNumNeihgbors)
{
	GET_CUDA_ID(id, numObjects);

	float3 position = positions[id];

	int ix = ComputeIntCoord(position.x);
	int iy = ComputeIntCoord(position.y);
	int iz = ComputeIntCoord(position.z);

	int neighborIndex = id * maxNumNeihgbors;
	for (int x = ix - 1; x <= ix + 1; x++)
	{
		for (int y = iy - 1; y <= iy + 1; y++)
		{
			for (int z = iz - 1; z <= iz + 1; z++)
			{
				int h = HashCoords(x, y, z);
				int start = cellStart[h];
				if (start == 0xffffffff) continue;

				int end = cellEnd[h];
				if (end > start + 64) end = start + 64;

				for (int i = start; i < end; i++)
				{
					uint neighbor = particleIndex[i];
					float distance = length(position - positions[neighbor]);
					if (neighbor != id && distance < distanceTolerance * d_hashCellSpacing)
					{
						neighbors[neighborIndex++] = neighbor;
					}
				}
			}
		}
	}
}

void Velvet::HashObjects(
	uint* particleHash,
	uint* particleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint* neighbors,
	CONST(float3*) positions,
	const uint numObjects,
	const uint maxNumNeighbors,
	const float hashCellSpacing, 
	const int hashTableSize)
{
	{
		ScopedTimerGPU timer("Solver_Hash_Particle");

		checkCudaErrors(cudaMemcpyToSymbol(d_hashCellSpacing, &hashCellSpacing, sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(d_hashTableSize, &hashTableSize, sizeof(int)));

		CUDA_CALL(ComputeParticleHash, numObjects)(particleHash, particleIndex, positions, numObjects);
	}

	{
		ScopedTimerGPU timer("Solver_Hash_Sort");
		// Sort Particle Hash
		thrust::sort_by_key(thrust::device_ptr<uint>(particleHash),
			thrust::device_ptr<uint>(particleHash + numObjects),
			thrust::device_ptr<uint>(particleIndex));
	}
	{
		ScopedTimerGPU timer("Solver_Hash_FindCellStart");
		cudaMemset(cellStart, 0xffffffff, sizeof(uint) * (hashTableSize + 1));
		uint numBlocks, numThreads;
		ComputeGridSize(numObjects, numBlocks, numThreads);
		uint smemSize = sizeof(uint) * (numThreads + 1);
		CUDA_CALL_V(FindCellStart, numBlocks, numThreads, smemSize)(cellStart, cellEnd, particleHash, numObjects);
	}
	{
		ScopedTimerGPU timer("Solver_Hash_CacheNeighbors");
		cudaMemset(neighbors, 0xffffffff, sizeof(uint) * maxNumNeighbors * numObjects);
		CUDA_CALL(CacheNeighbors, numObjects)(neighbors, particleIndex, cellStart, cellEnd, positions, numObjects, maxNumNeighbors);
	}
}

