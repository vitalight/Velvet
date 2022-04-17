#include "SpatialHashGPU.cuh"

#include <cub/device/device_radix_sort.cuh>

#include "Timer.hpp"
#include "VtBuffer.hpp"

using namespace Velvet;

__device__ __constant__ float d_hashCellSpacing;
__device__ __constant__ int d_hashTableSize;
int h_hashTableSize;

__device__ inline int ComputeIntCoord(float value)
{
	return (int)floor(value / d_hashCellSpacing);
}

__device__ inline int HashCoords(int x, int y, int z)
{
	int h = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);	// fantasy function
	return abs(h % d_hashTableSize);
}

__device__ inline int HashPosition(glm::vec3 position)
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
	CONST(glm::vec3*) positions,
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
	CONST(glm::vec3*) positions,
	CONST(glm::vec3*) originalPositions,
	const uint numObjects,
	const uint maxNumNeihgbors)
{
	GET_CUDA_ID(id, numObjects);

	glm::vec3 position = positions[id];
	glm::vec3 originalPos = originalPositions[id];
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

				int end = min(cellEnd[h], start+ maxNumNeihgbors);

				for (int i = start; i < end; i++)
				{
					uint neighbor = particleIndex[i];
					float distance = glm::length(position - positions[neighbor]);
					// ignore collision when particles are initially close
					bool filterCollision = glm::length(originalPos - originalPositions[neighbor]) > d_hashCellSpacing;
					if (distance < d_hashCellSpacing && filterCollision)
					{
						neighbors[neighborIndex++] = neighbor;
					}
				}
			}
		}
	}
	if (neighborIndex < (id+1) * maxNumNeihgbors)
	{
		neighbors[neighborIndex] = 0xffffffff;
	}
}

// Cub::sort provides better performance (2x) than thrust
void Sort(
	uint* d_keys_in,
	uint* d_values_in,
	int num_items)
{
	//static void* d_temp_storage = NULL;
	static VtBuffer<void*> d_temp_storage;
	static size_t temp_storage_bytes = 0;


	int maxBit = (int)ceil(log2(h_hashTableSize));
	// Determine temporary device storage requirements
	size_t new_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(NULL, new_storage_bytes,
		d_keys_in, d_keys_in, d_values_in, d_values_in, num_items);

	if (temp_storage_bytes != new_storage_bytes)
	{
		temp_storage_bytes = new_storage_bytes;
		d_temp_storage.resize(temp_storage_bytes);
	}

	// Run sorting operation
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
		d_keys_in, d_keys_in, d_values_in, d_values_in, num_items, 0, maxBit);
}

void Velvet::HashObjects(
	uint* particleHash,
	uint* particleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint* neighbors,
	CONST(glm::vec3*) positions,
	CONST(glm::vec3*) originalPositions,
	const uint numObjects,
	const uint maxNumNeighbors,
	const float hashCellSpacing, 
	const int hashTableSize)
{
	{
		ScopedTimerGPU timer("Solver_HashParticle");

		h_hashTableSize = hashTableSize;
		checkCudaErrors(cudaMemcpyToSymbolAsync(d_hashCellSpacing, &hashCellSpacing, sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbolAsync(d_hashTableSize, &hashTableSize, sizeof(int)));
		CUDA_CALL(ComputeParticleHash, numObjects)(particleHash, particleIndex, positions, numObjects);
	}

	{
		ScopedTimerGPU timer("Solver_HashSort");
		Sort(particleHash, particleIndex, numObjects);
	}

	{
		ScopedTimerGPU timer("Solver_HashBuildCell");
		cudaMemsetAsync(cellStart, 0xffffffff, sizeof(uint) * (hashTableSize + 1));
		uint numBlocks, numThreads;
		ComputeGridSize(numObjects, numBlocks, numThreads);
		uint smemSize = sizeof(uint) * (numThreads + 1);
		CUDA_CALL_V(FindCellStart, numBlocks, numThreads, smemSize)(cellStart, cellEnd, particleHash, numObjects);
	}
	{
		ScopedTimerGPU timer("Solver_HashCache");
		CUDA_CALL(CacheNeighbors, numObjects)(neighbors, particleIndex, cellStart, cellEnd, positions, originalPositions, numObjects, maxNumNeighbors);
	}
}

