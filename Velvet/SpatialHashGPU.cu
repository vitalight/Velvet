#include "SpatialHashGPU.cuh"

#include <cub/device/device_radix_sort.cuh>

#include "Timer.hpp"
#include "VtBuffer.hpp"

using namespace Velvet;

__device__ __constant__ HashParams d_params;
HashParams h_params;

__device__ inline int ComputeIntCoord(float value)
{
	return (int)floor(value / d_params.cellSpacing);
}

__device__ inline int HashCoords(int x, int y, int z)
{
	int h = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);	// fantasy function
	return abs(h % d_params.tableSize);
}

__device__ inline int HashPosition(glm::vec3 position)
{
	int x = ComputeIntCoord(position.x);
	int y = ComputeIntCoord(position.y);
	int z = ComputeIntCoord(position.z);

	int h = HashCoords(x, y, z);
	return h;
}

__global__ void ComputeParticleHash_Kernel(
	uint* particleHash,
	uint* particleIndex,
	CONST(glm::vec3*) positions)
{
	GET_CUDA_ID(id, d_params.numObjects);
	particleHash[id] = HashPosition(positions[id]);
	particleIndex[id] = id;
}

__global__ void FindCellStart_Kernel(
	uint* cellStart,
	uint* cellEnd,
	CONST(uint*) particleHash)
{
	extern __shared__ uint sharedHash[];

	GET_CUDA_ID_NO_RETURN(id, d_params.numObjects);

	uint hash = particleHash[id];
	sharedHash[threadIdx.x + 1] = hash;
	if (id > 0 && threadIdx.x == 0)
	{
		sharedHash[0] = particleHash[id - 1];
	}
	__syncthreads();

	if (id >= d_params.numObjects) return;

	if (id == 0 || hash != sharedHash[threadIdx.x])
	{
		cellStart[hash] = id;

		if (id > 0)
		{
			cellEnd[sharedHash[threadIdx.x]] = id;
		}
	}

	if (id == d_params.numObjects - 1)
	{
		cellEnd[hash] = id + 1;
	}
}

__global__ void CacheNeighbors_Kernel(
	uint* neighbors,
	CONST(uint*) particleIndex,
	CONST(uint*) cellStart,
	CONST(uint*) cellEnd,
	CONST(glm::vec3*) positions,
	CONST(glm::vec3*) originalPositions)
{
	GET_CUDA_ID(thread_id, d_params.numObjects);
	uint id = particleIndex[thread_id];
	//GET_CUDA_ID(id, d_params.numObjects);

	glm::vec3 position = positions[id];
	glm::vec3 originalPos = originalPositions[id];
	int ix = ComputeIntCoord(position.x);
	int iy = ComputeIntCoord(position.y);
	int iz = ComputeIntCoord(position.z);

	int neighborIndex = id;
	for (int x = ix - 1; x <= ix + 1; x++)
	{
		for (int y = iy - 1; y <= iy + 1; y++)
		{
			for (int z = iz - 1; z <= iz + 1; z++)
			{
				int h = HashCoords(x, y, z);
				int start = cellStart[h];
				if (start == 0xffffffff) continue;

				int end = min(cellEnd[h], start + d_params.maxNumNeighbors);

				for (int i = start; i < end; i++)
				{
					uint neighbor = particleIndex[i];
					// ignore collision when particles are initially close
					if (neighbor != id &&
						(length2(position - positions[neighbor]) < d_params.cellSpacing2) &&
						(length2(originalPos - originalPositions[neighbor]) > d_params.particleDiameter2))
					{
						neighbors[neighborIndex] = neighbor;
						neighborIndex += d_params.numObjects;
						if (neighborIndex >= d_params.numObjects * d_params.maxNumNeighbors) return;
					}
				}
			}
		}
	}
	if (neighborIndex < d_params.numObjects * d_params.maxNumNeighbors)
	{
		neighbors[neighborIndex] = 0xffffffff;
	}
}

// cub::sort outperform thrust (roughly half time)
void Sort(
	uint* d_keys_in,
	uint* d_values_in,
	int num_items,
	int maxBit)
{
	//static void* d_temp_storage = NULL;
	static VtBuffer<void*> d_temp_storage;
	static size_t temp_storage_bytes = 0;

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
	const HashParams params)
{
	{
		ScopedTimerGPU timer("Solver_HashParticle");

		h_params = params;
		checkCudaErrors(cudaMemcpyToSymbolAsync(d_params, &params, sizeof(HashParams)));
		CUDA_CALL(ComputeParticleHash_Kernel, h_params.numObjects)(particleHash, particleIndex, positions);
	}

	{
		ScopedTimerGPU timer("Solver_HashSort");
		int maxBit = (int)ceil(log2(h_params.tableSize));
		Sort(particleHash, particleIndex, h_params.numObjects, maxBit);
	}

	{
		ScopedTimerGPU timer("Solver_HashBuildCell");
		cudaMemsetAsync(cellStart, 0xffffffff, sizeof(uint) * (h_params.tableSize + 1));
		uint numBlocks, numThreads;
		ComputeGridSize(h_params.numObjects, numBlocks, numThreads);
		uint smemSize = sizeof(uint) * (numThreads + 1);
		CUDA_CALL_V(FindCellStart_Kernel, numBlocks, numThreads, smemSize)(cellStart, cellEnd, particleHash);
	}
	{
		ScopedTimerGPU timer("Solver_HashCache");
		CUDA_CALL(CacheNeighbors_Kernel, h_params.numObjects)(neighbors, particleIndex, cellStart, cellEnd,
			positions, originalPositions);
	}
}

