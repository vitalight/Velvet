#include "SpatialHashGPU.cuh"

using namespace Velvet;

__device__ __constant__ float d_hashCellSpacing;
__device__ __constant__ int d_hashTableSize;

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

// TODO: make all parameters conform (output, input, constants)
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

// TODO: shared mem (ref: cuda particles example)
__global__ void FindCellStart(
	uint* cellStart,
	CONST(uint*) particleHash,
	uint numObjects)
{
	GET_CUDA_ID(id, numObjects);

	uint hash = particleHash[id];
	uint prevHash = particleHash[id - 1];

	if (id == 0 || hash != prevHash)
	{
		cellStart[hash] = id;
	}

	if (id == numObjects - 1)
	{
		cellStart[d_hashTableSize] = numObjects;
	}
}

__global__ void CacheNeighbors(
	uint* neighbors,
	CONST(uint*) particleIndex,
	CONST(uint*) cellStart,
	CONST(glm::vec3*) positions,
	const uint numObjects,
	const uint maxNumNeihgbors)
{
	GET_CUDA_ID(id, numObjects);

	glm::vec3 position = positions[id];

	int ix = ComputeIntCoord(position.x);
	int iy = ComputeIntCoord(position.y);
	int iz = ComputeIntCoord(position.z);

	for (int x = ix - 1; x <= ix + 1; x++)
	{
		for (int y = iy - 1; y <= iy + 1; y++)
		{
			for (int z = iz - 1; z <= iz + 1; z++)
			{
				int h = HashCoords(x, y, z);
				int start = cellStart[h];
				int end = min(cellStart[h + 1], start + 64);

				for (int i = start; i < end; i++)
				{
					neighbors[id * maxNumNeihgbors + (i - start)] = particleIndex[i];
				}
			}
		}
	}
}

void Velvet::SetHashParams(float hashCellSpacing, int hashTableSize)
{
	checkCudaErrors(cudaMemcpyToSymbol(d_hashCellSpacing, &hashCellSpacing, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_hashTableSize, &hashTableSize, sizeof(int)));
}

void Velvet::HashObjects(
	uint* particleHash,
	uint* particleIndex,
	uint* cellStart,
	uint* neighbors,
	CONST(glm::vec3*) positions,
	const uint numObjects,
	const uint maxNumNeighbors)
{
	// TODO: merge the following three line
	uint numBlocks, numThreads;
	ComputeGridSize(numObjects, numBlocks, numThreads);
	ComputeParticleHash <<<numBlocks, numThreads >>> (particleHash, particleIndex, positions, numObjects);
	
	// Sort Particle Hash
	thrust::sort_by_key(thrust::device_ptr<uint>(particleHash),
		thrust::device_ptr<uint>(particleHash + numObjects),
		thrust::device_ptr<uint>(particleIndex));

	cudaMemset(cellStart, 0xffffffff, sizeof(uint) * (d_hashTableSize+1));
	ComputeGridSize(d_hashTableSize, numBlocks, numThreads);
	FindCellStart <<<numBlocks, numThreads >> > (cellStart, particleHash, numObjects);

	cudaMemset(neighbors, 0xffffffff, sizeof(uint)*maxNumNeighbors*numObjects);
	ComputeGridSize(numObjects, numBlocks, numThreads);
	CacheNeighbors << <numBlocks, numThreads >> > (neighbors, particleIndex, cellStart, positions, numObjects, maxNumNeighbors);
}

