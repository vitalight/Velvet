#pragma once

#include <tuple>

#include <fmt/format.h>
#include <glm/glm.hpp>

#include <helper_cuda.h>
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/sort.h>

#define CONST(type)				const type const
#define GET_CUDA_ID(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= maxID) return
#define GET_CUDA_ID_NO_RETURN(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x
#define EPSILON					1e-6f

namespace Velvet
{
	typedef unsigned int uint;

	const uint BLOCK_SIZE = 256;

	struct SimulationParams
	{
		uint numParticles;
		glm::vec3 gravity;

		float damping;
		float friction;
		float collisionMargin;
		float particleDiameter;

		//float hashCellSpacing;
		//uint hashTableSize;
		int maxNumNeighbors;
	};

	inline void ComputeGridSize(const uint& n, uint& numBlocks, uint& numThreads)
	{
		if (n == 0)
		{
			//fmt::print("Error(Solver): numParticles is 0\n");
			numBlocks = 0;
			numThreads = 0;
			return;
		}
		numThreads = min(n, BLOCK_SIZE);
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
	}

	template<class T>
	inline T* VtAllocBuffer(int elementCount)
	{
		T* devPtr = nullptr;
		checkCudaErrors(cudaMallocManaged((void**)&devPtr, elementCount * sizeof(T)));
		cudaDeviceSynchronize(); // this is necessary, otherwise realloc can cause crash
		return devPtr;
	}

	inline void VtFreeBuffer(void* buffer)
	{
		checkCudaErrors(cudaFree(buffer));
	}
}