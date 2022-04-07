#pragma once

#include <glm/glm.hpp>

#include "VtBuffer.hpp"
#include "Global.hpp"
#include "SpatialhashGPU.cuh"

namespace Velvet
{
	class SpatialHashGPU
	{
	public:
		SpatialHashGPU(float spacing, int maxNumObjects)
		{
			int tableSize = 2 * maxNumObjects;

			neighbors.resize(maxNumObjects * Global::Sim::maxNumNeighbors);
			particleHash.resize(maxNumObjects);
			particleIndex.resize(maxNumObjects);
			cellStart.resize(tableSize + 1);
			//cellEnd.resize(m_tableSize + 1, 0);
			
			//SetHashParams(spacing, tableSize);
		}

		void Hash(const VtBuffer<glm::vec3>& positions)
		{
			HashObjects(particleHash, particleIndex, cellStart, neighbors, positions, positions.size(), Global::Sim::maxNumNeighbors);
		}

		VtBuffer<uint> neighbors;

	private:
		VtBuffer<uint> particleHash; 
		VtBuffer<uint> particleIndex;
		VtBuffer<uint> cellStart;
		//VtBuffer<uint> cellEnd;
	};
}