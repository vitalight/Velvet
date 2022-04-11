#pragma once

#include <unordered_set>

#include <glm/glm.hpp>

#include "VtBuffer.hpp"
#include "Global.hpp"
#include "SpatialhashGPU.cuh"

using namespace std;

namespace Velvet
{
	class SpatialHashGPU
	{
	public://debug
		inline int ComputeIntCoord(float value)
		{
			return (int)floor(value / m_spacing);
		}

		inline int HashCoords(int x, int y, int z)
		{
			int h = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);	// fantasy function
			return abs(h % m_tableSize);
		}

		inline glm::ivec3 HashPosition3i(glm::vec3 position)
		{
			int x = ComputeIntCoord(position.x);
			int y = ComputeIntCoord(position.y);
			int z = ComputeIntCoord(position.z);
			return glm::ivec3(x, y, z);
		}

		inline int HashPosition(glm::vec3 position)
		{
			int x = ComputeIntCoord(position.x);
			int y = ComputeIntCoord(position.y);
			int z = ComputeIntCoord(position.z);

			int h = HashCoords(x, y, z);
			return h;
		}
	public:
		SpatialHashGPU(float spacing, int maxNumObjects)
		{
			// BUG_LOG: m_spacing was miswritten as int
			m_spacing = spacing;
			m_tableSize = 2 * maxNumObjects;

			neighbors.resize(maxNumObjects * Global::simParams.maxNumNeighbors);
			particleHash.resize(maxNumObjects);
			particleIndex.resize(maxNumObjects);
			cellStart.resize(m_tableSize);
			cellEnd.resize(m_tableSize);
		}

		void Hash(const VtBuffer<glm::vec3>& positions)
		{
			HashObjects(particleHash, particleIndex, cellStart, cellEnd, neighbors, positions, (uint)positions.size(), 
				Global::simParams.maxNumNeighbors, m_spacing, m_tableSize);
		}

		VtBuffer<uint> neighbors;

		VtBuffer<uint> particleHash;
		VtBuffer<uint> particleIndex;
		VtBuffer<uint> cellStart;
		VtBuffer<uint> cellEnd; // BUG_LOG: early optimization (cpu differs from gpu)
	private:
		//VtBuffer<uint> cellEnd;

		float m_spacing;
		int m_tableSize;

		void Test(const VtBuffer<glm::vec3>& positions)
		{
			cudaDeviceSynchronize();

			vector<unordered_set<uint>> neighbors_truth(positions.size());
			vector<unordered_set<uint>> neighbors_gpu(positions.size());
			const float errorMargin = 0.999f;

			for (int i = 0; i < positions.size(); i++)
			{
				for (int j = 0; j < positions.size(); j++)
				{
					float distance = glm::length(positions[i] - positions[j]);
					if (distance < m_spacing && i != j)
					{
						neighbors_truth[i].insert(j);
					}
				}

				for (int k = 0; k < 64; k++)
				{
					uint neighbor = neighbors[i * Global::simParams.maxNumNeighbors + k];
					if (neighbor != 0xffffffff)
					{
						neighbors_gpu[i].insert(neighbor);
					}
				}

				vector<int> falsePositive, falseNegative;
				vector<float> dist1, dist2;
				for (auto val : neighbors_truth[i])
				{
					auto dist = glm::length(positions[i] - positions[val]);
					if (neighbors_gpu[i].count(val) == 0 && dist < errorMargin * m_spacing)
					{
						falseNegative.push_back(val);
						dist1.push_back(dist);
					}
				}
				for (auto val : neighbors_gpu[i])
				{
					auto dist = glm::length(positions[i] - positions[val]);
					if (neighbors_truth[i].count(val) == 0 && dist < errorMargin * m_spacing)
					{
						falsePositive.push_back(val);
						dist2.push_back(dist);
					}
				}

				if (falsePositive.size() + falseNegative.size() > 0)
				{
					fmt::print("id=[{}]\n\t falseNegative[{}]\n\t dist1[{}]\n\t falsePositive[{}]\n\t dist2[{}]\n",
						i, fmt::join(falseNegative, ", "), fmt::join(dist1, ", "),
						fmt::join(falsePositive, ", "), fmt::join(dist2, ", "));
				}
			}
		}

	};
}