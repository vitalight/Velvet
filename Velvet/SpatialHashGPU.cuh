#pragma once

#include "Common.cuh"

namespace Velvet
{
	void SetHashParams(float hashCellSpacing, int hashTableSize);

	void HashObjects(
		uint* particleHash,
		uint* particleIndex,
		uint* cellStart,
		uint* neighbors,
		CONST(glm::vec3*) positions,
		const uint numObjects,
		const uint maxNumNeighbors);
}