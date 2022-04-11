#pragma once

#include "Common.cuh"

namespace Velvet
{
	void HashObjects(
		uint* particleHash,
		uint* particleIndex,
		uint* cellStart,
		uint* cellEnd,
		uint* neighbors,
		CONST(float3*) positions,
		const uint numObjects,
		const uint maxNumNeighbors,
		const float hashCellSpacing,
		const int hashTableSize);
}