#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "helper_cuda.h"

typedef unsigned int uint;

namespace Velvet
{
	void AllocateArray(void** devPtr, size_t size);

	void FreeArray(void* devPtr);

	void InitializePositions(glm::vec3* positions, int count, glm::mat4 modelMatrix);

	void ApplyExternalForces(glm::vec3* positions, glm::vec3* velocities, uint count);
}
