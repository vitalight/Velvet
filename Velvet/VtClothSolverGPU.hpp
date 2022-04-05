#pragma once

#include <iostream>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "helper_cuda.h"
#include "Mesh.hpp"
#include "VtClothSolverGPU.cuh"
#include "VtBuffer.hpp"

using namespace std;

namespace Velvet
{
	class VtClothSolverGPU
	{
	public:

		void Initialize(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix)
		{
			m_numParticles = (int)mesh->vertices().size();

			m_positions.RegisterBuffer(mesh->verticesVBO());
			m_velocities.Resize(m_numParticles);

			m_positions.Map();
			InitializePositions(m_positions.raw(), m_numParticles, modelMatrix);
			m_positions.Unmap();
		}

		void Simulate()
		{
			// map OpenGL buffer object for writing from CUDA
			m_positions.Map();

			// launch kernel
			m_params.gravity = Global::Sim::gravity;
			m_params.numParticles = m_numParticles;
			m_params.deltaTime = Global::game->deltaTime;

			SetSimulationParams(&m_params);
			ApplyExternalForces(m_positions.raw(), m_velocities.raw(), m_numParticles);

			// unmap buffer object
			m_positions.Unmap();
		}

	private:

		SimulationParams m_params;
		uint m_numParticles;

		VtBuffer<glm::vec3> m_positions;
		VtBuffer<glm::vec3> m_velocities;
	};
}