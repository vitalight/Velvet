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
			m_velocities.resize(m_numParticles, glm::vec3(0));
			m_predicted.resize(m_numParticles);
			m_inverseMass.resize(m_numParticles, 1.0f);

			m_positions.Map();
			InitializePositions(m_positions.data(), m_numParticles, modelMatrix);
			m_positions.Unmap();

			GUI::RegisterDebug([this]() {
				static int debugIndex = 0;
				ImGui::SliderInt("index", &debugIndex, 0, m_numParticles-1);
				ImGui::Text(fmt::format("Position: {}", m_predicted[debugIndex]).c_str());
				});
		}

		void Simulate()
		{
			//==========================
			// prepare
			//==========================
			m_params.gravity = Global::Sim::gravity;
			m_params.numParticles = m_numParticles;
			m_params.damping = Global::Sim::damping;

			float frameTime = Global::game->fixedDeltaTime;
			float substepTime = Global::game->fixedDeltaTime / Global::Sim::numSubsteps;

			//==========================
			// map OpenGL buffer object for writing from CUDA
			//==========================
			m_positions.Map();

			//==========================
			// launch kernel
			//==========================
			SetSimulationParams(&m_params);
			EstimatePositions(m_positions, m_predicted, m_velocities, frameTime);
			//SolveStretch(m_predicted.data(), m_stretchIndices.data(), m_stretchLengths.data(), m_inverseMass.data(), m_stretchLengths.size());
			UpdatePositionsAndVelocities(m_predicted, m_velocities, m_positions, frameTime);

			//==========================
			// unmap buffer object
			//==========================
			m_positions.Unmap();
			cudaDeviceSynchronize();
		}

	public:
		void AddStretch(int idx1, int idx2)
		{
			auto DistanceBetween = [this](int idx1, int idx2) {
				return glm::length(m_positions[idx1] - m_positions[idx2]);
			};
			m_stretchIndices.push_back(idx1);
			m_stretchIndices.push_back(idx2);
			m_stretchLengths.push_back(DistanceBetween(idx1, idx2));
		}
	private:

		SimulationParams m_params;
		uint m_numParticles;

		VtBuffer<glm::vec3> m_positions;
		VtBuffer<glm::vec3> m_velocities;
		VtBuffer<glm::vec3> m_predicted;

		VtBuffer<int> m_stretchIndices;
		VtBuffer<float> m_stretchLengths;
		VtBuffer<float> m_inverseMass;

	};
}