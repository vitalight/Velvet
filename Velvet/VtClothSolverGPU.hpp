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
#include "SpatialHashGPU.hpp"

using namespace std;

namespace Velvet
{
	class VtClothSolverGPU
	{
	public:

		void Initialize(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix, float particleDiameter)
		{
			m_numParticles = (int)mesh->vertices().size();
			m_params.particleDiameter = particleDiameter;

			m_positions.RegisterBuffer(mesh->verticesVBO());
			m_normals.RegisterBuffer(mesh->normalsVBO());
			m_indices.Wrap(mesh->indices());

			m_velocities.resize(m_numParticles, glm::vec3(0));
			m_predicted.resize(m_numParticles, glm::vec3(0));
			m_positionDeltas.resize(m_numParticles, glm::vec3(0));
			m_positionDeltaCount.resize(m_numParticles, 0);
			m_inverseMass.resize(m_numParticles, 1.0f);

			InitializePositions(m_positions, m_numParticles, modelMatrix);

			//m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter, m_numParticles);;

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
			m_params.collisionMargin = Global::Sim::collisionMargin;
			m_params.maxNumNeighbors = Global::Sim::maxNumNeighbors;

			float frameTime = Global::game->fixedDeltaTime;
			float substepTime = Global::game->fixedDeltaTime / Global::Sim::numSubsteps;

			//==========================
			// map OpenGL buffer object for writing from CUDA
			//==========================
			//m_positions.Map();

			//==========================
			// launch kernel
			//==========================
			SetSimulationParams(&m_params);

			SolveSDFCollision(m_SDFColliders.size(), m_SDFColliders, m_positions, m_positions);

			for (int substep = 0; substep < Global::Sim::numSubsteps; substep++)
			{
				EstimatePositions(m_positions, m_predicted, m_velocities, substepTime);
				//m_spatialHash->Hash(m_predicted);

				for (int iteration = 0; iteration < Global::Sim::numIterations; iteration++)
				{
					SolveStretch(m_stretchLengths.size(), m_stretchIndices, m_stretchLengths, m_inverseMass, m_predicted, m_positionDeltas, m_positionDeltaCount);

					//SolveParticleCollision(m_inverseMass, m_spatialHash->neighbors, m_predicted, m_positionDeltas, m_positionDeltaCount);
					SolveSDFCollision(m_SDFColliders.size(), m_SDFColliders, m_positions, m_predicted);

					SolveAttachment(m_attachIndices.size(), m_attachIndices, m_attachPositions, m_predicted);
				}
				UpdatePositionsAndVelocities(m_predicted, m_velocities, m_positions, substepTime);
			}

			// UpdateNormal
			ComputeNormal(m_indices.size() / 3, m_positions, m_indices, m_normals);
			//==========================
			// unmap buffer object
			//==========================
			//m_positions.Unmap();
			cudaDeviceSynchronize();
		}

	public:
		void AddStretch(int idx1, int idx2, float distance)
		{
			m_stretchIndices.push_back(idx1);
			m_stretchIndices.push_back(idx2);
			m_stretchLengths.push_back(distance);
		}

		void AddAttach(int index, glm::vec3 position)
		{
			m_attachIndices.push_back(index);
			m_attachPositions.push_back(position);
		}

		void UpdateColliders(vector<Collider*>& colliders)
		{
			m_SDFColliders.resize(colliders.size());

			for (int i = 0; i < colliders.size(); i++)
			{
				const Collider* c = colliders[i];
				SDFCollider sc;
				sc.position = c->actor->transform->position;
				sc.scale = c->actor->transform->scale;
				sc.type = c->sphereOrPlane ? SDFCollider::SDFColliderType::Plane : SDFCollider::SDFColliderType::Sphere;
				m_SDFColliders[i] = sc;
			}
		}
	private:

		SimulationParams m_params;
		uint m_numParticles;
		float m_particleDiameter;
		shared_ptr<SpatialHashGPU> m_spatialHash;

		// TODO: wrap with SimBuffer class
		VtBuffer<glm::vec3> m_positions;
		VtBuffer<glm::vec3> m_normals;
		VtBuffer<uint> m_indices;

		VtBuffer<glm::vec3> m_velocities;
		VtBuffer<glm::vec3> m_predicted;
		VtBuffer<glm::vec3> m_positionDeltas;
		VtBuffer<int> m_positionDeltaCount;
		VtBuffer<float> m_inverseMass;

		VtBuffer<int> m_stretchIndices;
		VtBuffer<float> m_stretchLengths;
		VtBuffer<int> m_attachIndices;
		VtBuffer<glm::vec3> m_attachPositions;
		VtBuffer<SDFCollider> m_SDFColliders;
	};
}