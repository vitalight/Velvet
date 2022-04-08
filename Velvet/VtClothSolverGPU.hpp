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

			m_positions.registerBuffer(mesh->verticesVBO());
			m_normals.registerBuffer(mesh->normalsVBO());

			m_indices.wrap(mesh->indices());
			m_velocities.resize(m_numParticles, glm::vec3(0));
			m_predicted.resize(m_numParticles, glm::vec3(0));
			m_positionDeltas.resize(m_numParticles, glm::vec3(0));
			m_positionDeltaCount.resize(m_numParticles, 0);
			m_inverseMass.resize(m_numParticles, 1.0f);

			InitializePositions(m_positions, m_numParticles, modelMatrix);

			m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter, m_numParticles);;

			if (0)
			{
				GUI::RegisterDebug([this]() {
					{
						static int particleIndex1 = 0;
						//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
						IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID1", &particleIndex1, 0, m_numParticles - 1);
						ImGui::Indent(10);
						ImGui::Text(fmt::format("Position: {}", m_predicted[particleIndex1]).c_str());
						auto hash3i = m_spatialHash->HashPosition3i(m_predicted[particleIndex1]);
						auto hash = m_spatialHash->HashPosition(m_predicted[particleIndex1]);
						ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());

						static int neighborRange1 = 0;
						IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange1", &neighborRange1, 0, 63);
						ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange1 + particleIndex1 * Global::simParams.maxNumNeighbors]).c_str());
						ImGui::Indent(-10);
					}

					{
						static int particleIndex2 = 0;
						//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
						IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID2", &particleIndex2, 0, m_numParticles - 1);
						ImGui::Indent(10);
						ImGui::Text(fmt::format("Position: {}", m_predicted[particleIndex2]).c_str());
						auto hash3i = m_spatialHash->HashPosition3i(m_predicted[particleIndex2]);
						auto hash = m_spatialHash->HashPosition(m_predicted[particleIndex2]);
						ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());

						static int neighborRange2 = 0;
						IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange2", &neighborRange2, 0, 63);
						ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange2 + particleIndex2 * Global::simParams.maxNumNeighbors]).c_str());
						ImGui::Indent(-10);
					}
					static int cellID = 0;
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "CellID", &cellID, 0, m_spatialHash->cellStart.size() - 1);
					int start = m_spatialHash->cellStart[cellID];
					int end = m_spatialHash->cellEnd[cellID];
					ImGui::Indent(10);
					ImGui::Text(fmt::format("CellStart.HashID: {}", start).c_str());
					ImGui::Text(fmt::format("CellEnd.HashID: {}", end).c_str());

					if (start != 0xffffffff && end > start)
					{
						static int particleHash = 0;
						particleHash = clamp(particleHash, start, end - 1);
						IMGUI_LEFT_LABEL(ImGui::SliderInt, "HashID", &particleHash, start, end - 1);
						ImGui::Text(fmt::format("ParticleHash: {}", m_spatialHash->particleHash[particleHash]).c_str());
						ImGui::Text(fmt::format("ParticleIndex: {}", m_spatialHash->particleIndex[particleHash]).c_str());
					}
					});
			}
		}

		void Simulate()
		{
			//==========================
			// prepare
			//==========================
			m_params.gravity = Global::simParams.gravity;
			m_params.numParticles = m_numParticles;
			m_params.damping = Global::simParams.damping;
			m_params.collisionMargin = Global::simParams.collisionMargin;
			m_params.maxNumNeighbors = Global::simParams.maxNumNeighbors;
			m_params.friction = Global::simParams.friction;

			float frameTime = Global::game->fixedDeltaTime;
			float substepTime = Global::game->fixedDeltaTime / Global::simParams.numSubsteps;

			//==========================
			// map OpenGL buffer object for writing from CUDA
			//==========================
			//m_positions.Map();

			//==========================
			// launch kernel
			//==========================
			SetSimulationParams(&m_params);

			SolveSDFCollision(m_SDFColliders.size(), m_SDFColliders, m_positions, m_positions);

			EstimatePositions(m_positions, m_predicted, m_velocities, frameTime);
			m_spatialHash->Hash(m_predicted);

			for (int substep = 0; substep < Global::simParams.numSubsteps; substep++)
			{
				EstimatePositions(m_positions, m_predicted, m_velocities, substepTime);

				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{
					SolveStretch(m_stretchLengths.size(), m_stretchIndices, m_stretchLengths, m_inverseMass, m_predicted, m_positionDeltas, m_positionDeltaCount);

					SolveParticleCollision(m_inverseMass, m_spatialHash->neighbors, m_positions, m_predicted, m_positionDeltas, m_positionDeltaCount);
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
			m_inverseMass[index] = 0;
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

		// TODO: wrap with SimBuffer class
		VtRegisteredBuffer<glm::vec3> m_positions;
		VtRegisteredBuffer<glm::vec3> m_normals;
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
	private:

		VtSimParams m_params;
		uint m_numParticles;
		float m_particleDiameter;
		shared_ptr<SpatialHashGPU> m_spatialHash;

	};
}