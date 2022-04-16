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
	const float k_hashCellSizeScalar = 1.5f;

	class VtClothSolverGPU
	{
	public:

		void Initialize(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix, float particleDiameter)
		{
			Timer::StartTimer("INIT_SOLVER_GPU");

			int numParticles = (int)mesh->vertices().size();
			Global::simParams.numParticles = numParticles;
			Global::simParams.particleDiameter = particleDiameter;
			Global::simParams.deltaTime = Timer::fixedDeltaTime();

			positions.registerBuffer(mesh->verticesVBO());
			normals.registerBuffer(mesh->normalsVBO());

			indices.wrap(mesh->indices());
			velocities.resize(numParticles, glm::vec3(0));
			predicted.resize(numParticles, glm::vec3(0));
			positionDeltas.resize(numParticles, glm::vec3(0));
			positionDeltaCount.resize(numParticles, 0);
			inverseMass.resize(numParticles, 1.0f);

			InitializePositions(positions, numParticles, modelMatrix);

			m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter * k_hashCellSizeScalar, numParticles);;

			double time = Timer::EndTimer("INIT_SOLVER_GPU") * 1000;
			//ShowDebugGUI();
			fmt::print("Info(ClothSolverGPU): Initialize done. Took time {:.2f} ms\n", time);
			fmt::print("Info(ClothSolverGPU): Recommond max vel = {}\n", 2 * particleDiameter / Timer::fixedDeltaTime());
		}

		void Simulate()
		{
			Timer::StartTimerGPU("Solver_Total");
			//==========================
			// Prepare
			//==========================
			float frameTime = Timer::fixedDeltaTime();
			float substepTime = Timer::fixedDeltaTime() / Global::simParams.numSubsteps;

			//==========================
			// Launch kernel
			//==========================
			SetSimulationParams(&Global::simParams);

			CollideSDF((uint)sdfColliders.size(), sdfColliders, positions, positions);

			EstimatePositions(positions, predicted, velocities, frameTime);
			m_spatialHash->Hash(predicted);

			for (int substep = 0; substep < Global::simParams.numSubsteps; substep++)
			{
				EstimatePositions(positions, predicted, velocities, substepTime);

				if (Global::simParams.enableSelfCollision)
				{
					CollideParticles(inverseMass, m_spatialHash->neighbors, positions, predicted);
				}
				CollideSDF((uint)sdfColliders.size(), sdfColliders, positions, predicted);

				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{
					SolveStretch((uint)stretchLengths.size(), stretchIndices, stretchLengths, inverseMass, predicted,
						positionDeltas, positionDeltaCount);
					//SolveBending(predicted, positionDeltas, positionDeltaCount, bendIndices, bendAngles, inverseMass, (uint)bendAngles.size(), substepTime);
					SolveAttachment((uint)attachIndices.size(), attachIndices, attachPositions, predicted);
				}
				UpdatePositionsAndVelocities(predicted, velocities, positions, substepTime);
			}

			// UpdateNormal
			ComputeNormal((uint)(indices.size() / 3), positions, indices, normals);

			//==========================
			// Sync
			//==========================
			Timer::EndTimerGPU("Solver_Total");
			cudaDeviceSynchronize();
		}

	public:

		void AddStretch(int idx1, int idx2, float distance)
		{
			stretchIndices.push_back(idx1);
			stretchIndices.push_back(idx2);
			stretchLengths.push_back(distance);
		}

		void AddAttach(int index, glm::vec3 position)
		{
			inverseMass[index] = 0;
			attachIndices.push_back(index);
			attachPositions.push_back(position);
		}

		void AddBend(uint idx1, uint idx2, uint idx3, uint idx4, float angle)
		{
			bendIndices.push_back(idx1);
			bendIndices.push_back(idx2);
			bendIndices.push_back(idx3);
			bendIndices.push_back(idx4);
			bendAngles.push_back(angle);
		}

		void UpdateColliders(vector<Collider*>& colliders)
		{
			sdfColliders.resize(colliders.size());

			for (int i = 0; i < colliders.size(); i++)
			{
				const Collider* c = colliders[i];
				SDFCollider sc;
				sc.position = c->actor->transform->position;
				sc.scale = c->actor->transform->scale;
				sc.type = c->sphereOrPlane ? SDFCollider::SDFColliderType::Plane : SDFCollider::SDFColliderType::Sphere;
				sdfColliders[i] = sc;
			}
		}

	public: // Sim buffers

		VtRegisteredBuffer<glm::vec3> positions;
		VtRegisteredBuffer<glm::vec3> normals;
		VtBuffer<uint> indices;

		VtBuffer<glm::vec3> velocities;
		VtBuffer<glm::vec3> predicted;
		VtBuffer<glm::vec3> positionDeltas;
		VtBuffer<int> positionDeltaCount;
		VtBuffer<float> inverseMass;

		VtBuffer<int> stretchIndices;
		VtBuffer<float> stretchLengths;
		VtBuffer<uint> bendIndices;
		VtBuffer<float> bendAngles;
		VtBuffer<int> attachIndices;
		VtBuffer<glm::vec3> attachPositions;
		VtBuffer<SDFCollider> sdfColliders;

	private:

		shared_ptr<SpatialHashGPU> m_spatialHash;

		void ShowDebugGUI()
		{
			GUI::RegisterDebug([this]() {
				{
					static int particleIndex1 = 0;
					//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID1", &particleIndex1, 0, Global::simParams.numParticles - 1);
					ImGui::Indent(10);
					ImGui::Text(fmt::format("Position: {}", predicted[particleIndex1]).c_str());
					auto hash3i = m_spatialHash->HashPosition3i(predicted[particleIndex1]);
					auto hash = m_spatialHash->HashPosition(predicted[particleIndex1]);
					ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());

					static int neighborRange1 = 0;
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange1", &neighborRange1, 0, 63);
					ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange1 + particleIndex1 * Global::simParams.maxNumNeighbors]).c_str());
					ImGui::Indent(-10);
				}

				{
					static int particleIndex2 = 0;
					//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID2", &particleIndex2, 0, Global::simParams.numParticles - 1);
					ImGui::Indent(10);
					ImGui::Text(fmt::format("Position: {}", predicted[particleIndex2]).c_str());
					auto hash3i = m_spatialHash->HashPosition3i(predicted[particleIndex2]);
					auto hash = m_spatialHash->HashPosition(predicted[particleIndex2]);
					ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());

					static int neighborRange2 = 0;
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange2", &neighborRange2, 0, 63);
					ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange2 + particleIndex2 * Global::simParams.maxNumNeighbors]).c_str());
					ImGui::Indent(-10);
				}
				static int cellID = 0;
				IMGUI_LEFT_LABEL(ImGui::SliderInt, "CellID", &cellID, 0, (int)m_spatialHash->cellStart.size() - 1);
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

	};
}