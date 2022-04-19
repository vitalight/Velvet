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
#include "MouseGrabber.hpp"

using namespace std;

namespace Velvet
{
	class VtClothSolverGPU : public Component
	{
	public:

		void Start() override
		{
			Global::simParams.numParticles = 0;
			m_colliders = Global::game->FindComponents<Collider>();
			m_mouseGrabber.Initialize(&positions, &velocities, &invMasses);
			//ShowDebugGUI();
		}

		void Update() override
		{
			m_mouseGrabber.HandleMouseInteraction();
		}

		void FixedUpdate() override
		{
			m_mouseGrabber.UpdateGrappedVertex();
			UpdateColliders(m_colliders);

			Timer::StartTimer("GPU_TIME");
			Simulate();
			Timer::EndTimer("GPU_TIME");
		}

		void OnDestroy() override
		{
			positions.destroy();
			normals.destroy();
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

			// External colliders can move relatively fast, and cloth will have large velocity after colliding with them.
			// This can produce unstable behavior, such as vertex flashing between two sides.
			// We include a pre-stabilization step to mitigate this issue. Collision here will not influence velocity.
			CollideSDF(positions, sdfColliders, positions, (uint)sdfColliders.size(), frameTime);

			for (int substep = 0; substep < Global::simParams.numSubsteps; substep++)
			{
				PredictPositions(predicted, velocities, positions, substepTime);

				if (Global::simParams.enableSelfCollision)
				{
					if (substep % Global::simParams.interleavedHash == 0)
					{
						m_spatialHash->Hash(predicted);
					}
					CollideParticles(deltas, deltaCounts, predicted, invMasses, m_spatialHash->neighbors, positions);
				}
				CollideSDF(predicted, sdfColliders, positions, (uint)sdfColliders.size(), substepTime);

				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{
					SolveStretch(predicted, deltas, deltaCounts, stretchIndices, stretchLengths, invMasses, (uint)stretchLengths.size());
					SolveAttachment(predicted, deltas, deltaCounts, invMasses,
						attachParticleIDs, attachSlotIDs, attachSlotPositions, attachDistances, (uint)attachParticleIDs.size());
					//SolveBending(predicted, deltas, deltaCounts, bendIndices, bendAngles, invMasses, (uint)bendAngles.size(), substepTime);
					ApplyDeltas(predicted, deltas, deltaCounts);
				}

				Finalize(velocities, positions, predicted, substepTime);
			}

			ComputeNormal(normals, positions, indices, (uint)(indices.size() / 3));

			//==========================
			// Sync
			//==========================
			Timer::EndTimerGPU("Solver_Total");
			cudaDeviceSynchronize();

			positions.sync();
			normals.sync();
		}
	public:

		int AddCloth(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix, float particleDiameter)
		{
			Timer::StartTimer("INIT_SOLVER_GPU");

			int prevNumParticles = Global::simParams.numParticles;
			int newParticles = (int)mesh->vertices().size();

			// Set global parameters
			Global::simParams.numParticles += newParticles;
			Global::simParams.particleDiameter = particleDiameter;
			Global::simParams.deltaTime = Timer::fixedDeltaTime();
			Global::simParams.maxSpeed = 2 * particleDiameter / Timer::fixedDeltaTime() * Global::simParams.numSubsteps;

			// Allocate managed buffers
			positions.registerNewBuffer(mesh->verticesVBO());
			normals.registerNewBuffer(mesh->normalsVBO());

			for (int i = 0; i < mesh->indices().size(); i++)
			{
				indices.push_back(mesh->indices()[i] + prevNumParticles);
			}

			velocities.push_back(newParticles, glm::vec3(0));
			predicted.push_back(newParticles, glm::vec3(0));
			deltas.push_back(newParticles, glm::vec3(0));
			deltaCounts.push_back(newParticles, 0);
			invMasses.push_back(newParticles, 1.0f);

			// Initialize buffer datas
			InitializePositions(positions, prevNumParticles, newParticles, modelMatrix);
			cudaDeviceSynchronize();
			positions.sync();

			// Initialize member variables
			m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter, Global::simParams.numParticles);
			m_spatialHash->SetInitialPositions(positions);

			double time = Timer::EndTimer("INIT_SOLVER_GPU") * 1000;
			fmt::print("Info(ClothSolverGPU): AddCloth done. Took time {:.2f} ms\n", time);
			fmt::print("Info(ClothSolverGPU): Use recommond max vel = {}\n", Global::simParams.maxSpeed);

			return prevNumParticles;
		}

		void AddStretch(int idx1, int idx2, float distance)
		{
			stretchIndices.push_back(idx1);
			stretchIndices.push_back(idx2);
			stretchLengths.push_back(distance);
		}

		void AddAttachSlot(glm::vec3 attachSlotPos)
		{
			attachSlotPositions.push_back(attachSlotPos);
		}

		void AddAttach(int particleIndex, int slotIndex, float distance)
		{
			if (distance == 0) invMasses[particleIndex] = 0;
			attachParticleIDs.push_back(particleIndex);
			attachSlotIDs.push_back(slotIndex);
			attachDistances.push_back(distance);
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
				if (!c->enabled) continue;
				SDFCollider sc;
				sc.type = c->type;
				sc.position = c->actor->transform->position;
				sc.scale = c->actor->transform->scale;
				sc.curTransform = c->curTransform;
				sc.invCurTransform = glm::inverse(c->curTransform);
				sc.lastTransform = c->lastTransform;
				sc.deltaTime = Timer::fixedDeltaTime();
				sdfColliders[i] = sc;
			}
		}

	public: // Sim buffers

		VtMergedBuffer<glm::vec3> positions;
		VtMergedBuffer<glm::vec3> normals;
		VtBuffer<uint> indices;

		VtBuffer<glm::vec3> velocities;
		VtBuffer<glm::vec3> predicted;
		VtBuffer<glm::vec3> deltas;
		VtBuffer<int> deltaCounts;
		VtBuffer<float> invMasses;

		VtBuffer<int> stretchIndices;
		VtBuffer<float> stretchLengths;
		VtBuffer<uint> bendIndices;
		VtBuffer<float> bendAngles;

		// Attach attachParticleIndices[i] with attachSlotIndices[i] w
		// where their expected distance is attachDistances[i]
		VtBuffer<int> attachParticleIDs;
		VtBuffer<int> attachSlotIDs;
		VtBuffer<float> attachDistances;
		VtBuffer<glm::vec3> attachSlotPositions;

		VtBuffer<SDFCollider> sdfColliders;

	private:

		shared_ptr<SpatialHashGPU> m_spatialHash;
		vector<Collider*> m_colliders;
		MouseGrabber m_mouseGrabber;

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
					auto norm = normals[particleIndex1];
					ImGui::Text(fmt::format("Normal: [{:.3f},{:.3f},{:.3f}]", norm.x, norm.y, norm.z).c_str());

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