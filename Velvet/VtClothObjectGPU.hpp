#pragma once

#include "Component.hpp"
#include "VtClothSolverGPU.hpp"
#include "Actor.hpp"
#include "MeshRenderer.hpp"

namespace Velvet
{
	class VtClothObjectGPU : public Component
	{
	public:
		VtClothObjectGPU(int resolution)
		{
			SET_COMPONENT_NAME;
			m_solver = make_shared<VtClothSolverGPU>();
			m_resolution = resolution;
		}

		void SetAttachedIndices(vector<int> indices)
		{
			m_attachedIndices = indices;
		}

		auto particleDiameter() const
		{
			return m_particleDiameter;
		}

		auto solver() const
		{
			return m_solver;
		}

	public:
		void Start() override
		{
			auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
			auto transformMatrix = actor->transform->matrix();
			auto positions = mesh->vertices();
			m_particleDiameter = glm::length(positions[0] - positions[m_resolution + 1]) * 0.9f;

			m_solver->Initialize(mesh, transformMatrix, m_particleDiameter);
			actor->transform->Reset();

			ApplyTransform(positions, transformMatrix);
			GenerateStretch(positions);
			GenerateAttach(positions);

			m_colliders = Global::game->FindComponents<Collider>();
		}

		void FixedUpdate() override
		{
			//UpdateGrappedVertex();
			Timer::StartTimer("GPU_TIME");
			m_solver->UpdateColliders(m_colliders);
			m_solver->Simulate();
			Timer::EndTimer("GPU_TIME");
		}

		void OnDestroy() override
		{
			// some member variables need to be destructed earlier
			m_solver = nullptr;
		}

	private:
		int m_resolution;
		shared_ptr<VtClothSolverGPU> m_solver;
		vector<int> m_attachedIndices;
		vector<Collider*> m_colliders;
		float m_particleDiameter;

		void ApplyTransform(vector<glm::vec3>& positions, glm::mat4 transform)
		{
			for (int i = 0; i < positions.size(); i++)
			{
				positions[i] = transform * glm::vec4(positions[i], 1.0);
			}
		}

		void GenerateStretch(const vector<glm::vec3> &positions)
		{
			auto VertexAt = [this](int x, int y) {
				return x * (m_resolution + 1) + y;
			};
			auto DistanceBetween = [&positions](int idx1, int idx2) {
				return glm::length(positions[idx1] - positions[idx2]);
			};

			for (int x = 0; x < m_resolution + 1; x++)
			{
				for (int y = 0; y < m_resolution + 1; y++)
				{
					int idx1, idx2;

					if (y != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x, y + 1);
						m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));
					}

					if (x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y);
						m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));
					}

					if (y != m_resolution && x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y + 1);
						m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));

						idx1 = VertexAt(x, y + 1);
						idx2 = VertexAt(x + 1, y);
						m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));
					}
				}
			}
		}
	
		void GenerateAttach(const vector<glm::vec3>& positions)
		{
			for (auto idx : m_attachedIndices)
			{
				m_solver->AddAttach(idx, positions[idx]);
			}
		}
	};
}