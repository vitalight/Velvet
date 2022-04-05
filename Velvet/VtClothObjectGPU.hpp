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

		void Start() override
		{
			auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
			m_solver->Initialize(mesh, actor->transform->matrix());
			actor->transform->Reset();
		}

		void FixedUpdate() override
		{
			//UpdateGrappedVertex();
			m_solver->Simulate();
		}

		void OnDestroy() override
		{
			// some member variables need to be destructed earlier
			m_solver = nullptr;
		}

	private:
		shared_ptr<VtClothSolverGPU> m_solver;
		int m_resolution;

		void GenerateStretch()
		{
			auto VertexAt = [this](int x, int y) {
				return x * (m_resolution + 1) + y;
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
						m_solver->AddStretch(idx1, idx2);
					}

					if (x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y);
						m_solver->AddStretch(idx1, idx2);
					}

					if (y != m_resolution && x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y + 1);
						m_solver->AddStretch(idx1, idx2);

						idx1 = VertexAt(x, y + 1);
						idx2 = VertexAt(x + 1, y);
						m_solver->AddStretch(idx1, idx2);
					}
				}
			}
		}
	};
}