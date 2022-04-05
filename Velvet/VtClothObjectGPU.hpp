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
			m_solver->Finalize();
		}

	private:
		shared_ptr<VtClothSolverGPU> m_solver;
	};
}