#pragma once

#include "Actor.hpp"
#include "Component.hpp"
#include "MeshRenderer.hpp"
#include "Global.hpp"
#include "GameInstance.hpp"

namespace Velvet
{
	class VtClothSolver : public Component
	{
	public:
		VtClothSolver(int resolution)
		{
			m_resolution = resolution;
		}

		void Start() override
		{
			fmt::print("Info(VtClothSolver): Start\n");
			auto renderer = actor->GetComponent<MeshRenderer>();
			m_mesh = renderer->mesh();
			m_vertices[READ] = m_mesh->vertices();
			m_vertices[WRITE] = vector<glm::vec3>(m_vertices[READ].size());
		}

		void Update() override
		{
			if (Global::game->pause)
			{
				return;
			}
			// for all vertices, update forces and positions
			for (int i = 0; i < m_vertices[READ].size(); i++)
			{
				m_vertices[WRITE][i] = m_vertices[READ][i] + Helper::RandomUnitVector() * k_fixedDeltaTime * 0.2f;
			}

			// fixed position constraint
			m_vertices[WRITE][0] = m_vertices[READ][0];
			m_vertices[WRITE][m_resolution] = m_vertices[READ][m_resolution ];

			m_mesh->SetVertices(m_vertices[WRITE]);
			WRITE = 1-WRITE;
			READ = 1-READ;
		}
	private:
		shared_ptr<Mesh> m_mesh;
		vector<glm::vec3> m_vertices[2];
		const float k_fixedDeltaTime = 0.02f;
		int m_resolution;
		int READ = 0, WRITE = 1;
	};
}