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
			m_numVertices = m_vertices[READ].size();
			m_vertices[WRITE] = vector<glm::vec3>(m_numVertices);
			m_accelerations = vector<glm::vec3>(m_numVertices);
			m_translation = actor->transform->position;

			GenerateDistConstraints();
		}

		void FixedUpdate() override
		{
			if (Global::game->pause)
			{
				return;
			}
			// calculate force
			for (int i = 0; i < m_numVertices; i++)
			{
				m_accelerations[i] = Global::Sim::gravity;
			}
			for (auto c : m_distanceConstraints)
			{
				auto idx1 = get<0>(c);
				auto idx2 = get<1>(c);
				auto expectedDistance = get<2>(c);

				glm::vec3 diff = m_vertices[READ][idx1] - m_vertices[READ][idx2];
				float distance = glm::length(diff);
				if (distance > expectedDistance)
				{
					auto direction = diff / distance;
					auto repulsion = (distance - expectedDistance) * direction * Global::Sim::stiffness;
					m_accelerations[idx1] -= repulsion;
					m_accelerations[idx2] += repulsion;
				}
			}
			// apply force and update positions
			for (int i = 0; i < m_numVertices; i++)
			{
				//m_vertices[WRITE][i] = m_vertices[READ][i] + Helper::RandomUnitVector() * k_fixedDeltaTime * 0.2f;
				m_vertices[WRITE][i] = m_vertices[READ][i] + m_accelerations[i] * Global::game->fixedDeltaTime;
			}

			// ground constraint
			for (int i = 0; i < m_numVertices; i++)
			{
				if (m_vertices[WRITE][i].y + m_translation.y < 0)
				{
					m_vertices[WRITE][i].y = -m_translation.y + 1e-5;
				}
			}

			// fixed position constraint
			m_vertices[WRITE][0] = m_vertices[READ][0];
			m_vertices[WRITE][m_resolution] = m_vertices[READ][m_resolution ];

			m_mesh->SetVertices(m_vertices[WRITE]);
			WRITE = 1-WRITE;
			READ = 1-READ;
		}
	
	private:

		int READ = 0, WRITE = 1;

		int m_numVertices;
		glm::vec3 m_translation;
		shared_ptr<Mesh> m_mesh;
		vector<glm::vec3> m_vertices[2];
		vector<glm::vec3> m_accelerations;
		vector<tuple<int, int, float>> m_distanceConstraints; // idx1, idx2, distance
		int m_resolution;

		void GenerateDistConstraints()
		{
			auto VertexAt = [this](int x, int y) {
				return x * (m_resolution + 1) + y;
			};

			auto DistanceBetween = [this](int idx1, int idx2) {
				return glm::length(m_vertices[READ][idx1] - m_vertices[READ][idx2]);
			};

			for (int x = 0; x < m_resolution+1; x++)
			{
				for (int y = 0; y < m_resolution+1; y++)
				{
					int idx1, idx2;

					if (y != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x, y + 1);
						m_distanceConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));
					}

					if (x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y);
						m_distanceConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));
					}

					if (y != m_resolution && x!=m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y + 1);
						m_distanceConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));

						idx1 = VertexAt(x, y + 1);
						idx2 = VertexAt(x + 1, y);
						m_distanceConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));
					}
				}
			}
		}
	};
}