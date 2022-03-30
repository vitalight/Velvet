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

			m_positions[READ] = m_mesh->vertices();
			m_numVertices = m_positions[READ].size();
			m_translation = actor->transform->position;
			m_timeStep = Global::game->fixedDeltaTime;

			m_positions[WRITE] = vector<glm::vec3>(m_numVertices);
			m_velocities = vector<glm::vec3>(m_numVertices);
			m_predicted = vector<glm::vec3>(m_numVertices);

			GenerateDistConstraints();
		}

		void FixedUpdate() override
		{
			if (Global::game->pause)
			{
				return;
			}

			ApplyExternalForces();
			EstimatePositions();
			SolveConstraints();
			UpdatePositionsAndVelocities();
		}
	
	private:

		int READ = 0, WRITE = 1;

		float m_timeStep;
		int m_numVertices;
		glm::vec3 m_translation;
		shared_ptr<Mesh> m_mesh;
		vector<glm::vec3> m_positions[2];
		vector<glm::vec3> m_predicted;
		vector<glm::vec3> m_velocities;
		vector<tuple<int, int, float>> m_distanceConstraints; // idx1, idx2, distance
		int m_resolution;

		void GenerateDistConstraints()
		{
			auto VertexAt = [this](int x, int y) {
				return x * (m_resolution + 1) + y;
			};

			auto DistanceBetween = [this](int idx1, int idx2) {
				return glm::length(m_positions[READ][idx1] - m_positions[READ][idx2]);
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
	
		void ApplyExternalForces()
		{
			for (int i = 0; i < m_numVertices; i++)
			{
				// gravity
				m_velocities[i] += Global::Sim::gravity * m_timeStep;
				// damp
				m_velocities[i] *= (1 - Global::Sim::damping * m_timeStep);
			}
		}

		void EstimatePositions()
		{
			for (int i = 0; i < m_numVertices; i++)
			{
				m_predicted[i] = m_positions[READ][i] + m_velocities[i] * m_timeStep;
			}
		}

		void SolveConstraints()
		{
			// calculate force
			for (auto c : m_distanceConstraints)
			{
				auto idx1 = get<0>(c);
				auto idx2 = get<1>(c);
				auto expectedDistance = get<2>(c);

				glm::vec3 diff = m_predicted[idx1] - m_predicted[idx2];
				float distance = glm::length(diff);
				if (distance > expectedDistance)
				{
					auto direction = diff / distance;
					auto repulsion = 0.5f * (distance - expectedDistance) * direction * Global::Sim::stiffness;
					m_predicted[idx1] -= repulsion;
					m_predicted[idx2] += repulsion;
				}
			}

			// ground constraint
			for (int i = 0; i < m_numVertices; i++)
			{
				if (m_predicted[i].y + m_translation.y < 0)
				{
					m_predicted[i].y = -m_translation.y + 1e-5;
				}
			}

			// fixed position constraint
			m_predicted[0] = m_positions[READ][0];
			m_predicted[m_resolution] = m_positions[READ][m_resolution];

		}

		void UpdatePositionsAndVelocities()
		{
			// apply force and update positions
			for (int i = 0; i < m_numVertices; i++)
			{
				m_velocities[i] = m_predicted[i] - m_positions[READ][i];
				m_positions[WRITE][i] = m_predicted[i];
			}

			m_mesh->SetVertices(m_positions[WRITE]);
			WRITE = 1 - WRITE;
			READ = 1 - READ;
		}
	};
}