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
			m_timeStep = Global::game->fixedDeltaTime / Global::Sim::numSubsteps;
			m_indices = m_mesh->indices();

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

			for (int substep = 0; substep < Global::Sim::numSubsteps; substep++)
			{
				ApplyExternalForces();
				EstimatePositions();
				for (int iteration = 0; iteration < Global::Sim::numIterations; iteration++)
				{
					SolveConstraints();
				}
				UpdatePositionsAndVelocities();
				WRITE = 1 - WRITE;
				READ = 1 - READ;
			}
			auto normals = ComputeNormals(m_positions[WRITE]);
			m_mesh->SetVerticesAndNormals(m_positions[WRITE], normals);
		}
	
	private:

		int READ = 0, WRITE = 1;

		float m_timeStep;
		int m_numVertices;
		glm::vec3 m_translation;
		shared_ptr<Mesh> m_mesh;
		vector<glm::vec3> m_positions[2];
		vector<unsigned int> m_indices;
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
					m_predicted[i].y = -m_translation.y + 1e-2;
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
		}

		vector<glm::vec3> ComputeNormals(const vector<glm::vec3> positions)
		{
			vector<glm::vec3> normals(positions.size());
			for (int i = 0; i < m_indices.size(); i += 3)
			{
				auto idx1 = m_indices[i];
				auto idx2 = m_indices[i + 1];
				auto idx3 = m_indices[i + 2];

				auto p1 = positions[idx1];
				auto p2 = positions[idx2];
				auto p3 = positions[idx3];

				auto normal = glm::cross(p2 - p1, p3 - p1);
				normals[idx1] += normal;
				normals[idx2] += normal;
				normals[idx3] += normal;
			}
			for (int i = 0; i < normals.size(); i++)
			{
				normals[i] = glm::normalize(normals[i]);
			}
			return normals;
		}
	};
}