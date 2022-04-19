#pragma once

#include "Actor.hpp"
#include "Component.hpp"
#include "MeshRenderer.hpp"
#include "Global.hpp"
#include "GameInstance.hpp"
#include "Collider.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "GUI.hpp"
#include "SpatialHashCPU.hpp"
#include "Timer.hpp"

namespace Velvet
{
	class VtClothSolverCPU
	{
	public:
		// SimBuffer Begin
		vector<glm::vec3> m_positions;
		vector<glm::vec3> m_predicted;
		vector<glm::vec3> m_velocities;
		vector<float> m_inverseMass;

		vector<tuple<int, int, float>> m_stretchConstraints; // idx1, idx2, distance
		vector<tuple<int, glm::vec3>> m_attachmentConstriants; // idx1, position
		vector<tuple<int, int, int, int, float>> m_bendingConstraints; // idx1, idx2, idx3, idx4, angle
		vector<tuple<int, int, int, int>> m_selfCollisionConstraints; // idx1, triangle(idx2, idx3, idx4)
		// SimBuffer End

		VtClothSolverCPU(int resolution)
		{
			m_resolution = resolution;
		}

		void SetAttachedIndices(vector<int> indices)
		{
			m_attachedIndices = indices;
		}

		void Initialize(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix)
		{
			fmt::print("Info(VtClothSolver): Start\n");
			m_mesh = mesh;

			m_positions = m_mesh->vertices();
			m_numVertices = (int)m_positions.size();
			for (int i = 0; i < m_numVertices; i++)
			{
				m_positions[i] = modelMatrix * glm::vec4(m_positions[i], 1.0f);
			}

			m_indices = m_mesh->indices();
			m_colliders = Global::game->FindComponents<Collider>();

			m_velocities = vector<glm::vec3>(m_numVertices);
			m_predicted = vector<glm::vec3>(m_numVertices);
			m_inverseMass = vector<float>(m_numVertices, 1.0);

			m_particleDiameter = glm::length(m_positions[0] - m_positions[m_resolution + 1]);
			m_spatialHash = make_shared<SpatialHashCPU>(m_particleDiameter, m_numVertices);

			GenerateStretch();
			GenerateAttachment(m_attachedIndices);
			GenerateBending();
		}

		void Simulate()
		{
			float frameTime = Timer::fixedDeltaTime();
			float substepTime = Timer::fixedDeltaTime() / Global::simParams.numSubsteps;

			// Pre-stablization pass [Unified particle physics for real-time applications (4.4)]
			CollideSDF(m_positions);

			PredictPositions(frameTime);
			m_spatialHash->HashObjects(m_predicted);

			for (int substep = 0; substep < Global::simParams.numSubsteps; substep++)
			{
				PredictPositions(substepTime);
				//GenerateSelfCollision();
				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{
					SolveStretch(substepTime);
					SolveBending(substepTime);

					//SolveSelfCollision();
					CollideParticles();
					CollideSDF(m_predicted);

					SolveAttachment();
				}
				Finalize(substepTime);
			}

			auto normals = ComputeNormals(m_positions);
			m_mesh->SetVerticesAndNormals(m_positions, normals);
		}

		float particleDiameter() const
		{
			return m_particleDiameter;
		}

	private: // Generate constraints

		void GenerateStretch()
		{
			auto VertexAt = [this](int x, int y) {
				return x * (m_resolution + 1) + y;
			};

			auto DistanceBetween = [this](int idx1, int idx2) {
				return glm::length(m_positions[idx1] - m_positions[idx2]);
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
						m_stretchConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));
					}

					if (x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y);
						m_stretchConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));
					}

					if (y != m_resolution && x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y + 1);
						m_stretchConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));

						idx1 = VertexAt(x, y + 1);
						idx2 = VertexAt(x + 1, y);
						m_stretchConstraints.push_back(make_tuple(idx1, idx2, DistanceBetween(idx1, idx2)));
					}
				}
			}
		}

		void GenerateAttachment(vector<int> indices)
		{
			for (auto i : indices)
			{
				m_attachmentConstriants.push_back({ i, m_positions[i]});
				m_inverseMass[i] = 0;
			}
		}

		void GenerateBending()
		{
			// HACK: not for every kind of mesh
			for (int i = 0; i < m_indices.size(); i += 6)
			{
				int idx1 = m_indices[i];
				int idx2 = m_indices[i + 1];
				int idx3 = m_indices[i + 2];
				int idx4 = m_indices[i + 5];

				// calculate angle
				float angle = 0;
				m_bendingConstraints.push_back(make_tuple(idx1, idx2, idx3, idx4, angle));
			}
		}

		void GenerateSelfCollision()
		{

		}

	private: // Core physics

		void PredictPositions(float deltaTime)
		{
			for (int i = 0; i < m_numVertices; i++)
			{
				m_velocities[i] += Global::simParams.gravity * deltaTime;
				m_predicted[i] = m_positions[i] + m_velocities[i] * deltaTime;
			}
		}

		void SolveStretch(float deltaTime)
		{
			for (auto c : m_stretchConstraints)
			{
				auto idx1 = get<0>(c);
				auto idx2 = get<1>(c);
				auto expectedDistance = get<2>(c);

				glm::vec3 diff = m_predicted[idx1] - m_predicted[idx2];
				float distance = glm::length(diff);
				auto w1 = m_inverseMass[idx1];
				auto w2 = m_inverseMass[idx2];

				// We use unilateral constraints instead of bilateral constraints
				// Otherwise the cloth may not look well after collision
				if (distance > expectedDistance && w1 + w2 > 0)
				{
					auto gradient = diff / (distance + k_epsilon);
					// compliance is zero, therefore XPBD=PBD
					auto denom = w1 + w2;
					auto lambda = (distance - expectedDistance) / denom;
					m_predicted[idx1] -= w1 * lambda * gradient;
					m_predicted[idx2] += w2 * lambda * gradient;
				}
			}
		}

		void SolveBending(float deltaTime)
		{
			float xpbd_bend = Global::simParams.bendCompliance / deltaTime / deltaTime;
			for (auto c : m_bendingConstraints)
			{
				// tri(idx1, idx3, idx2) and tri(idx1, idx2, idx4)
				auto idx1 = get<2>(c);
				auto idx2 = get<1>(c);
				auto idx3 = get<0>(c);
				auto idx4 = get<3>(c);
				auto expectedAngle = get<4>(c);

				auto w1 = m_inverseMass[idx1];
				auto w2 = m_inverseMass[idx2];
				auto w3 = m_inverseMass[idx3];
				auto w4 = m_inverseMass[idx4];

				auto p1 = m_predicted[idx1];
				auto p2 = m_predicted[idx2] - p1;
				auto p3 = m_predicted[idx3] - p1;
				auto p4 = m_predicted[idx4] - p1;

				glm::vec3 n1 = glm::normalize(glm::cross(p2, p3));
				glm::vec3 n2 = glm::normalize(glm::cross(p2, p4));

				float d = clamp(glm::dot(n1, n2), 0.0f, 1.0f);
				float angle = acos(d);
				// cross product for two equal vector produces NAN
				if (angle < k_epsilon || isnan(d)) continue;

				glm::vec3 q3 = (glm::cross(p2, n2) + glm::cross(n1, p2) * d) / (glm::length(glm::cross(p2, p3)) + k_epsilon);
				glm::vec3 q4 = (glm::cross(p2, n1) + glm::cross(n2, p2) * d) / (glm::length(glm::cross(p2, p4)) + k_epsilon);
				glm::vec3 q2 = -(glm::cross(p3, n2) + glm::cross(n1, p3) * d) / (glm::length(glm::cross(p2, p3)) + k_epsilon)
					- (glm::cross(p4, n1) + glm::cross(n2, p4) * d) / (glm::length(glm::cross(p2, p4)) + k_epsilon);
				glm::vec3 q1 = -q2 - q3 - q4;

				float denom = xpbd_bend + (w1 * glm::dot(q1, q1) + w2 * glm::dot(q2, q2) + w3 * glm::dot(q3, q3) + w4 * glm::dot(q4, q4));
				if (denom < k_epsilon) continue; // ?
				float lambda = sqrt(1.0f - d * d) * (angle - expectedAngle) / denom;

				//if (isnan(lambda) || glm::all(glm::isnan(q1)) || glm::all(glm::isnan(q2)) || glm::all(glm::isnan(q3)) || glm::all(glm::isnan(q4)))
				//{
				//	fmt::print("NAN detected\n");
				//}

				m_predicted[idx1] += w1 * lambda * q1;
				m_predicted[idx2] += w2 * lambda * q2;
				m_predicted[idx3] += w3 * lambda * q3;
				m_predicted[idx4] += w4 * lambda * q4;

			}
		}

		void CollideSDF(vector<glm::vec3>& positions) const
		{
			// SDF collision
			for (int i = 0; i < m_numVertices; i++)
			{
				for (auto col : m_colliders)
				{
					auto pos = positions[i];
					glm::vec3 correction = col->ComputeSDF(pos);
					positions[i] += correction;

					glm::vec3 relativeVelocity = positions[i] - m_positions[i];
					auto friction = ComputeFriction(correction, relativeVelocity);
					positions[i] += friction;
				}
			}
		}

		void SolveAttachment()
		{
			for (const auto& c : m_attachmentConstriants)
			{
				int idx = get<0>(c);
				glm::vec attachPos = get<1>(c);
				m_predicted[idx] = attachPos;
			}
		}

		void SolveSelfCollision()
		{
			for (const auto& c : m_selfCollisionConstraints)
			{
				auto [idx1, idx2, idx3, idx4] = c;
				auto q = m_predicted[idx1];
				auto p1 = m_predicted[idx2];
				auto p2 = m_predicted[idx3];
				auto p3 = m_predicted[idx4];

				float constraint = glm::dot(q - p1, glm::normalize(glm::cross(p2 - p1, p3 - p1))) - Global::simParams.collisionMargin;
			}
		}

		void CollideParticles()
		{
			for (int i = 0; i < m_numVertices; i++)
			{
				const auto neighbors = m_spatialHash->GetNeighbors(i);
				for (int j : neighbors)
				{
					if (i >= j) continue;
					auto idx1 = i;
					auto idx2 = j;
					auto expectedDistance = m_particleDiameter;

					glm::vec3 diff = m_predicted[idx1] - m_predicted[idx2];
					float distance = glm::length(diff);
					auto w1 = m_inverseMass[idx1];
					auto w2 = m_inverseMass[idx2];

					if (distance < expectedDistance && w1 + w2 > 0)
					{
						auto gradient = diff / (distance + k_epsilon);
						auto denom = w1 + w2;
						auto lambda = (distance - expectedDistance) / denom;
						auto common = lambda * gradient;
						m_predicted[idx1] -= w1 * common;
						m_predicted[idx2] += w2 * common;

						glm::vec3 relativeVelocity = (m_predicted[idx1] - m_positions[idx1]) - (m_predicted[idx2] - m_positions[idx2]);
						auto friction = ComputeFriction(common, relativeVelocity);
						m_predicted[idx1] += w1 * friction;
						m_predicted[idx2] -= w2 * friction;
					}
				}
			}
		}

		void Finalize(float deltaTime)
		{
			// apply force and update positions
			for (int i = 0; i < m_numVertices; i++)
			{
				//m_velocities[i] = (m_predicted[i] - m_positions[i]) / deltaTime;
				// damp
				m_velocities[i] = (m_predicted[i] - m_positions[i]) / deltaTime * (1 - Global::simParams.damping * deltaTime);
				m_positions[i] = m_predicted[i];
			}
		}

	private: // Utility functions

		glm::vec3 ComputeFriction(glm::vec3 correction, glm::vec3 relativeVelocity) const
		{
			glm::vec3 friction = glm::vec3(0);
			float correctionLength = glm::length(correction);
			if (Global::simParams.friction > 0 && correctionLength > 0)
			{
				glm::vec3 correctionNorm = correction / correctionLength;

				glm::vec3 tangentialVelocity = relativeVelocity - correctionNorm * glm::dot(relativeVelocity, correctionNorm);
				float tangentialLength = glm::length(tangentialVelocity);
				float maxTangential = correctionLength * Global::simParams.friction;

				friction = -tangentialVelocity * min(maxTangential / tangentialLength, 1.0f);
			}
			return friction;
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

		inline bool CheckNAN(const vector<glm::vec3> positions)
		{
			for (int i = 0; i < positions.size(); i++)
			{
				if (glm::any(glm::isnan(positions[i])))
				{
					fmt::print("NAN position detected\n");
					return true;
				}
			}
			return false;
		}

	private:

		const float k_epsilon = 1e-6f;

		int m_numVertices;
		int m_resolution;
		float m_particleDiameter;

		vector<unsigned int> m_indices;
		vector<Collider*> m_colliders;
		vector<int> m_attachedIndices;

		shared_ptr<Mesh> m_mesh;
		shared_ptr<SpatialHashCPU> m_spatialHash;
	};
}