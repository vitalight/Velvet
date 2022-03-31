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

namespace Velvet
{
	struct Ray
	{
		glm::vec3 origin;
		glm::vec3 direction;
	};

	struct RaycastCollision
	{
		bool collide = false;
		int objectIndex;
		float distanceToOrigin;
	};

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

			m_positions = m_mesh->vertices();
			m_numVertices = m_positions.size();
			m_translation = actor->transform->position;
			//m_timeStep = Global::game->fixedDeltaTime / Global::Sim::numSubsteps;
			m_indices = m_mesh->indices();
			m_colliders = Global::game->FindComponents<Collider>();

			m_velocities = vector<glm::vec3>(m_numVertices);
			m_predicted = vector<glm::vec3>(m_numVertices);
			m_inverseMass = vector<float>(m_numVertices, 1.0);

			GenerateStretchConstraints();
			GenerateAttachmentConstraints();
			GenerateBendingConstraints();
		}

		void Update() override
		{
			HandleMouseInteraction();
		}

		void FixedUpdate() override
		{
			for (int substep = 0; substep < Global::Sim::numSubsteps; substep++)
			{
				ApplyExternalForces();
				EstimatePositions();
				for (int iteration = 0; iteration < Global::Sim::numIterations; iteration++)
				{
					SolveConstraints();
				}
				UpdatePositionsAndVelocities();
			}
			auto normals = ComputeNormals(m_positions);
			m_mesh->SetVerticesAndNormals(m_positions, normals);
		}
	
	private:

		float timeStep()
		{
			return Global::game->fixedDeltaTime / Global::Sim::numSubsteps;
		}

		void GenerateStretchConstraints()
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

		void GenerateAttachmentConstraints()
		{
			vector<int> indices = { 0, m_resolution };

			for (auto i : indices)
			{
				m_attachmentConstriants.push_back({ i, m_positions[i]});
				m_inverseMass[i] = 0;
			}
		}

		void GenerateBendingConstraints()
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

		void ApplyExternalForces()
		{
			for (int i = 0; i < m_numVertices; i++)
			{
				// gravity
				m_velocities[i] += Global::Sim::gravity * timeStep();
				// damp
				//m_velocities[i] *= (1 - Global::Sim::damping * m_timeStep);
			}
		}

		void EstimatePositions()
		{
			for (int i = 0; i < m_numVertices; i++)
			{
				m_predicted[i] = m_positions[i] + m_velocities[i] * timeStep();
			}
		}

		void SolveConstraints()
		{
			float xpbd_stretch = Global::Sim::stretchCompliance / timeStep() / timeStep();
			float epsilon = 1e-6;

			for (auto c : m_stretchConstraints)
			{
				auto idx1 = get<0>(c);
				auto idx2 = get<1>(c);
				auto expectedDistance = get<2>(c);

				glm::vec3 diff = m_predicted[idx1] - m_predicted[idx2];
				float distance = glm::length(diff);
				auto w1 = m_inverseMass[idx1];
				auto w2 = m_inverseMass[idx2];

				if (distance > expectedDistance && w1 + w2 > 0)
				{
					auto gradient = diff / distance;
					auto denom = w1 + w2 + xpbd_stretch;
					auto lambda = (distance - expectedDistance) / denom;
					m_predicted[idx1] -= w1 * lambda * gradient;
					m_predicted[idx2] += w2 * lambda * gradient;
				}
			}

			float xpbd_bend = Global::Sim::bendCompliance / timeStep() / timeStep();
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
				if (angle < epsilon) continue;

				glm::vec3 q3 = (glm::cross(p2, n2) + glm::cross(n1, p2) * d) / (glm::length(glm::cross(p2, p3)) + epsilon);
				glm::vec3 q4 = (glm::cross(p2, n1) + glm::cross(n2, p2) * d) / (glm::length(glm::cross(p2, p4)) + epsilon);
				glm::vec3 q2 = -(glm::cross(p3, n2) + glm::cross(n1, p3) * d) / (glm::length(glm::cross(p2, p3)) + epsilon)
					- (glm::cross(p4, n1) + glm::cross(n2, p4) * d) / (glm::length(glm::cross(p2, p4)) + epsilon);
				glm::vec3 q1 = -q2 - q3 - q4;

				float denom = xpbd_bend + (w1 * glm::dot(q1, q1) + w2 * glm::dot(q2, q2) + w3 * glm::dot(q3, q3) + w4 * glm::dot(q4, q4));
				if (denom < epsilon) continue; // ?
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

			// ground collision constraint
			for (int i = 0; i < m_numVertices; i++)
			{
				if (m_predicted[i].y + m_translation.y < 0)
				{
					m_predicted[i].y = -m_translation.y + 1e-2;
				}
			}

			// SDF collision
			for (int i = 0; i < m_numVertices; i++)
			{
				for (auto col : m_colliders)
				{
					auto pos = m_predicted[i] + m_translation;
					m_predicted[i] += col->ComputeSDF(pos);
				}
			}

			for (const auto& c : m_attachmentConstriants)
			{
				int idx = get<0>(c);
				glm::vec attachPos = get<1>(c);
				float mass = m_inverseMass[idx];
				m_predicted[idx] = attachPos;
			}

		}

		void UpdatePositionsAndVelocities()
		{
			// apply force and update positions
			for (int i = 0; i < m_numVertices; i++)
			{
				m_velocities[i] = (m_predicted[i] - m_positions[i]) / timeStep();
				m_positions[i] = m_predicted[i];
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

		void HandleMouseInteraction()
		{
			static bool isGrabbing = false;
			static int constraintIdx = 0;
			static RaycastCollision collision;

			bool shouldPickObject = Global::input->GetMouseDown(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldPickObject)
			{
				Ray ray = GetMouseRay();
				collision = FindClosestVertexToRay(ray);

				if (collision.collide)
				{
					isGrabbing = true;
					m_attachmentConstriants.push_back(make_tuple(collision.objectIndex, m_positions[collision.objectIndex]));
					constraintIdx = m_attachmentConstriants.size() - 1;
				}
			}

			if (!shouldPickObject && isGrabbing)
			{
				Ray ray = GetMouseRay();
				glm::vec3 target = ray.origin + ray.direction * collision.distanceToOrigin;

				auto& c = m_attachmentConstriants[constraintIdx];
				get<1>(c) = target - m_translation;
			}

			bool shouldReleaseObject = Global::input->GetMouseUp(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldReleaseObject && isGrabbing)
			{
				isGrabbing = false;
				m_attachmentConstriants.pop_back();
			}
		}

		Ray GetMouseRay()
		{
			glm::vec2 screenPos = Global::input->GetMousePos();
			// [0, 1]
			auto normalizedScreenPos = 2.0f * screenPos / glm::vec2(Global::Config::screenWidth, Global::Config::screenHeight) - 1.0f;
			normalizedScreenPos.y = -normalizedScreenPos.y;

			glm::mat4 invVP = glm::inverse(Global::camera->projection() * Global::camera->view());
			glm::vec4 nearPointRaw = invVP * glm::vec4(normalizedScreenPos, 0, 1);
			glm::vec4 farPointRaw = invVP * glm::vec4(normalizedScreenPos, 1, 1);

			glm::vec3 nearPoint = glm::vec3(nearPointRaw.x, nearPointRaw.y, nearPointRaw.z) / nearPointRaw.w;
			glm::vec3 farPoint = glm::vec3(farPointRaw.x, farPointRaw.y, farPointRaw.z) / farPointRaw.w;
			glm::vec3 direction = glm::normalize(farPoint - nearPoint);

			return Ray{ nearPoint, direction };
		}

		RaycastCollision FindClosestVertexToRay(Ray ray)
		{
			int result = -1;
			float minDistanceToRay = FLT_MAX;
			float distanceToView = 0;
			for (int i = 0; i < m_positions.size(); i++)
			{
				// TODO: translation
				const auto& position = m_positions[i] + m_translation;
				float distanceToRay = glm::length(glm::cross(ray.direction, position - ray.origin));
				if (distanceToRay < minDistanceToRay)
				{
					result = i;
					minDistanceToRay = distanceToRay;
					distanceToView = glm::dot(ray.direction, position - ray.origin);
				}
			}
			return RaycastCollision{ minDistanceToRay < 0.2, result, distanceToView };
		}

	private:

		int m_numVertices;
		int m_resolution;
		glm::vec3 m_translation;
		shared_ptr<Mesh> m_mesh;

		vector<glm::vec3> m_positions;
		vector<unsigned int> m_indices;
		vector<glm::vec3> m_predicted;
		vector<glm::vec3> m_velocities;
		vector<float> m_inverseMass;
		vector<Collider*> m_colliders;

		vector<tuple<int, int, float>> m_stretchConstraints; // idx1, idx2, distance
		vector<tuple<int, glm::vec3>> m_attachmentConstriants; // idx1, position
		vector<tuple<int, int, int, int, float>> m_bendingConstraints; // idx1, idx2, idx3, idx4, angle

	};
}