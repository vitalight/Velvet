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

		void Update() override
		{
			HandleMouseInteraction();
		}

		void FixedUpdate() override
		{
			UpdateGrappedVertex();

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
	
	private:

		bool m_isGrabbing = false;
		float m_grabbedVertexMass = 0;
		RaycastCollision m_rayCollision;

		void HandleMouseInteraction()
		{
			bool shouldPickObject = Global::input->GetMouseDown(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldPickObject)
			{
				Ray ray = GetMouseRay();
				m_rayCollision = FindClosestVertexToRay(ray);

				if (m_rayCollision.collide)
				{
					m_isGrabbing = true;
					m_grabbedVertexMass = m_solver->m_inverseMass[m_rayCollision.objectIndex];
					m_solver->m_inverseMass[m_rayCollision.objectIndex] = 0;
				}
			}

			bool shouldReleaseObject = Global::input->GetMouseUp(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldReleaseObject && m_isGrabbing)
			{
				m_isGrabbing = false;
				m_solver->m_inverseMass[m_rayCollision.objectIndex] = m_grabbedVertexMass;
			}
		}

		RaycastCollision FindClosestVertexToRay(Ray ray)
		{
			int result = -1;
			float minDistanceToRay = FLT_MAX;
			float distanceToView = 0;

			m_solver->m_positions.pull();

			for (int i = 0; i < m_solver->m_positions.size(); i++)
			{
				const auto& position = m_solver->m_positions[i];
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

		void UpdateGrappedVertex()
		{
			if (m_isGrabbing)
			{
				m_solver->m_positions.pull();

				Ray ray = GetMouseRay();
				glm::vec3 mousePos = ray.origin + ray.direction * m_rayCollision.distanceToOrigin;
				int id = m_rayCollision.objectIndex;
				auto curPos = m_solver->m_positions[id];
				glm::vec3 target = Helper::Lerp(mousePos, curPos, 0.8f);

				m_solver->m_positions[id] = target;
				m_solver->m_velocities[id] += (target - curPos) / Timer::fixedDeltaTime();

				m_solver->m_positions.push();
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
	};
}