#pragma once

#include "Component.hpp"
#include "VtClothSolverCPU.hpp"
#include "MeshRenderer.hpp"
#include "MouseGrabber.hpp"

namespace Velvet
{

	class VtClothObjectCPU : public Component
	{
	public:
		VtClothObjectCPU(int resolution)
		{
			SET_COMPONENT_NAME;
			m_solver = make_shared<VtClothSolverCPU>(resolution);
		}

		void SetAttachedIndices(vector<int> indices)
		{
			m_solver->SetAttachedIndices(indices);
		}

		void Start() override
		{
			auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
			m_solver->Initialize(mesh, actor->transform->matrix());
			actor->transform->Reset();
		}

		void Update() override
		{
			HandleMouseInteraction();
		}

		void FixedUpdate() override
		{
			UpdateGrappedVertex();
			m_solver->Simulate();
		}

		shared_ptr<VtClothSolverCPU> solver() const
		{
			return m_solver;
		}

	private:
		shared_ptr<VtClothSolverCPU> m_solver;

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
				Ray ray = GetMouseRay();
				glm::vec3 mousePos = ray.origin + ray.direction * m_rayCollision.distanceToOrigin;
				int id = m_rayCollision.objectIndex;
				auto curPos = m_solver->m_positions[id];
				glm::vec3 target = Helper::Lerp(mousePos, curPos, 0.8f);

				m_solver->m_positions[id] = target;
				m_solver->m_velocities[id] += (target - curPos) / Timer::fixedDeltaTime();
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