#pragma once

#include "MeshRenderer.hpp"
#include "Resource.hpp"
#include "VtClothObjectCPU.hpp"

namespace Velvet
{
	// Render particles using instanced rendering
	class ParticleInstancedRenderer : public MeshRenderer
	{
	public:

		ParticleInstancedRenderer() : MeshRenderer(
			Resource::LoadMesh("sphere.obj"),
			Resource::LoadMaterial("_InstancedDefault"))
			//Resource::LoadMaterial("_InstancedShadowDepth"))
		{
			SET_COMPONENT_NAME;

			m_instanceVBO = m_mesh->AllocateVBO(3, true);

			m_material->SetVec3("material.tint", glm::vec3(0.2, 0.3, 0.6));
			m_material->specular = 0.0f;
			m_material->SetBool("material.useTexture", false);
		}

		void Start() override
		{
			m_cloth = actor->GetComponent<VtClothObjectGPU>();

			auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
			auto particles = mesh->verticesVBO();
			m_numInstances = (int)mesh->vertices().size();

			glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_numInstances, nullptr, GL_DYNAMIC_DRAW);
		}

		void Update() override
		{
			if (!m_cloth)
			{
				//fmt::print("Warning(ParticleRenderer): No cloth found\n");
				return;
			}
			auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
			auto verticesVBO = mesh->verticesVBO();

			glBindBuffer(GL_COPY_READ_BUFFER, verticesVBO);
			glBindBuffer(GL_COPY_WRITE_BUFFER, m_instanceVBO);
			glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(glm::vec3) * m_numInstances);
		}

		void Render(glm::mat4 lightMatrix) override
		{
			if (!Global::gameState.drawParticles)
			{
				return;
			}
			m_material->Use();
			m_material->SetFloat("_ParticleRadius", m_cloth->particleDiameter() * 0.5f);
			MeshRenderer::Render(lightMatrix);
			return;
		}

		void RenderShadow(glm::mat4 lightMatrix) override
		{
			if (!Global::gameState.drawParticles)
			{
				return;
			}
			if (m_shadowMaterial == nullptr)
			{
				return;
			}

			m_shadowMaterial->Use();
			m_shadowMaterial->SetFloat("_ParticleRadius", m_cloth->particleDiameter() * 0.5f);
			MeshRenderer::RenderShadow(lightMatrix);
			return;
		}
	private:
		unsigned int m_instanceVBO;
		VtClothObjectGPU* m_cloth;
	};
}