#pragma once

#include "MeshRenderer.hpp"
#include "Resource.hpp"
#include "VtClothObject.hpp"

namespace Velvet
{
	class ParticleRenderer : public MeshRenderer
	{
	public:

		ParticleRenderer() : MeshRenderer(
			Resource::LoadMesh("sphere.obj"),
			Resource::LoadMaterial("_InstancedDefault"))
			//Resource::LoadMaterial("_InstancedShadowDepth"))
		{
			SET_COMPONENT_NAME;

			m_instanceVBO = m_mesh->AllocateVBO(3, true);

			m_material->SetVec3("material.tint", glm::vec3(0.2, 0.3, 0.6));
			m_material->specular = 0.0f;
			m_material->SetBool("material.useTexture", false);
			m_material->SetTexture("_ShadowTex", Global::game->depthFrameBuffer());
		}

		void Start() override
		{
			m_cloth = actor->GetComponent<VtClothObject>();
		}

		void Update() override
		{
			if (!m_cloth)
			{
				//fmt::print("Warning(ParticleRenderer): No cloth found\n");
				return;
			}
			vector<glm::vec3> &translations = m_cloth->solver()->m_positions;
			m_numInstances = (int)translations.size();

			glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_numInstances, &translations[0], GL_DYNAMIC_DRAW);
		}

		void Render(glm::mat4 lightMatrix) override
		{
			if (!Global::Sim::drawParticles)
			{
				return;
			}
			m_material->Use();
			m_material->SetFloat("_ParticleRadius", m_cloth->solver()->particleDiameter() * 0.5f);
			MeshRenderer::Render(lightMatrix);
			return;
		}

		void RenderShadow(glm::mat4 lightMatrix) override
		{
			if (!Global::Sim::drawParticles)
			{
				return;
			}
			if (m_shadowMaterial == nullptr)
			{
				return;
			}

			m_shadowMaterial->Use();
			m_shadowMaterial->SetFloat("_ParticleRadius", m_cloth->solver()->particleDiameter() * 0.5f);
			MeshRenderer::RenderShadow(lightMatrix);
			return;
		}
	private:
		unsigned int m_instanceVBO;
		VtClothObject* m_cloth;
	};
}