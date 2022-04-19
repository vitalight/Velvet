#pragma once

#include "MeshRenderer.hpp"

namespace Velvet
{
	// Render particles using geometry shader
	class ParticleGeometryRenderer : public MeshRenderer
	{
	public:
		ParticleGeometryRenderer() : MeshRenderer(
			nullptr,
			Resource::LoadMaterial("_InstancedParticle", true))
		{
			m_material->doubleSided = true;
			m_material->SetVec3("material.tint", glm::vec3(0.2, 0.3, 0.6));
			m_material->specular = 0.0f;
			m_material->SetBool("material.useTexture", false);
		}

		void Start() override
		{
			m_cloth = actor->GetComponent<VtClothObjectGPU>();
			m_mesh = CustomMesh();
		}

		//void Update() override
		//{
		//	if (!m_cloth)
		//	{
		//		return;
		//	}
		//	auto clothMesh = actor->GetComponent<MeshRenderer>()->mesh();
		//	auto verticesVBO = clothMesh->verticesVBO();

		//	glBindBuffer(GL_COPY_READ_BUFFER, verticesVBO);
		//	glBindBuffer(GL_COPY_WRITE_BUFFER, m_mesh->verticesVBO());
		//	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(glm::vec3) * m_numInstances);
		//}

		shared_ptr<Mesh> CustomMesh()
		{
			// placeholder mesh
			vector<glm::vec3> points = {
				glm::vec3(0), 
			};
			auto mesh = make_shared<Mesh>(points);
			glBindVertexArray(mesh->VAO());
			auto clothMesh = actor->GetComponent<MeshRenderer>()->mesh();
			m_numParticles = (int)clothMesh->vertices().size();
			glBindBuffer(GL_ARRAY_BUFFER, clothMesh->verticesVBO());
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
			glBindVertexArray(0);

			return mesh;
		}

		void DrawCall() override
		{
			if (!Global::gameState.drawParticles)
			{
				return;
			}
			m_material->Use();
			m_material->SetFloat("_ParticleRadius", m_cloth->particleDiameter() * 0.5f);
			glBindVertexArray(m_mesh->VAO());
			glDrawArrays(GL_POINTS, 0, m_numParticles);
		}
	private:
		VtClothObjectGPU* m_cloth;
		int m_numParticles;
	};
}