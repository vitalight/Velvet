#include "MeshRenderer.hpp"

#include "GameInstance.hpp"
#include "Global.hpp"
#include "Camera.hpp"
#include "Actor.hpp"
#include "Light.hpp"
#include "Config.hpp"

namespace Velvet
{
	MeshRenderer::MeshRenderer(shared_ptr<Mesh> mesh, shared_ptr<Material> material)
		: m_mesh(mesh), m_material(material) 
	{
		name = __func__;
	}

	MeshRenderer::MeshRenderer(shared_ptr<Mesh> mesh, shared_ptr<Material> material, shared_ptr<Material> shadowMaterial)
		: m_mesh(mesh), m_material(material), m_shadowMaterial(shadowMaterial)
	{
		name = __func__;
	}

	// Only support spot light for now
	void MeshRenderer::SetupLighting(shared_ptr<Material> m_material)
	{
		auto light = Global::lights[0];

		auto prefix = fmt::format("spotLight.");
		auto front = Helper::RotateWithDegree(glm::vec3(0, -1, 0), light->transform()->rotation);

		m_material->SetVec3(prefix + "position", light->position());
		m_material->SetVec3(prefix + "direction", front);
		m_material->SetFloat(prefix + "cutOff", glm::cos(glm::radians(light->innerCutoff)));
		m_material->SetFloat(prefix + "outerCutOff", glm::cos(glm::radians(light->outerCutoff)));

		m_material->SetFloat(prefix + "constant", light->constant);
		m_material->SetFloat(prefix + "linear", light->linear);
		m_material->SetFloat(prefix + "quadratic", light->quadratic);

		m_material->SetVec3(prefix + "color", light->color);
		m_material->SetFloat(prefix + "ambient", light->ambient);
	}

	void MeshRenderer::Render(glm::mat4 lightMatrix)
	{
		if (hidden)
			return;

		m_material->Use();

		// material
		m_material->SetFloat("material.specular", m_material->specular);
		m_material->SetFloat("material.smoothness", m_material->smoothness);

		// camera param
		m_material->SetVec3("_CameraPos", Global::camera->transform()->position);

		// light params
		if (Global::lights.size() > 0)
		{
			SetupLighting(m_material);
		}

		// texture
		for (int i = 0; i < m_material->textures.size(); i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, m_material->textures[i]);
		}

		// matrices
		m_material->SetMat4("_Model", actor->transform->matrix());
		m_material->SetMat4("_View", Global::camera->view());
		m_material->SetMat4("_Projection", Global::camera->projection());
		m_material->SetMat4("_WorldToLight", lightMatrix);

		glBindVertexArray(m_mesh->VAO());
		if (m_mesh->useIndices())
		{
			glDrawElements(GL_TRIANGLES, m_mesh->drawCount(), GL_UNSIGNED_INT, 0);
		}
		else
		{
			glDrawArrays(GL_TRIANGLES, 0, m_mesh->drawCount());
		}
	}	

	void MeshRenderer::RenderShadow(glm::mat4 lightMatrix)
	{
		if (m_shadowMaterial == nullptr)
		{
			return;
		}

		m_shadowMaterial->Use();

		m_shadowMaterial->SetMat4("_Model", actor->transform->matrix());
		m_shadowMaterial->SetMat4("_WorldToLight", lightMatrix);

		glBindVertexArray(m_mesh->VAO());
		if (m_mesh->useIndices())
		{
			glDrawElements(GL_TRIANGLES, m_mesh->drawCount(), GL_UNSIGNED_INT, 0);
		}
		else
		{
			glDrawArrays(GL_TRIANGLES, 0, m_mesh->drawCount());
		}
	}

	shared_ptr<Material> MeshRenderer::material() const
	{
		return m_material;
	}
}