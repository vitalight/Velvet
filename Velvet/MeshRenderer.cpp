#include "MeshRenderer.hpp"

#include "VtGraphics.hpp"
#include "Global.hpp"
#include "Camera.hpp"
#include "Actor.hpp"
#include "Light.hpp"
#include "Config.hpp"

namespace Velvet
{
	MeshRenderer::MeshRenderer(Mesh &mesh, Material &material)
		: m_mesh(mesh), m_material(material) 
	{
		name = __func__;
	}

	MeshRenderer::MeshRenderer(Mesh &mesh, Material &material, Material &shadowMaterial)
		: m_mesh(mesh), m_material(material), m_shadowMaterial(shadowMaterial)
	{
		name = __func__;
	}

	// Only support spot light for now
	void MeshRenderer::SetupLighting(Material m_material)
	{
		auto light = Global::light[0];

		auto prefix = fmt::format("spotLight.");
		auto front = Helper::RotateWithDegree(glm::vec3(0,-1,0), light->transform()->rotation);
		m_material.SetVec3(prefix+"position", light->position());
		m_material.SetVec3(prefix+"direction", front);
		m_material.SetVec3(prefix+"ambient", 0.5f, 0.5f, 0.5f);
		m_material.SetVec3(prefix+"diffuse", 1.0f, 1.0f, 1.0f);
		m_material.SetVec3(prefix+"specular", 1.0f, 1.0f, 1.0f);
		m_material.SetFloat(prefix+"constant", 1.0f);
		m_material.SetFloat(prefix+"linear", 0.09f);
		m_material.SetFloat(prefix+"quadratic", 0.032f);
		m_material.SetFloat(prefix+"cutOff", glm::cos(glm::radians(45.0f)));
		m_material.SetFloat(prefix+"outerCutOff", glm::cos(glm::radians(60.0f)));
	}

	void MeshRenderer::Render(glm::mat4 lightMatrix)
	{
		if (hidden)
			return;

		m_material.Use();

		// camera
		m_material.SetVec3("_CameraPos", Global::camera->transform()->position);

		// light
		if (Global::light.size() > 0)
		{
			m_material.SetVec3("_LightPos", Global::light[0]->position());
			SetupLighting(m_material);
		}

		for (int i = 0; i < m_material.textures.size(); i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, m_material.textures[i]);
		}

		glm::mat4 model = actor->transform->matrix();
		glm::mat4 view = Global::camera->view();
		glm::mat4 projection = Global::camera->projection();

		m_material.SetMat4("_Model", model);
		m_material.SetMat4("_View", view);
		m_material.SetMat4("_Projection", projection);
		m_material.SetMat4("_WorldToLight", lightMatrix);

		glBindVertexArray(m_mesh.VAO());
		if (m_mesh.useIndices())
		{
			glDrawElements(GL_TRIANGLES, m_mesh.drawCount(), GL_UNSIGNED_INT, 0);
		}
		else
		{
			glDrawArrays(GL_TRIANGLES, 0, m_mesh.drawCount());
		}
	}	

	void MeshRenderer::RenderShadow(glm::mat4 lightMatrix)
	{
		if (m_shadowMaterial.shaderID() == -1)
		{
			return;
		}

		m_shadowMaterial.Use();

		m_shadowMaterial.SetMat4("_Model", actor->transform->matrix());
		m_shadowMaterial.SetMat4("_WorldToLight", lightMatrix);

		glBindVertexArray(m_mesh.VAO());
		if (m_mesh.useIndices())
		{
			glDrawElements(GL_TRIANGLES, m_mesh.drawCount(), GL_UNSIGNED_INT, 0);
		}
		else
		{
			glDrawArrays(GL_TRIANGLES, 0, m_mesh.drawCount());
		}
	}

	Material MeshRenderer::material() const
	{
		return m_material;
	}
}