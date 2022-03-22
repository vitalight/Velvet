#include "MeshRenderer.hpp"

#include "VtGraphics.hpp"
#include "Global.hpp"
#include "Camera.hpp"
#include "Actor.hpp"
#include "Light.hpp"

namespace Velvet
{
	MeshRenderer::MeshRenderer(Mesh mesh, Material material)
		: m_mesh(mesh), m_material(material) 
	{
		name = __func__;
	}

	MeshRenderer::MeshRenderer(Mesh mesh, Material material, Material shadowMaterial)
		: m_mesh(mesh), m_material(material), m_shadowMaterial(shadowMaterial)
	{
		name = __func__;
	}

	void SetupLighting(Material m_material)
	{
		int numPointLight = 0;
		// lighting
		for (int i = 0; i < Global::light.size(); i++)
		{
			auto light = Global::light[i];

			if (light->type == LightType::Point)
			{
				auto prefix = fmt::format("pointLights[{}].", numPointLight);
				m_material.SetVec3(prefix + "position", light->position());
				m_material.SetVec3(prefix + "ambient", 0.05f, 0.05f, 0.05f);
				m_material.SetVec3(prefix + "diffuse", 0.8f, 0.8f, 0.8f);
				m_material.SetVec3(prefix + "specular", 1.0f, 1.0f, 1.0f);
				m_material.SetFloat(prefix + "constant", 1.0f);
				m_material.SetFloat(prefix + "linear", 0.09f);
				m_material.SetFloat(prefix + "quadratic", 0.032f);
				numPointLight++;
			}
			else if (light->type == LightType::Directional)
			{
				m_material.SetVec3("dirLight.direction", -0.2f, -1.0f, -0.3f); // light->position()
				m_material.SetVec3("dirLight.ambient", 0.05f, 0.05f, 0.05f);
				m_material.SetVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
				m_material.SetVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);
			}
			else
			{
				m_material.SetVec3("spotLight.position", Global::camera->position());
				m_material.SetVec3("spotLight.direction", Global::camera->front());
				m_material.SetVec3("spotLight.ambient", 0.0f, 0.0f, 0.0f);
				m_material.SetVec3("spotLight.diffuse", 1.0f, 1.0f, 1.0f);
				m_material.SetVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
				m_material.SetFloat("spotLight.constant", 1.0f);
				m_material.SetFloat("spotLight.linear", 0.09f);
				m_material.SetFloat("spotLight.quadratic", 0.032f);
				m_material.SetFloat("spotLight.cutOff", glm::cos(glm::radians(12.5f)));
				m_material.SetFloat("spotLight.outerCutOff", glm::cos(glm::radians(15.0f)));
			}
		}
	}

	void MeshRenderer::Render(glm::mat4 lightMatrix)
	{
		if (hidden)
			return;

		m_material.Use();
		SetupLighting(m_material);

		// camera
		m_material.SetVec3("viewPos", Global::camera->transform()->position);
		if (Global::light.size() > 0)
		{
			m_material.SetVec3("lightPos", Global::light[0]->position());
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_material.texture1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, m_material.texture2);

		glm::mat4 model = actor->transform->matrix();
		glm::mat4 view = Global::camera->view();
		glm::mat4 projection = glm::perspective(glm::radians(Global::camera->zoom), 800.0f / 600.0f, 0.1f,
			100.0f);

		m_material.SetMat4("model", model);
		m_material.SetMat4("view", view);
		m_material.SetMat4("projection", projection);
		m_material.SetMat4("lightSpaceMatrix", lightMatrix);

		glBindVertexArray(m_mesh.VAO());
		if (m_mesh.useIndices())
		{
			glDrawElements(GL_TRIANGLES, m_mesh.numVertices(), GL_UNSIGNED_INT, 0);
		}
		else
		{
			glDrawArrays(GL_TRIANGLES, 0, m_mesh.numVertices());
		}
	}	

	void MeshRenderer::RenderShadow(glm::mat4 lightMatrix)
	{
		if (m_shadowMaterial.shaderID() == -1)
		{
			return;
		}

		m_shadowMaterial.Use();
		m_shadowMaterial.SetMat4("lightSpaceMatrix", lightMatrix);

		glm::mat4 model = actor->transform->matrix();
		m_shadowMaterial.SetMat4("model", model);

		glBindVertexArray(m_mesh.VAO());
		if (m_mesh.useIndices())
		{
			glDrawElements(GL_TRIANGLES, m_mesh.numVertices(), GL_UNSIGNED_INT, 0);
		}
		else
		{
			glDrawArrays(GL_TRIANGLES, 0, m_mesh.numVertices());
		}
	}

	Material MeshRenderer::material() const
	{
		return m_material;
	}
}