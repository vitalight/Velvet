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
		int numDirLight = 0;
		int numSpotLight = 0;
		const int maxLightPerType = 3;
		// lighting
		for (int i = 0; i < Global::light.size(); i++)
		{
			auto light = Global::light[i];

			if (light->type == LightType::Point)
			{
				if (numPointLight > maxLightPerType)
				{
					fmt::print("Max light per type is [{}]. Ignore extra point light\n", maxLightPerType);
					continue;
				}
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
				if (numDirLight > maxLightPerType)
				{
					fmt::print("Max light per type is [{}]. Ignore extra dir light\n", maxLightPerType);
					continue;
				}
				auto prefix = fmt::format("dirLights[{}].", numDirLight);
				m_material.SetVec3(prefix + "direction", -light->position());
				m_material.SetVec3(prefix + "ambient", 0.05f, 0.05f, 0.05f);
				m_material.SetVec3(prefix + "diffuse", 0.4f, 0.4f, 0.4f);
				m_material.SetVec3(prefix + "specular", 0.5f, 0.5f, 0.5f);
				numDirLight++;
			}
			else
			{
				if (numSpotLight > maxLightPerType)
				{
					fmt::print("Max light per type is [{}]. Ignore extra point light\n", maxLightPerType);
					continue;
				}
				auto prefix = fmt::format("spotLights[{}].", numSpotLight);
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
				numSpotLight++;
			}
		}
		m_material.SetInt("numSpotLight", numSpotLight);
		m_material.SetInt("numDirLight", numDirLight);
		m_material.SetInt("numPointLight", numPointLight);
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