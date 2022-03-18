#include "MeshRenderer.hpp"

#include "VtGraphics.hpp"
#include "Global.hpp"
#include "Camera.hpp"
#include "Actor.hpp"
#include "Light.hpp"

namespace Velvet
{
	MeshRenderer::MeshRenderer()
	{
		name = __func__;
	}
	MeshRenderer::MeshRenderer(Mesh mesh, Material material)
		: m_mesh(mesh), m_material(material) 
	{
		name = __func__;
	}

	void MeshRenderer::Update()
	{
		// draw triangles
		m_material.Use();
		
		m_material.SetVec4("light.position", Global::light->position());
		m_material.SetVec3("light.direction", Global::camera->front());
		m_material.SetFloat("light.cutOff", glm::cos(glm::radians(12.5f)));
		m_material.SetFloat("light.outerCutOff", glm::cos(glm::radians(17.5f)));
		m_material.SetVec3("viewPos", Global::camera->transform()->position);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_material.texture1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, m_material.texture2);

		glm::mat4 model = actor->transform->matrix();
		glm::mat4 view = Global::camera->view();
		glm::mat4 projection = glm::perspective(glm::radians(Global::camera->zoom), 800.0f / 600.0f, 0.1f,
			100.0f);

		int modelLoc = glGetUniformLocation(m_material.shaderID(), "model");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

		int viewLoc = glGetUniformLocation(m_material.shaderID(), "view");
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

		int projectionLoc = glGetUniformLocation(m_material.shaderID(), "projection");
		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

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