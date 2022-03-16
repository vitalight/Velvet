#include "MeshRenderer.h"

#include "Global.h"
#include "Camera.h"
#include "Actor.h"

namespace Velvet
{
	inline MeshRenderer::MeshRenderer()
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
		m_material.SetInt("texture1", 0);
		m_material.SetInt("texture2", 1);
		m_material.SetVec3("objectColor", 1.0f, 0.5f, 0.31f);
		m_material.SetVec3("lightColor", 1.0f, 1.0f, 1.0f);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_material.texture1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, m_material.texture2);

		glm::mat4 model = actor->transform.matrix();
		glm::mat4 view = Global::mainCamera->view();
		glm::mat4 projection = glm::perspective(glm::radians(Global::mainCamera->zoom), 800.0f / 600.0f, 0.1f,
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