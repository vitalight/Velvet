#pragma once

#include "Component.h"
#include "Mesh.h"
#include "Material.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace Velvet
{
	class MeshRenderer : public Component
	{
	public:
		MeshRenderer() {}

		MeshRenderer(Mesh mesh, Material material)
			: m_mesh(mesh), m_material(material) {}

		void Update() override
		{
			// draw triangles
			m_material.Use();
			m_material.SetInt("texture1", 0);
			m_material.SetInt("texture2", 1);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m_material.texture1);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, m_material.texture2);
			
			glm::mat4 model = glm::mat4(1.0f);
			model = glm::rotate(model, (float)glfwGetTime() * glm::radians(50.0f),
				glm::vec3(0.5f, 1.0f, 0.0f));

			glm::mat4 view = glm::mat4(1.0f);
			// note that we¡¯re translating the scene in the reverse direction
			view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));

			glm::mat4 projection;
			projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f,
				100.0f);

			int modelLoc = glGetUniformLocation(m_material.shaderID(), "model");
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

			int viewLoc = glGetUniformLocation(m_material.shaderID(), "view");
			glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

			int projectionLoc = glGetUniformLocation(m_material.shaderID(), "projection");
			glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

			glBindVertexArray(m_mesh.VAO());

			//if (m_mesh.useIndices())
			//{
			//	glDrawElements(GL_TRIANGLES, m_mesh.numVertices(), GL_UNSIGNED_INT, 0);
			//}
			//else
			//{
			//	glDrawArrays(GL_TRIANGLES, 0, m_mesh.numVertices());
			//}

			glm::vec3 cubePositions[] = {
				glm::vec3(0.0f, 0.0f, 0.0f),
				glm::vec3(2.0f, 5.0f, -15.0f),
				glm::vec3(-1.5f, -2.2f, -2.5f),
				glm::vec3(-3.8f, -2.0f, -12.3f),
				glm::vec3(2.4f, -0.4f, -3.5f),
				glm::vec3(-1.7f, 3.0f, -7.5f),
				glm::vec3(1.3f, -2.0f, -2.5f),
				glm::vec3(1.5f, 2.0f, -2.5f),
				glm::vec3(1.5f, 0.2f, -1.5f),
				glm::vec3(-1.3f, 1.0f, -1.5f)
			};

			for (unsigned int i = 0; i < 10; i++)
			{
				glm::mat4 model = glm::mat4(1.0f);
				model = glm::translate(model, cubePositions[i]);
				float angle = 20.0f * i;
				model = glm::rotate(model, glm::radians(angle),
					glm::vec3(1.0f, 0.3f, 0.5f));
				m_material.SetMat4("model", model);
				glDrawArrays(GL_TRIANGLES, 0, 36);
			}
		}

		Material material() const
		{
			return m_material;
		}

	private:
		Mesh m_mesh;
		Material m_material;
	};
}