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

			//glm::mat4 trans = glm::mat4(1.0f);
			//trans = glm::translate(trans, glm::vec3(0.5f, -0.5f, 0.0f));
			//trans = glm::rotate(trans, (float)glfwGetTime(),
			//	glm::vec3(0.0f, 0.0f, 1.0f));
			//unsigned int transformLoc = glGetUniformLocation(m_material.shaderID(), "transform");
			//glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(trans));

			glm::mat4 model = glm::mat4(1.0f);
			model = glm::rotate(model, glm::radians(-55.0f),
				glm::vec3(1.0f, 0.0f, 0.0f));

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
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			//glBindVertexArray(0);
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