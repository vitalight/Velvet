#pragma once

#include "Component.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Mesh.hpp"
#include "Material.hpp"

namespace Velvet
{
	class MeshRenderer : public Component
	{
	public:
		MeshRenderer(Mesh mesh, Material material);

		MeshRenderer(Mesh mesh, Material material, Material shadowMaterial);

		void SetupLighting(Material m_material);

		void Render(glm::mat4 lightMatrix);

		void RenderShadow(glm::mat4 lightMatrix);

		Material material() const;

		bool hidden = false;

	private:
		Mesh m_mesh;
		Material m_material;
		Material m_shadowMaterial;
	};
}