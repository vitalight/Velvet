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
		MeshRenderer(shared_ptr<Mesh> mesh, shared_ptr<Material> material);

		MeshRenderer(shared_ptr<Mesh> mesh, shared_ptr<Material> material, shared_ptr<Material> shadowMaterial);

		void SetupLighting(shared_ptr<Material> m_material);

		void Render(glm::mat4 lightMatrix);

		void RenderShadow(glm::mat4 lightMatrix);

		shared_ptr<Material> material() const;

		bool hidden = false;

	private:
		shared_ptr<Mesh> m_mesh;
		shared_ptr<Material> m_material;
		shared_ptr<Material> m_shadowMaterial;
	};
}