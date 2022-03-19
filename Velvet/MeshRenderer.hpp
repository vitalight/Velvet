#pragma once

#include "Component.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Model.hpp"
#include "Mesh.hpp"
#include "Material.hpp"

namespace Velvet
{
	class MeshRenderer : public Component
	{
	public:
		//MeshRenderer();

		MeshRenderer(Model model, Material material);

		void Update() override;

		Material material() const;

		bool hidden = false;

	private:
		Model m_model;
		Material m_material;
	};
}