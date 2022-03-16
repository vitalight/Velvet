#pragma once

#include "Component.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Mesh.h"
#include "Material.h"

namespace Velvet
{
	class MeshRenderer : public Component
	{
	public:
		MeshRenderer();

		MeshRenderer(Mesh mesh, Material material);

		void Update() override;

		Material material() const;

	private:
		Mesh m_mesh;
		Material m_material;
	};
}