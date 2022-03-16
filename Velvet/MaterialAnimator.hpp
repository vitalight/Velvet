#pragma once

#include "Component.h"
#include "MeshRenderer.h"
#include "Actor.h"

namespace Velvet
{
	// Animate material color with time
	class MaterialAnimator : public Component
	{
	public:
		MaterialAnimator(float speed = 1.0f) : m_speed(speed) 
		{
			name = __func__;
		};

		void Update() override
		{
			auto renderer = actor->GetComponent<MeshRenderer>();
			if (!renderer)
			{
				fmt::print("MaterialAnimator: Renderer not found\n");
				return;
			}
			auto material = renderer->material();
			material.Use();
			float timeValue = glfwGetTime() * m_speed;
			float greenValue = sin(timeValue) / 2.0f + 0.5f;
			int vertexColorLocation = glGetUniformLocation(material.shaderID(), "ourColor");
			glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
		}
	private:
		float m_speed = 1.0f;
	};
}