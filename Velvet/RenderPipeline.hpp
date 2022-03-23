#pragma once

#include "VtGraphics.hpp"

namespace Velvet
{
	class RenderPipeline
	{
	public:
		RenderPipeline()
		{
			SetupShadow();
		}

		void Render()
		{
			vector<MeshRenderer*> renderers = Global::graphics->FindComponents<MeshRenderer>();
			renderers = Cull(renderers);
			RenderShadow(renderers);
			RenderObjects(renderers);
		}

		unsigned int depthMapFBO;
	private:
		const unsigned int SCR_WIDTH = 800;
		const unsigned int SCR_HEIGHT = 600;
		const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;

		vector<MeshRenderer*> Cull(vector<MeshRenderer*> renderers)
		{
			return renderers;
		}

		glm::mat4 ComputeLightMatrix()
		{
			// 1. render depth of scene to texture (from light's perspective)
			// --------------------------------------------------------------
			glm::mat4 lightProjection, lightView;
			glm::mat4 lightSpaceMatrix;
			float near_plane = 1.0f, far_plane = 7.5f;
			glm::vec3 lightPos = Global::light[0]->position();
			//if (Global::light[0]->type == LightType::SpotLight)
			//{
			//	lightProjection = glm::perspective(glm::radians(45.0f), (GLfloat)SHADOW_WIDTH / (GLfloat)SHADOW_HEIGHT, near_plane, far_plane); // note that if you use a perspective projection matrix you'll have to change the light position as the current light position isn't enough to reflect the whole scene
			//}
			//else
			{
				lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
			}
			lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
			lightSpaceMatrix = lightProjection * lightView;
			return lightSpaceMatrix;
		}

		void RenderShadow(vector<MeshRenderer*> renderers)
		{
			if (Global::light.size() == 0)
				return;

			glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
			glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
			glClear(GL_DEPTH_BUFFER_BIT);

			auto lightSpaceMatrix = ComputeLightMatrix();

			for (auto r : renderers)
			{
				r->RenderShadow(lightSpaceMatrix);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}

		void RenderObjects(vector<MeshRenderer*> renderers)
		{        
			// reset viewport
			glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			auto lightSpaceMatrix = ComputeLightMatrix();

			for (auto r : renderers)
			{
				r->Render(lightSpaceMatrix);
			}
		}
	private:

		void Unknown()
		{
			// shader configuration
			// --------------------
			//shader.use();
			//shader.setInt("diffuseTexture", 0);
			//shader.setInt("shadowMap", 1);
			//debugDepthQuad.use();
			//debugDepthQuad.setInt("depthMap", 0);
		}

		void SetupShadow()
		{
			// configure depth map FBO
			// -----------------------
			const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
			//unsigned int depthMapFBO;
			glGenFramebuffers(1, &depthMapFBO);
			// create depth texture
			unsigned int depthMap;
			glGenTextures(1, &depthMap);
			glBindTexture(GL_TEXTURE_2D, depthMap);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
			glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
			// attach depth texture as FBO's depth buffer
			glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
	};
}