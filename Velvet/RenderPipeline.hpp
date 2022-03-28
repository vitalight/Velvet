#pragma once

#include "VtGraphics.hpp"
#include "Config.hpp"

namespace Velvet
{
	class RenderPipeline
	{
	public:
		RenderPipeline()
		{
			SetupShadowMap();
		}

		RenderPipeline(const RenderPipeline&) = delete;

		~RenderPipeline()
		{
			if (depthMapFBO > 0)
			{
				glDeleteFramebuffers(1, &depthMapFBO);
			}
		}

		void Render()
		{
			vector<MeshRenderer*> renderers = Global::graphics->FindComponents<MeshRenderer>();
			//renderers = Cull(renderers);
			RenderShadow(renderers);
			RenderObjects(renderers);
		}

		unsigned int depthMapFBO;
	private:

		vector<MeshRenderer*> Cull(vector<MeshRenderer*> renderers)
		{
			return renderers;
		}

		glm::mat4 ComputeLightMatrix()
		{
			if (Global::light.size() == 0)
			{
				return glm::mat4(1);
			}
			// 1. render depth of scene to texture (from light's perspective)
			// --------------------------------------------------------------
			glm::mat4 lightProjection, lightView;
			glm::mat4 lightSpaceMatrix;
			float near_plane = 1.0f, far_plane = 7.5f;
			auto light = Global::light[0];
			glm::vec3 lightPos = light->position();
			if (light->type == LightType::SpotLight)
			{
				// note that if you use a perspective projection matrix you'll have to change the light position as the current light position isn't enough to reflect the whole scene
				lightProjection = glm::perspective(glm::radians(90.0f), (GLfloat)Config::shadowWidth / (GLfloat)Config::shadowHeight, near_plane, far_plane);
			}
			else
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

			auto originalWindowSize = Global::graphics->windowSize();
			glViewport(0, 0, Config::shadowWidth, Config::shadowHeight);
			glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
			glClear(GL_DEPTH_BUFFER_BIT);
			glCullFace(GL_FRONT);

			auto lightSpaceMatrix = ComputeLightMatrix();

			for (auto r : renderers)
			{
				r->RenderShadow(lightSpaceMatrix);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, originalWindowSize.x, originalWindowSize.y);
		}

		void RenderObjects(vector<MeshRenderer*> renderers)
		{        
			// reset viewport
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glCullFace(GL_BACK);

			auto lightSpaceMatrix = ComputeLightMatrix();

			for (auto r : renderers)
			{
				r->Render(lightSpaceMatrix);
			}
		}
	private:

		void SetupShadowMap()
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