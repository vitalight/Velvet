#include "Actor.h"
#include "External/stb_image.h"
#include "DefaultAssets.h"


namespace Velvet
{
	inline Velvet::Actor::Actor() {}

	inline Velvet::Actor::Actor(string name) : m_name(name) {}

	shared_ptr<Actor> Actor::FixedQuad()
	{
		vector<float> vertices = DefaultAssets::quad_vertices;
		vector<unsigned int> indices = DefaultAssets::quad_indices;

		const char* vertexShaderSource =
			DefaultAssets::quad_shader_vertex;

		const char* fragmentShaderSource = 
			DefaultAssets::quad_shader_fragment;

		stbi_set_flip_vertically_on_load(true);
		unsigned int texture1;
		{
			glGenTextures(1, &texture1);
			glBindTexture(GL_TEXTURE_2D, texture1);
			// set the texture wrapping/filtering options (on currently bound texture)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			// load and generate the texture
			int width, height, nrChannels;
			unsigned char* data = stbi_load("Assets/container.jpg", &width, &height,
				&nrChannels, 0);
			if (data)
			{
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
					GL_UNSIGNED_BYTE, data);
				glGenerateMipmap(GL_TEXTURE_2D);
			}
			else
			{
				std::cout << "Failed to load texture" << std::endl;
			}
			stbi_image_free(data);
		}

		unsigned int texture2;
		{
			glGenTextures(1, &texture2);
			glBindTexture(GL_TEXTURE_2D, texture2);
			// set the texture wrapping/filtering options (on currently bound texture)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			// load and generate the texture
			int width, height, nrChannels;
			unsigned char* data = stbi_load("Assets/awesomeface.png", &width, &height,
				&nrChannels, 0);
			if (data)
			{
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA,
					GL_UNSIGNED_BYTE, data);
				glGenerateMipmap(GL_TEXTURE_2D);
			}
			else
			{
				std::cout << "Failed to load texture" << std::endl;
			}
			stbi_image_free(data);
		}

		Mesh mesh(vertices, indices);
		{
			// position attribute
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
				(void*)0);
			glEnableVertexAttribArray(0);
			// color attribute
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
				(void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
		}
		Material material(vertexShaderSource, fragmentShaderSource);
		material.texture1 = texture1;
		material.texture2 = texture2;

		shared_ptr<Actor> actor(new Actor("Fixed Quad"));

		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));
		actor->AddComponent(renderer);

		//shared_ptr<MaterialAnimator> animator(new MaterialAnimator(5.0f));
		//actor->AddComponent(animator);

		return actor;
	}

	shared_ptr<Actor> Actor::FixedTriangle()
	{
		vector<float> vertices = {
			// positions // colors // texture coords
			0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top right
			0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom right
			-0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
			-0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // top left
		};
		vector<unsigned int> indices = { // note that we start from 0!
			0, 1, 2,
		};

		const char* vertexShaderSource = SHADER(
			layout(location = 0) in vec3 aPos;
		layout(location = 1) in vec3 aColor;
		layout(location = 2) in vec2 aTexCoord;
		out vec3 ourColor;
		out vec2 TexCoord;
		void main()
		{
			gl_Position = vec4(aPos, 1.0);
			ourColor = aColor;
			TexCoord = aTexCoord;
		}
		);

		const char* fragmentShaderSource = SHADER(
			out vec4 FragColor;
		in vec3 ourColor;
		in vec2 TexCoord;
		uniform sampler2D ourTexture;
		void main()
		{
			FragColor = texture(ourTexture, TexCoord);
		}
		);

		Mesh mesh(vertices, indices);
		{
			// position attribute
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
				(void*)0);
			glEnableVertexAttribArray(0);
			// color attribute
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
				(void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
		}
		Material material(vertexShaderSource, fragmentShaderSource);
		shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));

		shared_ptr<Actor> actor(new Actor("Fixed Triangle"));
		actor->AddComponent(renderer);

		return actor;
	}

	void Actor::Start()
	{
		for (const auto& c : m_components)
		{
			c->Start();
		}
	}

	void Actor::AddComponent(shared_ptr<Component> component)
	{
		m_components.push_back(component);
		component->actor = this;
	}

	void Actor::OnDestroy()
	{
		for (const auto& c : m_components)
		{
			c->OnDestroy();
		}
	}

	void Actor::Update()
	{
		for (const auto& c : m_components)
		{
			c->Update();
		}
	}
}