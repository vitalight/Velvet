#include "VtActor.h"
#include "External/stb_image.h"

using namespace Velvet;

shared_ptr<VtActor> VtActor::FixedQuad()
{
	vector<float> vertices = {
		// positions // colors // texture coords
		0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top right
		0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom right
		-0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
		-0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // top left
	};
	vector<unsigned int> indices = { // note that we start from 0!
		0, 1, 3, // first triangle
		1, 2, 3 // second triangle
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
	uniform sampler2D texture1;
	uniform sampler2D texture2;
	void main()
	{
		FragColor = mix(texture(texture1, TexCoord),
			texture(texture2, TexCoord), 0.2);
	}
	);

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
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
			(void*)0);
		glEnableVertexAttribArray(0);
		// color attribute
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
			(void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
			(void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);
	}
	Material material(vertexShaderSource, fragmentShaderSource);
	material.texture1 = texture1;
	material.texture2 = texture2;

	shared_ptr<VtActor> actor(new VtActor("Fixed Quad"));

	shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));
	actor->AddComponent(renderer);

	//shared_ptr<MaterialAnimator> animator(new MaterialAnimator(5.0f));
	//actor->AddComponent(animator);

	return actor;
}

shared_ptr<VtActor> Velvet::VtActor::FixedTriangle()
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

	shared_ptr<VtActor> actor(new VtActor("Fixed Triangle"));
	actor->AddComponent(renderer);

	return actor;
}

void VtActor::Start()
{
	for (const auto& c : m_components)
	{
		c->Start();
	}
}

void VtActor::AddComponent(shared_ptr<VtComponent> component)
{
	m_components.push_back(component);
	component->actor = this;
}

void VtActor::OnDestroy()
{
	for (const auto& c : m_components)
	{
		c->OnDestroy();
	}
}

void VtActor::Update()
{
	for (const auto& c : m_components)
	{
		c->Update();
	}
}
