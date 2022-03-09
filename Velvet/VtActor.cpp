#include "VtActor.h"

using namespace Velvet;

shared_ptr<VtActor> VtActor::FixedQuad()
{
	vector<float> vertices = {
		0.5f, 0.5f, 0.0f, // top right
		0.5f, -0.5f, 0.0f, // bottom right
		-0.5f, -0.5f, 0.0f, // bottom left
		-0.5f, 0.5f, 0.0f // top left
	};
	vector<unsigned int> indices = { // note that we start from 0!
		0, 1, 3, // first triangle
		1, 2, 3 // second triangle
	};

	const char* vertexShaderSource = SHADER(
	layout(location = 0) in vec3 aPos;

		void main()
		{
			gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
		}
	);

	const char* fragmentShaderSource = SHADER(
	out vec4 FragColor;
	uniform vec4 ourColor;

		void main()
		{
			FragColor = ourColor;// vec4(1.0f, 0.5f, 0.2f, 1.0f);
		}
	);

	Mesh mesh(vertices, indices);
	Material material(vertexShaderSource, fragmentShaderSource);
	shared_ptr<MeshRenderer> renderer(new MeshRenderer(mesh, material));
	shared_ptr<MaterialAnimator> animator(new MaterialAnimator(5.0f));

	shared_ptr<VtActor> actor(new VtActor("Fixed Quad"));
	actor->AddComponent(renderer);
	actor->AddComponent(animator);
	return actor;
}

shared_ptr<VtActor> Velvet::VtActor::FixedTriangle()
{
	vector<float> vertices = {
		// positions // colors
		0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom right
		-0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom left
		0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f // top
	};
	vector<unsigned int> indices = { // note that we start from 0!
		0, 1, 2,
	};

	const char* vertexShaderSource = SHADER(
		layout(location = 0) in vec3 aPos; // position has attribute position 0
		layout(location = 1) in vec3 aColor; // color has attribute position 1
		out vec3 ourColor; // output a color to the fragment shader
		void main()
		{
			gl_Position = vec4(aPos, 1.0);
			ourColor = aColor; // set ourColor to input color from the vertex data
		}
	);

	const char* fragmentShaderSource = SHADER(
		out vec4 FragColor;
		in vec3 ourColor;
		void main()
		{
			FragColor = vec4(ourColor, 1.0);
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
