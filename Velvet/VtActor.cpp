#include "VtActor.h"

using namespace Velvet;

shared_ptr<VtActor> VtActor::FixedTriangle()
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
	VtActor actor;
	actor.AddComponent(renderer);
	return make_shared<VtActor>(actor);
}

void VtActor::Start()
{
	for (const auto& c : m_components)
	{
		c->Start();
	}
}

void VtActor::AddComponent(shared_ptr<Component> component)
{
	m_components.push_back(component);
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
