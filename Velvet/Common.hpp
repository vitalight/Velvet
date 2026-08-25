#pragma once

#include <glm/glm.hpp>
#include <imgui.h>
#include <functional>
#include <vector>

#define IMGUI_LEFT_LABEL(func, label, ...) (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

// Only initialize value on host.
// Since CUDA/HIP does not allow dynamic initialization for __device__ variables,
// suppress default member initializers when compiling with NVCC or hipcc.
// The struct is used both as a host variable and as __device__ __constant__.
#if defined(__CUDACC__) || defined(__HIPCC__)
	#define HOST_INIT(val)
#else
	#define HOST_INIT(val) = val
#endif

struct VtSimParams
{
	int numSubsteps					HOST_INIT(2);
	int numIterations				HOST_INIT(4);						//!< Number of solver iterations to perform per-substep
	int maxNumNeighbors				HOST_INIT(64);
	float maxSpeed					HOST_INIT(50);						//!< The magnitude of particle velocity will be clamped to this value at the end of each step

	// forces
	glm::vec3 gravity				HOST_INIT(glm::vec3(0, -9.8f, 0));	//!< Constant acceleration applied to all particles
	float bendCompliance			HOST_INIT(0.0f);
	float damping					HOST_INIT(0.25f);					//!< Viscous drag force, applies a force proportional, and opposite to the particle velocity
	float relaxationFactor			HOST_INIT(1.0f);					//!< Control the convergence rate of the parallel solver, default: 1, values greater than 1 may lead to instability
	float longRangeStretchiness		HOST_INIT(1.2f);

	// collision
	float collisionMargin			HOST_INIT(0.06f);					//!< Distance particles maintain against shapes, note that for robust collision against triangle meshes this distance should be greater than zero
	float friction					HOST_INIT(0.1f);					//!< Coefficient of friction used when colliding against shapes
	bool enableSelfCollision		HOST_INIT(true);
	int interleavedHash				HOST_INIT(3);						//!< Hash once every n substeps. This can improves performance greatly.

	// runtime info
	unsigned int numParticles;											//!< Total number of particles 
	float particleDiameter;												//!< The maximum interaction radius for particles
	float deltaTime;	

	// misc
	float particleDiameterScalar	HOST_INIT(1.5f);					//!< multiply original stretch length by this scalar to obtain particle diameter
	float hashCellSizeScalar		HOST_INIT(1.5f);					//!< multiply particle diameter by this scalar to obtain hash cell size

	// future updates
	//float wind[3];													//!< Constant acceleration applied to particles that belong to dynamic triangles, drag needs to be > 0 for wind to affect triangles
	//int relaxationMode;												//!< How the relaxation is applied inside the solver

	void OnGUI()
	{
		IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Substeps", &numSubsteps, 1, 20);
		IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Iterations", &numIterations, 1, 20);
		IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Max Speed", &maxSpeed, 1e-2f, 100);
		ImGui::Separator();
		IMGUI_LEFT_LABEL(ImGui::SliderFloat3, "Gravity", (float*)&gravity, -50, 50);
		IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Damping", &damping, 0, 10.0f);
		IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Friction", &friction, 0, 1);
		IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Collision Margin", &collisionMargin, 0, 0.5);
		IMGUI_LEFT_LABEL(ImGui::Checkbox, "Enable Self Collision", &enableSelfCollision);
		IMGUI_LEFT_LABEL(ImGui::SliderInt, "Interleaved Hash", &interleavedHash, 1, 10);
		ImGui::Separator();
		IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Relaxation Factor", &relaxationFactor, 0, 3.0);
		IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Bend Compliance", &bendCompliance, 0, 100.0, "%.3f", ImGuiSliderFlags_Logarithmic);
		IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Long Range Stretch", &longRangeStretchiness, 1.0, 2.0, "%.3f");
	}
};

struct VtGameState
{
	bool step = false;
	bool pause = false;
	bool renderWireframe = false;
	bool drawParticles = false;
	bool hideGUI = false;
	bool detailTimer = false;
};

template <class T, class... TArgs>
class VtCallback
{
public:
	void Register(const std::function<T>& func)
	{
		m_funcs.push_back(func);
	}

	// Named Args, not TArgs: VtCallback already has a TArgs parameter pack, and
	// reusing that name here shadows it, which nvcc/gcc reject ("shadows
	// template parameter"). Keep this name distinct.
	template <class... Args>
	void Invoke(Args... args)
	{
		for (const auto& func : m_funcs)
		{
			func(std::forward<Args>(args)...);
		}
	}

	void Clear()
	{
		m_funcs.clear();
	}

	bool empty()
	{
		return m_funcs.size() == 0;
	}

private:
	std::vector<std::function<T>> m_funcs;
};

enum class ColliderType
{
	Sphere,
	Plane,
	Cube,
};