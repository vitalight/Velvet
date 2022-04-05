#pragma once

#include <iostream>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "helper_cuda.h"
#include "Mesh.hpp"
#include "VtClothSolverGPU.cuh"


using namespace std;

namespace Velvet
{
	class VtClothSolverGPU
	{
	public:

		void Initialize(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix)
		{
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVboResource, mesh->verticesVBO(), cudaGraphicsRegisterFlagsNone));

			glm::vec3* positions;
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaVboResource, 0));
			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes,
				m_cudaVboResource));

			m_numParticles = mesh->vertices().size();
			InitializePositions(positions, m_numParticles, modelMatrix);
			
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0));

			AllocateArray((void**)&m_velocities, sizeof(glm::vec3) * m_numParticles);
		}

		void Simulate()
		{
			// map OpenGL buffer object for writing from CUDA
			glm::vec3* positions;
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaVboResource, 0));
			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes,
				m_cudaVboResource));
			//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

			// execute the kernel
			//    dim3 block(8, 8, 1);
			//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
			//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

			//launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
			ApplyExternalForces(positions, m_velocities, m_numParticles);

			// unmap buffer object
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0));
		}

		void Finalize()
		{
			if (m_cudaVboResource != nullptr)
			{
				checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVboResource));
			}
			FreeArray((void*)m_velocities);
		}

	private:

		uint m_numParticles;
		glm::vec3* m_velocities;
		GLuint m_vbo;
		struct cudaGraphicsResource* m_cudaVboResource;
	};
}