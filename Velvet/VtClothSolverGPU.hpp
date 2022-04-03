#pragma once

#include <glad/glad.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// CUDA helper functions
//#include <helper_cuda.h> 

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

namespace Velvet
{
	class VtClothSolverGPU
	{
	public:

		VtClothSolverGPU()
		{
			createVBO(&m_vbo, &m_cudaVboResource, cudaGraphicsMapFlagsNone);
		}

		~VtClothSolverGPU()
		{
			deleteVBO(&m_vbo, m_cudaVboResource);
		}

		void Simulate()
		{

		}

	private:

		GLuint m_vbo;
		struct cudaGraphicsResource* m_cudaVboResource;

		void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
			unsigned int vbo_res_flags)
		{
			//assert(vbo);

			// create buffer object
			glGenBuffers(1, vbo);
			glBindBuffer(GL_ARRAY_BUFFER, *vbo);

			// initialize buffer object
			unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
			glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

			glBindBuffer(GL_ARRAY_BUFFER, 0);

			// register this buffer object with CUDA
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

			//SDK_CHECK_ERROR_GL();
		}

		void runCuda(struct cudaGraphicsResource** vbo_resource)
		{
			// map OpenGL buffer object for writing from CUDA
			float4* dptr;
			checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
				*vbo_resource));
			//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

			// execute the kernel
			//    dim3 block(8, 8, 1);
			//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
			//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

			//launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

			// unmap buffer object
			checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
		}

		void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
		{
			// unregister this buffer object with CUDA
			checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

			glBindBuffer(1, *vbo);
			glDeleteBuffers(1, vbo);

			*vbo = 0;
		}
	};
}