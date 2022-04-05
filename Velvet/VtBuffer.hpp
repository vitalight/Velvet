#pragma once

namespace Velvet
{
	template <class T>
	class VtBuffer
	{
	public:
		VtBuffer() {}

		VtBuffer(uint size)
		{
			Resize(size);
		}

		VtBuffer(const VtBuffer&) = delete;

		VtBuffer& operator=(const VtBuffer&) = delete;

		~VtBuffer()
		{
			if (m_cudaVboResource != nullptr)
			{
				fmt::print("dtor cudaResource\n");
				checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVboResource));
			}
			else if (m_buffer != nullptr)
			{
				FreeArray((void*)m_buffer);
			}
		}

		void Resize(uint size)
		{
			if (m_buffer != nullptr)
			{
				FreeArray((void*)m_buffer);
			}
			AllocateArray((void**)&m_buffer, sizeof(T) * size);
		}

		void RegisterBuffer(GLuint vbo)
		{
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVboResource, vbo, cudaGraphicsRegisterFlagsNone));
		}

		void Map()
		{
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaVboResource, 0));
			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_buffer, &num_bytes,
				m_cudaVboResource));
		}

		void Unmap()
		{
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0));
		}

		T* raw() const
		{
			return m_buffer;
		}

	private:
		T* m_buffer = nullptr;
		struct cudaGraphicsResource* m_cudaVboResource;
	};
}