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
			resize(size);
		}

		VtBuffer(const VtBuffer&) = delete;

		VtBuffer& operator=(const VtBuffer&) = delete;

		~VtBuffer()
		{
			destroy();
		}

	public:
		T& operator[](int index)
		{
			assert(m_buffer);
			assert(index < m_count);
			return m_buffer[index];
		}

		int size() const { return m_count; }

		void push_back(const T& t)
		{
			assert(m_buffer);
			reserve(m_count + 1);
			m_buffer[m_count++] = t;
		}

		void reserve(int minCapacity)
		{
			if (minCapacity > m_capacity)
			{
				// growth factor of 1.5
				const int newCapacity = minCapacity * 3 / 2;

				T* newBuf = VtAllocBuffer<T>(newCapacity);
				fmt::print("VtAllocBuffer {}\n", (int)newBuf);

				// copy contents to new buffer			
				memcpy(newBuf, m_buffer, m_count * sizeof(T));

				if (m_buffer)
					VtFreeBuffer(m_buffer);

				// swap
				m_buffer = newBuf;
				m_capacity = newCapacity;
			}
		}

		void resize(int newCount)
		{
			reserve(newCount);
			m_count = newCount;
		}

		void resize(int newCount, const T& val)
		{
			const int startInit = m_count;
			const int endInit = newCount;

			resize(newCount);

			// init any new entries
			for (int i = startInit; i < endInit; ++i)
				m_buffer[i] = val;
		}

		T* data() const
		{
			return m_buffer;
		}

		void destroy()
		{
			if (m_cudaVboResource != nullptr)
			{
				checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVboResource));
			}
			else if (m_buffer != nullptr)
			{
				fmt::print("VtFreeBuffer {}\n", (int)m_buffer);
				VtFreeBuffer(m_buffer);
			}
			m_count = 0;
			m_capacity = 0;
			m_buffer = nullptr;
			m_cudaVboResource = nullptr;
		}

	public:
		// CUDA interop with OpenGL
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
			m_count = (int)(num_bytes / sizeof(T));
			m_capacity = 0;
		}

		void Unmap()
		{
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0));
		}

	private:
		int m_count = 0;
		int m_capacity = 0;
		T* m_buffer = nullptr;
		struct cudaGraphicsResource* m_cudaVboResource;
	};
}