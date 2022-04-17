#pragma once

#include "Common.cuh"

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

		operator T* () const { return m_buffer; }

		~VtBuffer()
		{
			destroy();
		}

		T& operator[](size_t index)
		{
			assert(m_buffer);
			assert(index < m_count);
			return m_buffer[index];
		}

		size_t size() const { return m_count; }

		void push_back(const T& t)
		{
			reserve(m_count + 1);
			m_buffer[m_count++] = t;
		}

		void push_back(size_t newCount, const T& val)
		{
			for (int i = 0; i < newCount; i++)
			{
				push_back(val);
			}
		}

		void push_back(const vector<T>& data)
		{
			size_t offset = m_count;
			resize(m_count + data.size());
			memcpy(m_buffer + offset, data.data(), data.size() * sizeof(T));
		}

		void reserve(size_t minCapacity)
		{
			if (minCapacity > m_capacity)
			{
				// growth factor of 1.5
				const size_t newCapacity = minCapacity * 3 / 2;

				T* newBuf = VtAllocBuffer<T>(newCapacity);

				// copy contents to new buffer			
				if (m_buffer)
				{
					memcpy(newBuf, m_buffer, m_count * sizeof(T));
					VtFreeBuffer(m_buffer);
				}

				// swap
				m_buffer = newBuf;
				m_capacity = newCapacity;
			}
		}

		void resize(size_t newCount)
		{
			reserve(newCount);
			m_count = newCount;
		}

		void resize(size_t newCount, const T& val)
		{
			const size_t startInit = m_count;
			const size_t endInit = newCount;

			resize(newCount);

			// init any new entries
			for (size_t i = startInit; i < endInit; ++i)
				m_buffer[i] = val;
		}

		T* data() const
		{
			return m_buffer;
		}

		void destroy()
		{
			if (m_buffer != nullptr)
			{
				VtFreeBuffer(m_buffer);
			}
			m_count = 0;
			m_capacity = 0;
			m_buffer = nullptr;
		}

	private:
		size_t m_count = 0;
		size_t m_capacity = 0;
		T* m_buffer = nullptr;
	};

	template <class T>
	class VtRegisteredBuffer
	{
	public:
		VtRegisteredBuffer() {}

		VtRegisteredBuffer(const VtRegisteredBuffer&) = delete;

		VtRegisteredBuffer& operator=(const VtRegisteredBuffer&) = delete;

		~VtRegisteredBuffer()
		{
			destroy();
		}

		T* data() const { return m_buffer; }

		operator T* () const { return m_buffer; }

		T& operator[](size_t index)
		{
			assert(m_bufferCPU);
			assert(index < m_count);
			return m_bufferCPU[index];
		}

		size_t size() const { return m_count; }

		void destroy()
		{
			if (m_cudaVboResource != nullptr)
			{
				//fmt::print("Info(VtBuffer): Release CUDA Resource ({})\n", (int)m_cudaVboResource);
				checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVboResource));
			}
			if (m_bufferCPU)
			{
				cudaFree(m_bufferCPU);
			}
			m_count = 0;
			m_buffer = nullptr;
			m_bufferCPU = nullptr;
			m_cudaVboResource = nullptr;
		}
	public:
		// CUDA interop with OpenGL
		void registerBuffer(GLuint vbo)
		{
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVboResource, vbo, cudaGraphicsRegisterFlagsNone));

			// map (example 'gl_cuda_interop_pingpong_st' says map and unmap only needs to be done once)
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaVboResource, 0));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_buffer, &m_numBytes,
				m_cudaVboResource));
			m_count = m_numBytes / sizeof(T);

			// unmap
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0));
		}

		size_t m_count = 0;
		size_t m_numBytes = 0;
		T* m_buffer = nullptr;
		T* m_bufferCPU = nullptr;
		struct cudaGraphicsResource* m_cudaVboResource = nullptr;
	};

	template <class T>
	class VtMergedBuffer
	{
	public:
		VtMergedBuffer() {}
		VtMergedBuffer(const VtMergedBuffer&) = delete;
		VtMergedBuffer& operator=(const VtMergedBuffer&) = delete;

		void registerNewBuffer(GLuint vbo)
		{
			auto rbuf = make_shared<VtRegisteredBuffer<T>>();
			rbuf->registerBuffer(vbo);
			m_rbuffers.push_back(rbuf);

			int last = m_offsets.size() - 1;
			size_t offset = m_offsets.empty() ? 0 : m_offsets[last] + m_rbuffers[last]->size();
			m_offsets.push_back(offset);

			// copy from rbuffers to vbuffer
			m_vbuffer.resize(m_vbuffer.size() + rbuf->size());
			cudaMemcpy(m_vbuffer.data() + offset, rbuf->data(), rbuf->size() * sizeof(T), cudaMemcpyDefault);
		}

		size_t size()
		{
			return m_vbuffer.size();
		}

		void sync()
		{
			// copy from vbuffer to rbuffers
			for (int i = 0; i < m_rbuffers.size(); i++)
			{
				cudaMemcpy(m_rbuffers[i]->data(), m_vbuffer.data() + m_offsets[i], m_rbuffers[i]->size() * sizeof(T), cudaMemcpyDefault);
			}
		}

		operator T* () const { return m_vbuffer; }
	private:
		vector<shared_ptr<VtRegisteredBuffer<T>>> m_rbuffers;
		vector<size_t> m_offsets;
		VtBuffer<T> m_vbuffer;
	};
}