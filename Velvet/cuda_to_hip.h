#pragma once

// CUDA-to-HIP compatibility header for Velvet.
// Provides CUDA API mappings when building with HIP for AMD GPUs.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <hip/hip_gl_interop.h>

// Runtime API - memory
#define cudaMallocManaged          hipMallocManaged
#define cudaFree                   hipFree
#define cudaMemcpy                 hipMemcpy
#define cudaMemcpyDefault          hipMemcpyDefault
#define cudaMemsetAsync            hipMemsetAsync
#define cudaDeviceSynchronize      hipDeviceSynchronize
#define cudaMemcpyHostToDevice     hipMemcpyHostToDevice

// hipMemcpyToSymbolAsync requires offset and kind args; CUDA has defaults
#define cudaMemcpyToSymbolAsync(symbol, src, size) \
    hipMemcpyToSymbolAsync(symbol, src, size, 0, hipMemcpyHostToDevice, 0)

// Events
#define cudaEvent_t                hipEvent_t
#define cudaEventCreate            hipEventCreate
#define cudaEventRecord            hipEventRecord
#define cudaEventSynchronize       hipEventSynchronize
#define cudaEventElapsedTime       hipEventElapsedTime
#define cudaEventDestroy           hipEventDestroy

// Error handling
#define cudaError_t                hipError_t
#define cudaSuccess                hipSuccess
#define cudaGetErrorName           hipGetErrorName
#define cudaGetLastError           hipGetLastError

// OpenGL interop
#define cudaGraphicsResource       hipGraphicsResource
#define cudaGraphicsGLRegisterBuffer       hipGraphicsGLRegisterBuffer
#define cudaGraphicsMapResources           hipGraphicsMapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsUnmapResources         hipGraphicsUnmapResources
#define cudaGraphicsUnregisterResource     hipGraphicsUnregisterResource
#define cudaGraphicsRegisterFlagsNone      hipGraphicsRegisterFlagsNone

// Status check for the call sites that spell it checkCudaErrors(). The CUDA
// build gets that macro from the CUDA samples helper header; the HIP build
// uses this instead, so no CUDA samples header is on the AMD include path.
namespace Velvet
{
	inline void AbortOnHipError(hipError_t status, const char* expr,
		const char* file, int line)
	{
		if (status == hipSuccess)
		{
			return;
		}
		std::fprintf(stderr, "%s:%d: %s -> %s (%d)\n", file, line, expr,
			hipGetErrorString(status), static_cast<int>(status));
		std::exit(EXIT_FAILURE);
	}
}

#ifndef checkCudaErrors
#define checkCudaErrors(expr) \
	::Velvet::AbortOnHipError((expr), #expr, __FILE__, __LINE__)
#endif

#else  // CUDA path

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>  // checkCudaErrors, from Velvet/External/cuda

#endif
