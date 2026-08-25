cmake_minimum_required(VERSION 3.24)

project(Velvet LANGUAGES C CXX)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)

# GPU language setup
if(USE_HIP)
  # enable_language(HIP) auto-detects the host GPU arch (and errors on a
  # no-GPU host); pass -DCMAKE_HIP_ARCHITECTURES to override.
  enable_language(HIP)
  message(STATUS "Building with HIP, architectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
  enable_language(CUDA)
  message(STATUS "Building with CUDA")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages (all provided via vcpkg)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# HIP-specific dependencies
if(USE_HIP)
  find_package(hipcub REQUIRED)
  find_package(rocthrust REQUIRED)
endif()

# Host/GPU sources. Upstream keeps these as .cpp (built by MSVC) plus the two
# .cu files (built by nvcc) in Velvet.vcxproj; that layout is preserved here,
# and the AMD build below keeps the same split: the .cu files carry every
# kernel and every launch, the .cpp files are host code.
set(CPP_SOURCES
  Velvet/Actor.cpp
  Velvet/Component.cpp
  Velvet/GUI.cpp
  Velvet/GameInstance.cpp
  Velvet/Helper.cpp
  Velvet/Input.cpp
  Velvet/MeshRenderer.cpp
  Velvet/Timer.cpp
  Velvet/VtEngine.cpp
  Velvet/stb_image.cpp
  Velvet/main.cpp
)
set(CU_SOURCES
  Velvet/VtClothSolverGPU.cu
  Velvet/SpatialHashGPU.cu
)
set(SOURCES ${CPP_SOURCES} ${CU_SOURCES})

# Create executable
add_executable(Velvet ${SOURCES})

if(USE_HIP)
  # Only the device-code sources are HIP translation units. The host .cpp files
  # must stay C++: the GPU compiler defines __HIPCC__, and Common.hpp keys
  # HOST_INIT() off that macro to drop the default member initializers of
  # VtSimParams (a __device__ __constant__ copy of it cannot be dynamically
  # initialized). Compiling the host files as GPU sources therefore zeroed the
  # host copy of the simulation parameters too, so the solver ran with zero
  # substeps and zero gravity. This mirrors the NVIDIA build, where the .cpp
  # files are plain C++ and never see __CUDACC__.
  set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_definitions(Velvet PRIVATE USE_HIP)
  # hipCUB and rocThrust are header only, but their imported targets pull in
  # hip::device, whose interface appends "-x hip --offload-arch=..." to every
  # C++ source of the consuming target. That is what makes a mixed-language
  # target fail; take their header search paths instead, and link only
  # hip::host, for the runtime library and __HIP_PLATFORM_AMD__.
  foreach(rocm_target hip::hipcub roc::rocthrust roc::rocprim_hip)
    if(TARGET ${rocm_target})
      get_target_property(rocm_target_includes ${rocm_target}
        INTERFACE_INCLUDE_DIRECTORIES)
      if(rocm_target_includes)
        target_include_directories(Velvet SYSTEM PRIVATE ${rocm_target_includes})
      endif()
    endif()
  endforeach()
  target_link_libraries(Velvet PRIVATE hip::host)
  set_target_properties(Velvet PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )
endif()
# For the non-HIP (CUDA) build, language auto-detection drives the split:
# .cpp -> CXX, .cu -> CUDA, matching upstream's Visual Studio project.

# Include directories
target_include_directories(Velvet PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Velvet
  ${CMAKE_CURRENT_SOURCE_DIR}/Velvet/External
)

if(NOT USE_HIP)
  # The bundled CUDA samples helpers are for the CUDA build only. The HIP build
  # defines checkCudaErrors itself in Velvet/cuda_to_hip.h and never puts this
  # directory on the include path.
  target_include_directories(Velvet PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Velvet/External/cuda
  )
endif()

# Link libraries
target_link_libraries(Velvet PRIVATE
  OpenGL::GL
  glfw
  glad::glad
  glm::glm
  fmt::fmt
  assimp::assimp
  imgui::imgui
)

# Set output directory
set_target_properties(Velvet PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Copy assets to build directory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Velvet/Assets")
  add_custom_command(TARGET Velvet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/Velvet/Assets"
    "$<TARGET_FILE_DIR:Velvet>/Assets"
  )
endif()
