cmake_minimum_required(VERSION 3.20)

project(intra_kernel_profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(IKP_BUILD_EXAMPLES "Build Intra-Kernel Profiler examples" ON)
option(IKP_BUILD_TOOLS    "Build Intra-Kernel Profiler tools (requires CUPTI / NVBit)" OFF)

find_package(CUDAToolkit REQUIRED)

# Header-only library target
add_library(intra_kernel_profiler_headers INTERFACE)
target_include_directories(intra_kernel_profiler_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (IKP_BUILD_EXAMPLES)
  add_executable(ikp_trace_record examples/trace/record.cu)
  target_link_libraries(ikp_trace_record PRIVATE intra_kernel_profiler_headers CUDA::cudart)

  add_executable(ikp_trace_block_filter examples/trace/block_filter.cu)
  target_link_libraries(ikp_trace_block_filter PRIVATE intra_kernel_profiler_headers CUDA::cudart)

  add_executable(ikp_trace_sampled_loop examples/trace/sampled_loop.cu)
  target_link_libraries(ikp_trace_sampled_loop PRIVATE intra_kernel_profiler_headers CUDA::cudart)

  add_executable(ikp_gemm_demo examples/gemm/tiled_gemm.cu)
  target_link_libraries(ikp_gemm_demo PRIVATE intra_kernel_profiler_headers CUDA::cudart)
endif()

if (IKP_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# Install rules for the header-only library
include(GNUInstallDirs)

install(DIRECTORY include/intra_kernel_profiler
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS intra_kernel_profiler_headers
  EXPORT intra_kernel_profiler-targets
)

install(EXPORT intra_kernel_profiler-targets
  FILE intra_kernel_profiler-targets.cmake
  NAMESPACE intra_kernel_profiler::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intra_kernel_profiler
)
