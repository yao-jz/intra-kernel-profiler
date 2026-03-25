# NVBit region profiler tool
# Requires NVBIT_PATH to point to the NVBit root (contains core/libnvbit.a).
#
# Standalone usage:  make NVBIT_PATH=/path/to/nvbit ARCH=90a

cmake_minimum_required(VERSION 3.20)

if (NOT DEFINED NVBIT_PATH)
  message(FATAL_ERROR "NVBIT_PATH must be set to build the NVBit region profiler.")
endif()

set(NVBIT_CORE "${NVBIT_PATH}/core")

# inject_funcs is compiled as a regular object (not relocatable device code)
add_library(nvbit_inject_funcs OBJECT src/inject_funcs.cu)
target_include_directories(nvbit_inject_funcs PRIVATE ${NVBIT_CORE})
set_target_properties(nvbit_inject_funcs PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

# region_profiler needs relocatable device code
add_library(nvbit_region_profiler_obj OBJECT src/region_profiler.cu)
target_include_directories(nvbit_region_profiler_obj PRIVATE ${NVBIT_CORE})
set_target_properties(nvbit_region_profiler_obj PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_library(region_profiler SHARED
  $<TARGET_OBJECTS:nvbit_region_profiler_obj>
  $<TARGET_OBJECTS:nvbit_inject_funcs>
)
target_link_directories(region_profiler PRIVATE ${NVBIT_CORE})
target_link_libraries(region_profiler PRIVATE nvbit CUDA::cuda_driver CUDA::cudart_static)
