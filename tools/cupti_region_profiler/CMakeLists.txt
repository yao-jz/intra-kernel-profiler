# CUPTI region profiler injection libraries
#
# Standalone usage:  make -j

cmake_minimum_required(VERSION 3.20)

find_package(CUDAToolkit REQUIRED)

# Locate CUPTI
set(CUPTI_INCLUDE_DIR "${CUDAToolkit_LIBRARY_DIR}/../extras/CUPTI/include" CACHE PATH "CUPTI include dir")
set(CUPTI_LIB_DIR     "${CUDAToolkit_LIBRARY_DIR}/../extras/CUPTI/lib64"   CACHE PATH "CUPTI lib dir")

function(add_cupti_injector target source)
  add_library(${target} SHARED ${source})
  target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR})
  target_link_directories(${target} PRIVATE ${CUPTI_LIB_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs)
  target_link_libraries(${target} PRIVATE cupti CUDA::cudart CUDA::cuda_driver dl pthread)
endfunction()

add_cupti_injector(ikp_cupti_pcsamp        src/ikp_cupti_pcsamp_inject.cpp)
add_cupti_injector(ikp_cupti_sassmetrics   src/ikp_cupti_sassmetrics_inject.cpp)
add_cupti_injector(ikp_cupti_instrexec     src/ikp_cupti_instrexec_inject.cpp)
add_cupti_injector(ikp_cupti_pmsamp        src/ikp_cupti_pmsamp_inject.cpp)
