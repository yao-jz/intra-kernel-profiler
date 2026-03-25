# CUPTI minimal activity example
#
# Standalone usage:  make -j

cmake_minimum_required(VERSION 3.20)

find_package(CUDAToolkit REQUIRED)

set(CUPTI_INCLUDE_DIR "${CUDAToolkit_LIBRARY_DIR}/../extras/CUPTI/include" CACHE PATH "CUPTI include dir")
set(CUPTI_LIB_DIR     "${CUDAToolkit_LIBRARY_DIR}/../extras/CUPTI/lib64"   CACHE PATH "CUPTI lib dir")

add_executable(cupti_minimal_activity src/main.cu)
target_include_directories(cupti_minimal_activity PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR})
target_link_directories(cupti_minimal_activity PRIVATE ${CUPTI_LIB_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs)
target_link_libraries(cupti_minimal_activity PRIVATE cupti CUDA::cudart CUDA::cuda_driver dl pthread)
