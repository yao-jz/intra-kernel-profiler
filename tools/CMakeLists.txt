# Umbrella CMakeLists for Intra-Kernel Profiler tools.
# Each tool can also be built standalone via its own Makefile.

add_subdirectory(cupti_region_profiler)
add_subdirectory(cupti_minimal_activity)

# NVBit tool requires NVBIT_PATH; only include if set.
if (DEFINED NVBIT_PATH)
  add_subdirectory(nvbit_region_profiler)
else()
  message(STATUS "Skipping nvbit_region_profiler (set NVBIT_PATH to enable)")
endif()
