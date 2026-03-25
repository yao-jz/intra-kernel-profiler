NVCC ?= nvcc

# If CUDA_HOME is not set, infer from NVCC location.
CUDA_HOME ?= $(abspath $(dir $(dir $(shell command -v $(NVCC) 2>/dev/null))))
ifeq ($(CUDA_HOME),)
  CUDA_HOME := /usr/local/cuda
endif

CUPTI_HOME ?= $(CUDA_HOME)/extras/CUPTI

CUDA_INCLUDE_DIR ?= $(CUDA_HOME)/include
CUDA_LIB_DIR ?= $(firstword $(wildcard $(CUDA_HOME)/lib64) $(wildcard $(CUDA_HOME)/lib))
CUDA_STUBS_DIR ?= $(CUDA_LIB_DIR)/stubs

CUPTI_INCLUDE_DIR ?= $(CUPTI_HOME)/include
CUPTI_LIB_DIR ?= $(firstword $(wildcard $(CUPTI_HOME)/lib64) $(wildcard $(CUPTI_HOME)/lib))

# Some installs ship only libcupti.so.<version> (no libcupti.so symlink).
CUPTI_SO := $(firstword \
  $(wildcard $(CUPTI_LIB_DIR)/libcupti.so) \
  $(wildcard $(CUPTI_LIB_DIR)/libcupti.so.*) \
)

ifeq ($(strip $(CUPTI_SO)),)
  $(error "Could not find libcupti.so* under $(CUPTI_LIB_DIR). Set CUDA_HOME or CUPTI_LIB_DIR.")
endif

TARGETS := \
  ikp_cupti_pcsamp.so \
  ikp_cupti_sassmetrics.so \
  ikp_cupti_instrexec.so \
  ikp_cupti_pmsamp.so

NVCCFLAGS ?= -O2 -lineinfo -std=c++17 -Xcompiler -fPIC -shared
INCLUDES := -I"$(CUDA_INCLUDE_DIR)" -I"$(CUPTI_INCLUDE_DIR)"

# Link against libcuda via stubs so we can compile on nodes without driver dev files.
LDFLAGS := \
  -L"$(CUDA_LIB_DIR)" \
  -L"$(CUDA_STUBS_DIR)" \
  -L"$(CUPTI_LIB_DIR)" \
  -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) \
  -Xlinker -rpath -Xlinker $(CUPTI_LIB_DIR)

LDLIBS := \
  -l:$(notdir $(CUPTI_SO)) \
  -lcudart \
  -lcuda \
  -ldl \
  -lpthread

.PHONY: all clean print-vars

all: $(TARGETS)

ikp_cupti_pcsamp.so: src/ikp_cupti_pcsamp_inject.cpp src/ikp_cupti_common.h
	"$(NVCC)" $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LDLIBS)

ikp_cupti_sassmetrics.so: src/ikp_cupti_sassmetrics_inject.cpp src/ikp_cupti_common.h
	"$(NVCC)" $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LDLIBS)

ikp_cupti_instrexec.so: src/ikp_cupti_instrexec_inject.cpp src/ikp_cupti_common.h
	"$(NVCC)" $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LDLIBS)

ikp_cupti_pmsamp.so: src/ikp_cupti_pmsamp_inject.cpp src/ikp_cupti_common.h
	"$(NVCC)" $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS) *.o

print-vars:
	@echo "NVCC=$(NVCC)"
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "CUPTI_HOME=$(CUPTI_HOME)"
	@echo "CUDA_INCLUDE_DIR=$(CUDA_INCLUDE_DIR)"
	@echo "CUDA_LIB_DIR=$(CUDA_LIB_DIR)"
	@echo "CUDA_STUBS_DIR=$(CUDA_STUBS_DIR)"
	@echo "CUPTI_INCLUDE_DIR=$(CUPTI_INCLUDE_DIR)"
	@echo "CUPTI_LIB_DIR=$(CUPTI_LIB_DIR)"
	@echo "CUPTI_SO=$(CUPTI_SO)"

