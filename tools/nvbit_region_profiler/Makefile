NVCC ?= nvcc
CXX ?= g++
PTXAS ?= ptxas

NVBIT_PATH ?=
# Allow `make clean` without requiring NVBIT_PATH (parse-time checks would fail).
ifeq ($(filter clean,$(MAKECMDGOALS)),clean)
  # ok
else
  ifeq ($(NVBIT_PATH),)
    $(error NVBIT_PATH is not set. Point it to NVBit root (contains core/).)
  endif
endif

NVBIT_CORE := $(NVBIT_PATH)/core
INCLUDES := -I$(NVBIT_CORE)
LIBS := -L$(NVBIT_CORE) -lnvbit

NVCC_PATH := -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# Optional CUPTI (for cubinCrc compatibility with CUPTI tools).
# Default is enabled; set IKP_NVBIT_USE_CUPTI_CRC=0 to disable.
IKP_NVBIT_USE_CUPTI_CRC ?= 1
CUDA_HOME ?= $(dir $(dir $(shell which nvcc)))
CUPTI_LIBDIR ?= $(CUDA_HOME)/extras/CUPTI/lib64
CUPTI_INCDIR ?= $(CUDA_HOME)/extras/CUPTI/include
ifeq ($(IKP_NVBIT_USE_CUPTI_CRC),1)
  INCLUDES += -I$(CUPTI_INCDIR) -DIKP_NVBIT_USE_CUPTI_CRC=1
  LIBS += -L$(CUPTI_LIBDIR) -lcupti
else
  INCLUDES += -DIKP_NVBIT_USE_CUPTI_CRC=0
endif

ARCH ?= 90
NVCCFLAGS := -std=c++17 -O3 -arch=sm_$(ARCH) -Xcompiler -fPIC -D_FORCE_INLINES

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

all: region_profiler.so

region_profiler.so: region_profiler.o inject_funcs.o $(NVBIT_CORE)/libnvbit.a
	$(NVCC) $(NVCCFLAGS) region_profiler.o inject_funcs.o $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

region_profiler.o: src/region_profiler.cu src/common.h
	$(NVCC) -dc -c $(NVCCFLAGS) $(INCLUDES) -Xcompiler -Wall $< -o $@

inject_funcs.o: src/inject_funcs.cu src/common.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -c $< -o $@

clean:
	rm -f *.so *.o

