# Dev / test build of the small-tile MXFP4 NT pipeline.  Standard ROCm toolchain (no pin attributes, no opus).
HERE      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GFX950_INCLUDE ?= $(HERE)../include/gfx950
ROCM_PATH ?= /opt/rocm
CLANG     ?= $(ROCM_PATH)/llvm/bin/clang++
CXX        = $(CLANG)
FLAGS      = -std=c++17 -O3 -ffast-math --offload-arch=gfx950 --rocm-path=$(ROCM_PATH) -fopenmp -I$(ROCM_PATH)/lib/llvm/include -I$(GFX950_INCLUDE) $(EXTRA)
LDFLAGS    = -L$(ROCM_PATH)/lib/llvm/lib -lomp -Wl,-rpath,$(ROCM_PATH)/lib/llvm/lib -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib
OUT       ?= /tmp/opus_mxfp4_small/small.exe
DEPS       = $(HERE)gemm_a4w4_mxfp4_nt_small_host.cc $(wildcard $(GFX950_INCLUDE)/opus_gemm_*_a4w4_mxfp4_nt_small*_gfx950.cuh)
all: $(OUT)
$(OUT): $(DEPS)
	mkdir -p $(dir $(OUT))
	$(CXX) -x hip $(HERE)gemm_a4w4_mxfp4_nt_small_host.cc $(FLAGS) $(LDFLAGS) -o $@
dis: $(OUT)
	$(CXX) -x hip $(HERE)gemm_a4w4_mxfp4_nt_small_host.cc $(FLAGS) --cuda-device-only -o $(dir $(OUT))kernel.co && $(ROCM_PATH)/llvm/bin/llvm-objdump -d --mcpu=gfx950 $(dir $(OUT))kernel.co > $(dir $(OUT))kernel.dis
.PHONY: all dis
