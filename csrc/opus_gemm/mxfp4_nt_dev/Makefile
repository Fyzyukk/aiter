# Standalone dev / test build of the MXFP4 x MXFP4 NT 2x2 pipeline
#   ../include/gfx950/opus_gemm_pipeline_a4w4_mxfp4_nt_gfx950.cuh  (kernel)
#   ../include/gfx950/opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh    (traits / kargs)
# amdgpu_pin_agpr / amdgpu_pin_vgpr need the pin toolchain (clang 24 + pin patches).
HERE       := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# opus from the aiter blockscale branch: its tiled mma supports packed fp4 and builds with the pin toolchain
# (this branch's csrc/include/opus does not compile with clang 24 + pin patches yet)
OPUS_INCLUDE_DIR ?= /root/workspace/aiter-blockscale-mxfp8-nn-nt-tt/csrc/include
GFX950_INCLUDE   ?= $(HERE)../include/gfx950
ROCM_PATH ?= /opt/rocm
CLANG23_ROOT ?= /root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build
CXX       = $(CLANG23_ROOT)/bin/clang++
ARCH      ?= gfx950
FLAGS      = -I$(HERE) -I$(GFX950_INCLUDE) -I$(OPUS_INCLUDE_DIR) -std=c++17 -O3 -Wall -ffast-math --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH) $(EXTRA)
OMP_LIB    = $(ROCM_PATH)/lib/llvm/lib
LDFLAGS    = -fopenmp -L$(OMP_LIB) -lomp -Wl,-rpath,$(OMP_LIB) -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib
DEPS       = $(GFX950_INCLUDE)/opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh $(GFX950_INCLUDE)/opus_gemm_pipeline_a4w4_mxfp4_nt_gfx950.cuh
BUILD      ?= build
OUT        = $(BUILD)/gemm_a4w4_mxfp4_4wave.exe

all: $(OUT)
$(BUILD):
	mkdir -p $@
$(BUILD)/kernel.o: $(HERE)gemm_a4w4_mxfp4_nt_kernel.cc $(DEPS) | $(BUILD)
	cd $(BUILD) && $(CXX) -x hip $< $(FLAGS) -D__HIPCC_RTC__ -save-temps=obj -c -o kernel.o
$(BUILD)/host.o: $(HERE)gemm_a4w4_mxfp4_nt_host.cc $(GFX950_INCLUDE)/opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh | $(BUILD)
	$(CXX) -x hip $< $(FLAGS) -fopenmp -I$(ROCM_PATH)/lib/llvm/include -c -o $@
$(OUT): $(BUILD)/kernel.o $(BUILD)/host.o
	$(CXX) $^ --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH) $(LDFLAGS) -o $@
.PHONY: all
