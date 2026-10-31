NVCC  ?= nvcc
# -arch=native targets the GPU in the build machine; override e.g. ARCH=sm_86
ARCH  ?= native

# Workaround for glibc >= 2.41 + CUDA 13.1: glibc declares rsqrt/rsqrtf (C23) with
# noexcept, which clashes with CUDA's own declarations. We shadow bits/mathcalls.h with
# a private copy that lacks the rsqrt line (CUDA provides rsqrt itself).
COMPAT := build/compat
MATHCALLS := $(firstword $(wildcard /usr/include/x86_64-linux-gnu/bits/mathcalls.h /usr/include/bits/mathcalls.h))

FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -I$(COMPAT)
HDRS  := $(wildcard common/*.cuh) $(COMPAT)/bits/mathcalls.h

GEMM_SRCS := $(wildcard gemm/*.cu)
ATTN_SRCS := $(wildcard attention/*.cu)
# GEMM kernels are standalone binaries; attention kernels are shared libs driven from Python.
BINS := $(patsubst gemm/%.cu,build/%,$(GEMM_SRCS)) $(patsubst attention/%.cu,build/lib%.so,$(ATTN_SRCS))

all: $(BINS)

$(COMPAT)/bits/mathcalls.h: | build
	mkdir -p $(COMPAT)/bits
	sed '/(rsqrt,,/d' $(MATHCALLS) > $@

build/%: gemm/%.cu $(HDRS)
	$(NVCC) $(FLAGS) $< -o $@

build/lib%.so: attention/%.cu $(HDRS)
	$(NVCC) $(FLAGS) -shared -Xcompiler -fPIC $< -o $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
