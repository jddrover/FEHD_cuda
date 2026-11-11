NVCC = /usr/local/cuda/bin/nvcc
LIBS = -lblas -llapacke -lcublas -lcusolver -lfftw3 -lstdc++fs

# -O3              optimise host (CPU) code too; nvcc does not do this by default
# -arch=native     compile GPU code for the GPU in this machine (CUDA 11.6 or newer)
#                  older CUDA: use e.g. -arch=sm_86 for your card instead
# -lineinfo        lets profilers map time back to source lines; no effect on speed
NVFLAGS = -O3 -arch=native -lineinfo

FEHDvruj: driver_new.cpp FEHD.cpp GC.cu kernels_vruj.cu kernels.cu mkARGPU.cu utility.cpp workArray.cu dataCompute.cpp dataManip.cpp
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LIBS)
