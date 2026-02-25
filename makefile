NVCC = /usr/local/cuda/bin/nvcc
LIBS = -lblas -llapacke -lcublas -lcusolver -lfftw3 -lstdc++fs
FEHDSVD: driver_new.cpp FEHD.cpp GC.cu kernels.cu mkARGPU.cu timeSeriesOPs.cpp utility.cpp workArray.cu dataCompute.cpp dataManip.cpp
	$(NVCC) -Wno-deprecated-gpu-targets $^ -o $@ $(LIBS)
