NVCC = /usr/local/cuda/bin/nvcc
LIBS = -lblas -llapacke -lcublas -lcusolver -lstdc++fs
FEHD: driver_new.cpp FEHD.cpp GC.cu kernels.cu mkARGPU.cu timeSeriesOPs.cpp utility.cpp
	$(NVCC) $^ -o $@ $(LIBS)
