#ifndef GC_H
#define GC_H
#include "dataContainers.h"
#include <vector>
#include "utility.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>

void granger(float *, std::vector<float>, std::vector<float> &, paramContainer, int, int *,float *,float *,float *,float2 *,float2 *,
	     float2 *,float2 *,float2 *,float *,int *,int &,float2 *,float *,float *,float *);
void runFEHDstep(std::vector<float> &, matrix &, dataList, paramContainer, int);
void compGradient(float *, std::vector<float> &, std::vector<float>,std::vector<float>,paramContainer,int,int *,float *,float *,float *,float2 *,float2 *,
		  float2 *,float2 *,float2 *, float *,int *,int,float2 *,float *,float *,float *);
void computeBlocks(int &,int &,size_t,paramContainer,int);
#endif
