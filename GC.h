#ifndef GC_H
#define GC_H
#include "dataContainers.h"
#include <vector>
#include "utility.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "workArray.h"

void granger(std::vector<float>, std::vector<float> &, paramContainer, int,workForGranger workArray);
void runFEHDstep(std::vector<float> &, matrix &, dataList, paramContainer, int);
void compGradient(std::vector<float> &, std::vector<float>,std::vector<float>,paramContainer,int,workForGranger workArray);
void computeBlocks(int &,int &,size_t,paramContainer,int);
#endif
