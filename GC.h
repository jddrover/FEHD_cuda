#ifndef GC_H
#define GC_H
#include <vector>
#include "utility.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "workArray.h"
#include "dataClass.h"

void granger(std::vector<float>, std::vector<float> &, paramContainer, int,workForGranger workArray);
void runFEHDstep(std::vector<float> &, std::vector<float> &, dataClass<float>, paramContainer, int);
void compGradient(std::vector<float> &, std::vector<float>,std::vector<float>,paramContainer,int,workForGranger workArray);
void computeBlocks(int &,int &,size_t,paramContainer,int);
#endif
