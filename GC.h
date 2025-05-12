#ifndef GC_H
#define GC_H
#include "dataContainers.h"
#include <vector>
#include "utility.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "workArray.h"

struct particleObject {
  std::vector<float> location;
  float value;
};

void granger(std::vector<float>, std::vector<float> &, paramContainer, int,workForGranger workArray);
void runFEHDstep(std::vector<float> &, matrix &, dataList, paramContainer, int);
void compGradient(std::vector<float> &, std::vector<float>,std::vector<float>,paramContainer,int,workForGranger workArray);
void computeBlocks(int &,int &,size_t,paramContainer,int);
void PSOstep(std::vector<particleObject> &,std::vector<particleObject> &, particleObject &,paramContainer,int,int,workForGranger);
#endif
