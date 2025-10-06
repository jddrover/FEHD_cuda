#ifndef WORKARRAY_H
#define WORKARRAY_H
#include "utility.h"
#include <vector>
#include <complex>

struct workForGranger
{
  int lworkVal;
  float2 *Qdev;
  float2 *rotatedModels;
  float2 *wArray;
  float2 *Tf;
  float2 *Swhole;
  float2 *tmp;
  float2 *Spartial; // Make partial one size down
  float2 *d_wholeSpec;
  float *dev_W;
  int *d_info;
  float2 *d_work2;
  float *det_whole;
  float *det_partial;
  float *dev_GC;
  int *lagList_DEVICE;
  float2 *ARdev;
};

void allocateParams(workForGranger &,int numComps,int particleBlockSize,paramContainer params,std::vector<int> lagList,std::vector<std::complex<float>>);
void freeWorkArray(workForGranger &);
#endif
