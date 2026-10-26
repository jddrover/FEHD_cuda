#ifndef WORKARRAY_H
#define WORKARRAY_H
#include "utility.h"
#include <vector>
#include <cuComplex.h>

struct workForGranger
{
  float *q_dev; // CP*4
  float *q_tmp; // CP*4
  cuComplex *G_dev; // CCF*8
  cuComplex *s_dev; // FP*8
  float *absqtGq_dev; // FP*4
  cuComplex *Gq_dev; // CFP*8
  float *Gq2_dev; // FP*4
  float *GCvals_dev; // P*4
  float *AR_dev; // CCL*4
  float *gradient_dev; // CFP*4
  cuComplex *GsGq_dev; // CFP*8
  cuComplex *Gs_dev; // CF*8
  cuComplex *Gsq_dev; // CFP*8
  float *delF_dev; // CFP*4
  float *delF_proj_dev; // CFP*4
  cuComplex *GstarGq_dev; // CFP*8
  int *lagList_dev; // L
  
};

//void allocateParams(workForGranger &,int numComps,int particleBlockSize,paramContainer params,std::vector<int> lagList,std::vector<float> AR);
void allocateParams(workForGranger &,int numComps,int particleBlockSize,paramContainer params);
void freeWorkArray(workForGranger &);
#endif
