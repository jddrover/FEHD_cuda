#ifndef WORKARRAY_H
#define WORKARRAY_H
#include "utility.h"
#include <vector>
#include <cuComplex.h>

struct workForGranger
{
  float *q_dev;
  cuComplex *G_dev;
  cuComplex *s_dev;
  float *absqtGq_dev;
  cuComplex *Gq_dev;
  float *Gq2_dev;
  float *GCvals_dev;
  float *AR_dev;
  float *gradient_dev;
  cuComplex *GsGq_dev;
  cuComplex *Gs_dev;
  cuComplex *Gsq_dev;
  float *delF_dev;
  cuComplex *GstarGq_dev;
  int *lagList_dev;
  
};

//void allocateParams(workForGranger &,int numComps,int particleBlockSize,paramContainer params,std::vector<int> lagList,std::vector<float> AR);
void allocateParams(workForGranger &,int numComps,int particleBlockSize,paramContainer params);
void freeWorkArray(workForGranger &);
#endif
