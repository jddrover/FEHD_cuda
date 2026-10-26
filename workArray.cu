#include "workArray.h"
#include "utility.h"
#include <vector>
void allocateParams(workForGranger &workArray,int numComps,int particleBlockSize,paramContainer params)
{
  int numFreqs = params.numFreqs;
  int numLags = params.numLags;
  cudaMalloc((void**)&workArray.q_dev, sizeof(float)*numComps*particleBlockSize);
  cudaMalloc((void**)&workArray.G_dev, sizeof(cuComplex)*numComps*numComps*numFreqs);
  cudaMalloc((void**)&workArray.s_dev, sizeof(cuComplex)*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.absqtGq_dev, sizeof(float)*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.Gq_dev, sizeof(cuComplex)*numComps*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.Gq2_dev, sizeof(float)*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.GCvals_dev, sizeof(float)*particleBlockSize);
  cudaMalloc((void**)&workArray.AR_dev,sizeof(float)*numComps*numComps*numLags);
  cudaMalloc((void**)&workArray.lagList_dev,sizeof(int)*numLags);
  cudaMalloc((void**)&workArray.gradient_dev,sizeof(float)*numComps*particleBlockSize);
  cudaMalloc((void**)&workArray.GsGq_dev,sizeof(cuComplex)*numComps*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.Gs_dev,sizeof(cuComplex)*numComps*numFreqs*numComps);
  cudaMalloc((void**)&workArray.Gsq_dev,sizeof(cuComplex)*numComps*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.delF_dev,sizeof(float)*numComps*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.GstarGq_dev,sizeof(cuComplex)*numFreqs*numComps*particleBlockSize);
  return;
}

void freeWorkArray(workForGranger &workArray)
{
  cudaFree(workArray.q_dev);
  cudaFree(workArray.G_dev);
  cudaFree(workArray.s_dev);
  cudaFree(workArray.absqtGq_dev);
  cudaFree(workArray.Gq_dev);
  cudaFree(workArray.Gq2_dev);
  cudaFree(workArray.GCvals_dev);
  cudaFree(workArray.AR_dev);
  cudaFree(workArray.lagList_dev);
  cudaFree(workArray.gradient_dev);
  cudaFree(workArray.GsGq_dev);
  cudaFree(workArray.Gs_dev);
  cudaFree(workArray.Gsq_dev);
  cudaFree(workArray.delF_dev);
  cudaFree(workArray.GstarGq_dev);
  
  
  return;
}
