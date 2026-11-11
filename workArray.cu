#include "workArray.h"
#include "utility.h"
#include <vector>
void allocateParams(workForGranger &workArray,int numComps,int particleBlockSize,paramContainer params)
{
  int numFreqs = params.numFreqs;
  int numLags = params.numLags;
  cudaMalloc((void**)&workArray.q_dev, sizeof(float)*numComps*particleBlockSize);
  cudaMalloc((void**)&workArray.q_tmp, sizeof(float)*numComps*particleBlockSize);
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
  cudaMalloc((void**)&workArray.delF_proj_dev,sizeof(float)*numComps*numFreqs*particleBlockSize);
  cudaMalloc((void**)&workArray.GstarGq_dev,sizeof(cuComplex)*numFreqs*numComps*particleBlockSize);
  return;
}

void allocateParams_v2(workForGranger &workArray,int numComps,int particleBlockSize,
                       paramContainer params,int numSteps)
{
  // Only what version 2 needs. Nothing here is numComps x numFreqs x numParticles except
  // Y and Z (3*C*F*P floats each); version 1 allocates about 10*C*F*P floats.
  // sizeof(...) is a size_t, so every product below is computed in 64-bit.
  const int numFreqs = params.numFreqs;
  const int numLags  = params.numLags;
  const int P = particleBlockSize;
  cudaMalloc((void**)&workArray.q_dev,        sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.q_tmp,        sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.gradient_dev, sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.delF_proj_dev,sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.GCvals_dev,   sizeof(float)*P);
  cudaMalloc((void**)&workArray.G_dev,        sizeof(cuComplex)*numComps*numComps*numFreqs);
  cudaMalloc((void**)&workArray.AR_dev,       sizeof(float)*numComps*numComps*numLags);
  cudaMalloc((void**)&workArray.lagList_dev,  sizeof(int)*numLags);

  cudaMalloc((void**)&workArray.K_dev,        sizeof(float)*3*numComps*numComps*numFreqs);
  cudaMalloc((void**)&workArray.Y_dev,        sizeof(float)*3*numComps*numFreqs*P);
  cudaMalloc((void**)&workArray.Z_dev,        sizeof(float)*3*numComps*numFreqs*P);
  cudaMalloc((void**)&workArray.qq_dev,       sizeof(float)*3*numFreqs*P);
  cudaMalloc((void**)&workArray.qd_dev,       sizeof(float)*3*numFreqs*P);
  cudaMalloc((void**)&workArray.dd_dev,       sizeof(float)*3*numFreqs*P);
  cudaMalloc((void**)&workArray.steps_dev,    sizeof(float)*numSteps);
  return;
}

void allocateParams_v3(workForGranger &workArray,int numComps,int particleBlockSize,
                       paramContainer params,int paddedComps)
{
  // Version 3 stores nothing per (particle, frequency, component). The largest buffer is
  // qq: 3 floats per (particle, frequency).
  const int numFreqs = params.numFreqs;
  const int numLags  = params.numLags;
  const int P = particleBlockSize;
  cudaMalloc((void**)&workArray.q_dev,        sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.q_tmp,        sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.delF_proj_dev,sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.GCvals_dev,   sizeof(float)*P);
  cudaMalloc((void**)&workArray.G_dev,        sizeof(cuComplex)*numComps*numComps*numFreqs);
  cudaMalloc((void**)&workArray.AR_dev,       sizeof(float)*numComps*numComps*numLags);
  cudaMalloc((void**)&workArray.lagList_dev,  sizeof(int)*numLags);
  cudaMalloc((void**)&workArray.K_dev,        sizeof(float)*3*paddedComps*paddedComps*numFreqs);
  cudaMalloc((void**)&workArray.qq_dev,       sizeof(float)*3*numFreqs*P);
  cudaMalloc((void**)&workArray.bestGC_dev,   sizeof(float)*P);
  cudaMalloc((void**)&workArray.bestQ_dev,    sizeof(float)*numComps*P);
  cudaMalloc((void**)&workArray.recycleCount_dev, sizeof(int));
  return;
}

void freeWorkArray(workForGranger &workArray)
{
  cudaFree(workArray.q_dev);
  cudaFree(workArray.q_tmp);
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
  cudaFree(workArray.delF_proj_dev);
  cudaFree(workArray.GstarGq_dev);

  cudaFree(workArray.K_dev);
  cudaFree(workArray.Y_dev);
  cudaFree(workArray.Z_dev);
  cudaFree(workArray.qq_dev);
  cudaFree(workArray.qd_dev);
  cudaFree(workArray.dd_dev);
  cudaFree(workArray.steps_dev);
  cudaFree(workArray.bestGC_dev);
  cudaFree(workArray.bestQ_dev);
  cudaFree(workArray.recycleCount_dev);

  workArray = workForGranger();   // reset every pointer to nullptr
  return;
}
