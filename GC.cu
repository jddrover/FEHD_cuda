#include "GC.h"
//#include "dataContainers.h"
#include "utility.h"
//#include "timeSeriesOPs.h"
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "kernels.h"
#include <complex>
#include <chrono>
#include <cblas.h>
#include <algorithm>
#include "mkARGPU.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "workArray.h"
#include "dataClass.h"
#include "dataCompute.h"
#include <cuComplex.h>

void granger_vruj(float *q, cuComplex *Tf ,float *GCvals,int numComps, int numFreqs,int numParticles)
{
  // Input q is numComps x numParticles - on the device
  // Input Tf is (numComps x numFreqs) x numComps - on the device
  // Output GCvals is numParticles - on the device
  // worker Gq is numFreqs x numComps x numParticles
  // worker qtGq is numFreqs x numParticles

  // These creations should take place outside of the loop containing this function.
  // Once this works I will move them. 
  cuComplex *qtGq;
  cudaMalloc((void**)&qtGq,sizeof(cuComplex)*numFreqs*numParticles);
  float *absqtGq;
  cudaMalloc((void**)&absqtGq,sizeof(float)*numFreqs*numParticles);
  cuComplex *Gq;
  cudaMalloc((void**)&Gq,sizeof(cuComplex)*numFreqs*numParticles*numComps);
  float *Gq2;
  cudaMalloc((void**)&Gq2,sizeof(float)*numFreqs*numParticles);
  float *GCvals;
  cudaMalloc((void**)&GCvals,sizeof(float)*numParticles);

  
  int blksize = 1024;

  grdsize = (int)(numParticles*numFreqs+blksize-1)/blksize;
  const dim3 qtGqblockSize(blksize);
  const dim3 qtGqgridSize(grdsize);
  

  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);
  
  
  // Compute Gq
  const dim3 GqblockSize(blksize);
  grdSize = (int)((numParticles*numFreqs*numComps+blksize-1)/blksize);
  const dim3 GqgridSize(grdSize);
  makeGq<<<GqblockSize,GqgridSize>>>(Gq,G,q,numComps,numFreqs,numParticles);

  // Compute the norm of Gq
  const dim3 Gq2blockSize(blksize);
  grdsize = (int)((numParticles*numFreqs+blksize-1)/blksize);
  const dim3 Gq2gridSize(grdsize);
  norm2Gq<<<Gq2blockSize,Gq2gridSize>>>(Gq2,Gq,numComps,numFreqs,numParticles);
  
  // Compute qtGq   
  qTGq<<<qtGqgridSize,qtGqblockSize>>>(qtGq,q,Gq,numComps,numFreqs,numParticles);
  // Take the absolute value (so that the result is a float)
  magnitude<<<qtGqBlockSize,qtGqGridSize>>>(absqtGq,qtGq,numFreqs,numParticles);  
  // Compute the log(Gq2/qtGq)
  computeGC<<<cGCgridSize,cGCblockSize>>>(GCvals,Gq2,absqtGq,numFreqs,numParticles);

  // Do a copy here, but it is only temporary. Everything will be moved out so
  // everything stays on the GPU for the duration of the "peel".
  cudaFree(qtGq);
  cudaFree(absqtGq);
  cudaFree(Gq);
  cudaFree(Gq2);
  cudaFree(GCvals);

}

gradient_vruj(float *gradient, cuComplex *G,float *q,int numComps,int numFreqs,int numParticles);
{
  cuComplex *s;
  cudaMalloc((void**)&s,sizeof(cuComplex)*numFreqs*numParticles);
  float *absqtGq;
  cudaMalloc((void**)&absqtGq,sizeof(float)*numFreqs*numParticles);
  cuComplex *Gq;
  cudaMalloc((void**)&Gq,sizeof(cuComplex)*numFreqs*numParticles*numComps);
  float *Gq2;
  cudaMalloc((void**)&Gq2,sizeof(float)*numFreqs*numParticles);
  float *GCvals;
  cudaMalloc((void**)&GCvals,sizeof(float)*numParticles);

  // numComps * numFreqs * numParticles
  int blksize = 1024;
  const dim3 CFP_block(blksize);
  int grdSize = (int)((numParticles*numFreqs*numComps+blksize-1)/blksize);
  const dim3 CFP_grid(grdSize);

  // numParticles*numFreqs
  const dim3 FP_block(blksize);
  grdsize = (int)((numFreqs*numParticles+blksize-1)/blksize);
  const dim3 FP_grid(grdsize);

  // (numComps*numFreqs)*numComps
  const dim3 CCF_block(blksize);
  grdsize = (int)((numFreqs*numComps*numComps+blksize-1)/blksize);
  const dim3 CCF_grid(grdsize);


  // cublas handle
  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  // Compute Gq
  makeGq<<<CFP_grid,CFP_block>>>(Gq,G,q,numComps,numFreqs,numParticles);
  
  // Compute the norm of Gq
  norm2Gq<<<FP_grid,FP_block>>>(Gq2,Gq,numComps,numFreqs,numParticles);
  
  // Compute (G*)Gq
  GstarGq<<<CFP_grid,CFP_block>>>(GsGq,G,Gq,numComps,numFreqs,numParticles);

  // Compute conj(s) 
  conjS<<<FP_grid,FP_block>>>(sconj,q,Gq,numComps,numFreqs,numParticles);
  // |s|^2
  magnitude2<<<FP_grid,FP_block>>>(smag2,sconj,numFreqs,numParticles);

  // Compute (G^T+G)/2
  GTplusG<<<CCF_grid,CCF_block>>>(GTpGd2,G,numComps,numFreqs);
  


