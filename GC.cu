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
  manyDots<<<qtGqgridSize,qtGqblockSize>>>(qtGq,q,Gq,numComps,numFreqs,numParticles);
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

gradient_vruj(float *gradient, cuComplex *G,int numComps,int numFreqs,int numParticles);
{
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
  const dim3 GqblockSize(blksize);
  int grdSize = (int)((numParticles*numFreqs+blksize-1)/blksize);
  const dim3 GqgridSize(grdSize);
  grdsize = (int)(numParticles*numFreqs+blksize-1)/blksize;
  const dim3 qtGqblockSize(blksize);
  const dim3 qtGqgridSize(grdsize);
  
  
  
  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  // Compute Gq - This is mixing real and complex and cublas does not like that.
  // The same problem is in the GC function above.
  const dim3 qConvertBlockSize(blksize);
  grdsize = (int)((numParticles*numComps+blksize-1)/blksize);
  const dim3 qConvertGridSize(grdsize);
  make_q_complex<<<qConvertBlockSize,qConvertGridSize>>>(qcmplx,q,numComps,numParticles);
  // I might just want to write this whole thing out. 
  cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,numComps*numFreqs,numParticles,numComps,
	      &alpha,Tf,numComps*numFreqs,qmplx,numComps,
	      &beta,Gq,numComps*numFreqs);
  // Compute G*Gq
  
  // Compute the norm of Gq
  norm2Gq<<<GqblockSize,GqgridSize>>>(Gq2,Gq,numComps,numFreqs,numParticles);
  
  // Compute qtGq   
  manyDots<<<qtGqgridSize,qtGqblockSize>>>(qtGq,q,Gq,numComps,numFreqs,numParticles);
  // Take the absolute value (so that the result is a float)

  


