#include "GC.h"
//#include "dataContainers.h"
#include "utility.h"
//#include "timeSeriesOPs.h"
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
//#include "kernels.h"
#include "kernels_vruj.h"
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

void granger_vruj(workForGranger wa,int numComps, int numFreqs,int numParticles)
{
  // Input q is numComps x numParticles - on the device
  // Input Tf is (numComps x numFreqs) x numComps - on the device
  // Output GCvals is numParticles - on the device
  // worker Gq is numFreqs x numComps x numParticles
  // worker qtGq is numFreqs x numParticles
  
  int blksize = 1024;
  
  int grdsize = (int)((numParticles*numFreqs*numComps+blksize-1)/blksize);
  const dim3 CFP_block(blksize);
  const dim3 CFP_grid(grdsize);
  grdsize = (int)((numParticles+blksize-1)/blksize);
  const dim3 P_block(blksize);
  const dim3 P_grid(grdsize);
  grdsize = (int)((numParticles*numFreqs+blksize-1)/blksize);
  const dim3 FP_block(blksize);
  const dim3 FP_grid(grdsize);

  
  makeGq<<<CFP_grid,CFP_block>>>(wa.Gq_dev,wa.G_dev,wa.q_dev,numComps,numFreqs,numParticles);

  norm2Gq<<<FP_grid,FP_block>>>(wa.Gq2_dev,wa.Gq_dev,numComps,numFreqs,numParticles);
  
  // Compute qtGq   
  qTGq<<<FP_grid,FP_block>>>(wa.s_dev,wa.q_dev,wa.Gq_dev,numComps,numFreqs,numParticles);
  // Take the absolute value (so that the result is a float)
  magnitude2<<<FP_grid,FP_block>>>(wa.absqtGq_dev,wa.s_dev,numFreqs,numParticles);  
  // Compute the log(Gq2/qtGq)
  computeGC<<<P_grid,P_block>>>(wa.GCvals_dev,wa.Gq2_dev,wa.absqtGq_dev,numFreqs,numParticles);

  return;
}

void gradient_vruj(workForGranger wa,int numComps,int numFreqs,int numParticles)
{

  // numComps * numFreqs * numParticles
  int blksize = 1024;
  const dim3 CFP_block(blksize);
  int grdSize = (int)((numParticles*numFreqs*numComps+blksize-1)/blksize);
  const dim3 CFP_grid(grdSize);

  // numParticles*numFreqs
  const dim3 FP_block(blksize);
  grdSize = (int)((numFreqs*numParticles+blksize-1)/blksize);
  const dim3 FP_grid(grdSize);

  // (numComps*numFreqs)*numComps
  const dim3 CCF_block(blksize);
  grdSize = (int)((numFreqs*numComps*numComps+blksize-1)/blksize);
  const dim3 CCF_grid(grdSize);

  grdSize = (int)((numComps*numParticles+blksize-1)/blksize);
  const dim3 CP_block(blksize);
  const dim3 CP_grid(grdSize);

  // Compute Gq
  makeGq<<<CFP_grid,CFP_block>>>(wa.Gq_dev,wa.G_dev,wa.q_dev,numComps,numFreqs,numParticles);
  
  // Compute the norm of Gq
  norm2Gq<<<FP_grid,FP_block>>>(wa.Gq2_dev,wa.Gq_dev,numComps,numFreqs,numParticles);
  
  // Compute (G*)Gq
  GstarGq<<<CFP_grid,CFP_block>>>(wa.GsGq_dev,wa.G_dev,wa.Gq_dev,numComps,numFreqs,numParticles);

  // Compute s 
  qTGq<<<FP_grid,FP_block>>>(wa.s_dev,wa.q_dev,wa.Gq_dev,numComps,numFreqs,numParticles);

  // Compute Gs=(G^T+G)/2
  GTplusG<<<CCF_grid,CCF_block>>>(wa.Gs_dev,wa.G_dev,numComps,numFreqs);

  // Compute Gsq
  makeGq<<<CFP_grid,CFP_block>>>(wa.Gsq_dev,wa.Gs_dev,wa.q_dev,numComps,numFreqs,numParticles);

  calcGradient<<<CFP_grid,CFP_block>>>(wa.delF_dev,wa.Gsq_dev,wa.s_dev,wa.Gq2_dev,wa.GsGq_dev,wa.q_dev,
  				       numComps,numFreqs,numParticles);

  delgF<<<CP_grid,CP_block>>>(wa.q_tmp,wa.delF_dev,numComps,numFreqs,numParticles);
  
  projectQorth<<<CP_grid,CP_block>>>(wa.delF_proj_dev,wa.q_dev,wa.q_tmp,numComps,numParticles);

  cudaDeviceSynchronize();
  return;
}

void runFEHDstep(std::vector<float> &bestAngle, std::vector<float> &L,
		 dataClass<float> dataArray,paramContainer params,int numComps)
{
  int id;
  //size_t freemem,total;
  cudaGetDevice(&id); // I am not sure where I use this.
  // Random seed
  srand((unsigned)time(0));
  // Create MVAR model for the data using the lags in params.
  MVAR<float> model = mkARGPU(dataArray,params);
  

  // Some abbreviations
  int maxLag = *std::max_element(model.lagList.begin(),model.lagList.end());
  int epochAdj = dataArray.getEpochPoints() - maxLag;
  // Put the residuals in a container and get the PC transformation matrix,
  // use it to rotate the model to have orthonormal residuals
  // (which there is no need to compute). 
  dataClass<float> residuals(epochAdj,dataArray.getNumComps(),model.R,dataArray.getSampRate());
  std::vector<float> Dmat = PCA(residuals);
  
  MVAR<float> rModel = rotate_model(model,Dmat);
  L.resize(Dmat.size());
  std::copy(Dmat.begin(),Dmat.end(),L.begin());

  workForGranger wa;
  allocateParams(wa,numComps,params.numParticles,params);

  // From here, I would like to create the transfer function
  int blksize = 768;
  int gridsize = (int)((numComps*numComps*params.numFreqs+blksize-1)/blksize);
  const dim3 CCF_block(blksize);
  const dim3 CCF_grid(gridsize);

  cudaMemcpy(wa.AR_dev,rModel.A.data(),sizeof(float)*params.numLags*numComps*numComps,cudaMemcpyHostToDevice);
  cudaMemcpy(wa.lagList_dev,rModel.lagList.data(),sizeof(int)*params.numLags,cudaMemcpyHostToDevice);
  
  float dt = 1.0/float(params.sampRate);

  // Compute the array of transfer functions
  G_vruj<<<CCF_grid,CCF_block>>>(wa.G_dev,wa.AR_dev,wa.lagList_dev,numComps,params.freqLo,
	    params.freqHi,params.numFreqs,params.numLags,dt);
  cudaDeviceSynchronize();
  // random q's
  std::vector<float> q_host(params.numParticles*numComps);
  for(int k=0;k<numComps*params.numParticles;++k)
    q_host[k] = (float)((rand()%2000)/1000.0f-1.0f );

  
  cudaMemcpy(wa.q_tmp,q_host.data(),sizeof(float)*numComps*params.numParticles,cudaMemcpyHostToDevice);

  gridsize = (int)((numComps*params.numParticles+blksize-1)/blksize);
  const dim3 CP_block(blksize);
  const dim3 CP_grid(gridsize);

  normalize_q<<<CP_grid,CP_block>>>(wa.q_dev,wa.q_tmp,numComps,params.numParticles);


  std::vector<float> q_host_tmp(numComps*params.numParticles);
  cudaMemcpy(q_host_tmp.data(),wa.q_dev,sizeof(float)*numComps*params.numParticles,
  	     cudaMemcpyDeviceToHost);

  std::vector<float> p = {0.0,0.0001,0.01,0.05,0.5};

  // Make a GC table numPoint x numParticles
  float *GCtable;
  cudaMalloc((void**)&GCtable,sizeof(float)*p.size()*params.numParticles);
  float *qStore;
  cudaMalloc((void**)&qStore,sizeof(float)*p.size()*params.numParticles*numComps);
  
  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  gridsize = (int)((params.numParticles+blksize-1)/blksize);
  const dim3 P_block(blksize);
  const dim3 P_grid(gridsize);
  
  gradient_vruj(wa,numComps,params.numFreqs,params.numParticles);
  
  std::vector<float> diffF(params.numParticles);

  float *q_base;
  cudaMalloc((void**)&q_base,sizeof(float)*numComps*params.numParticles);

  int numIts = 100;
  int minIndx;
  int storeIndx;
  float val;
  float val2;
  for(int iter = 0;iter<numIts;iter++)
    {      
      cublasScopy(cublasH,numComps*params.numParticles,wa.q_dev,1,q_base,1);
      cudaDeviceSynchronize();
      gradient_vruj(wa,numComps,params.numFreqs,params.numParticles);
      
      for(int piter=0;piter<p.size();piter++)
	{
	  float alpha = -p[piter];

	  // Copy, move, scale
	  cublasScopy(cublasH,numComps*params.numParticles,q_base,1,wa.q_tmp,1);
	  cublasSaxpy(cublasH,numComps*params.numParticles,&alpha,wa.delF_proj_dev,1,wa.q_tmp,1);
	  normalize_q<<<CP_grid,CP_block>>>(wa.q_dev,wa.q_tmp,numComps,params.numParticles);
	  granger_vruj(wa,numComps,params.numFreqs,params.numParticles);
	  cublasScopy(cublasH,numComps*params.numParticles,
		      wa.q_dev,1,qStore+piter*numComps*params.numParticles,1);
	  cublasScopy(cublasH,params.numParticles,wa.GCvals_dev,1,GCtable+piter,p.size());
	  cudaDeviceSynchronize();
	}
  
      parseTable<<<P_grid,P_block>>>(wa.q_dev,GCtable,qStore,numComps,params.numParticles,p.size());
      cublasIsamin(cublasH,params.numParticles,GCtable,p.size(),&minIndx);
      minIndx = minIndx - 1; // Claude's first win, if it works.
      cudaMemcpy(&val,GCtable+p.size()*minIndx,sizeof(float),cudaMemcpyDeviceToHost);
      std::cout << "Iteration: " << iter << " Particle: " << minIndx <<
	" Value: " << val << std::endl;
    }

  // Determine the minimum granger causality value
  cublasIsamin(cublasH,params.numParticles,GCtable,p.size(),&minIndx);
  minIndx = minIndx - 1; // Claude's first win, if it works.
  
  std::vector<float> q_best(numComps);
  cudaMemcpy(q_best.data(),wa.q_dev+minIndx*numComps,sizeof(float)*numComps,
	     cudaMemcpyDeviceToHost);
  // bookmark  
  std::cout << "Minimum Value = " << val << " particle = " << minIndx << std::endl;
  
  


  
  exit(0);
  return;
}
  


