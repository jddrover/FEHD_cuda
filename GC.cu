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
  
  int blksize = 768;
  
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
  int blksize = 768;
  const dim3 CFP_block(blksize);
  int grdSize = (int)((numParticles*numFreqs*numComps+blksize-1)/blksize);
  const dim3 CFP_grid(grdSize);

  // numParticles*numFreqs
  const dim3 FP_block(blksize);
  grdSize = (int)((numFreqs*numParticles+blksize-1)/blksize);
  const dim3 FP_grid(grdSize);

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

  // Gs=(G^T+G)/2 is now computed once in runFEHDstep, right after G_vruj,
  // because G does not change between gradient calls.

  // Compute Gsq
  makeGq<<<CFP_grid,CFP_block>>>(wa.Gsq_dev,wa.Gs_dev,wa.q_dev,numComps,numFreqs,numParticles);

  calcGradient<<<CFP_grid,CFP_block>>>(wa.delF_dev,wa.Gsq_dev,wa.s_dev,wa.Gq2_dev,wa.GsGq_dev,wa.q_dev,
  				       numComps,numFreqs,numParticles);

  delgF<<<CP_grid,CP_block>>>(wa.q_tmp,wa.delF_dev,numComps,numFreqs,numParticles);
  
  projectQorth<<<CP_grid,CP_block>>>(wa.delF_proj_dev,wa.q_dev,wa.q_tmp,numComps,numParticles);

  // No cudaDeviceSynchronize needed: everything here and in the caller runs on the default
  // stream, so the GPU executes it in the order it was issued.
  return;
}

void runFEHDstep(std::vector<float> &q_best, std::vector<float> &L,
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
  // Gs=(G^T+G)/2, needed by every gradient call. G is fixed, so compute it once here.
  GTplusG<<<CCF_grid,CCF_block>>>(wa.Gs_dev,wa.G_dev,numComps,params.numFreqs);
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
  
  // (Removed a gradient_vruj call here: the first loop iteration recomputed it straight away.)
  
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
	}
  
      parseTable<<<P_grid,P_block>>>(wa.q_dev,GCtable,qStore,numComps,
				     params.numParticles,p.size());
      if(params.verbose)
	{
	  cublasIsamin(cublasH,params.numParticles,GCtable,p.size(),&minIndx);
	  minIndx = minIndx - 1; // Claude's first win, if it works.
	  cudaMemcpy(&val,GCtable+p.size()*minIndx,sizeof(float),cudaMemcpyDeviceToHost);
	  std::cout << "Iteration: " << iter << " Particle: " << minIndx <<
	    " Value: " << val << std::endl;
	}
    }

  // Determine the minimum granger causality value
  cublasIsamin(cublasH,params.numParticles,GCtable,p.size(),&minIndx);
  minIndx = minIndx - 1; // Claude's first win, if it works.
  
  
  cudaMemcpy(q_best.data(),wa.q_dev+minIndx*numComps,sizeof(float)*numComps,
	     cudaMemcpyDeviceToHost);

  cublasDestroy(cublasH);   // the handle was created on every call but never released
  freeWorkArray(wa);
  cudaFree(q_base);
  cudaFree(qStore);
  cudaFree(GCtable);
  //cudaMemcpy(&val,GCtable+p.size()*minIndx,sizeof(float),cudaMemcpyDeviceToHost);
  // bookmark  
  //std::cout << "Minimum Value = " << val << " particle = " << minIndx << std::endl;
    
  //exit(0);
  return;
}
  



// ===========================================================================================
//  Version 2: same algorithm, reformulated with the stacked real matrix K = [M_f; Sr_f; Si_f].
//  See the comment block above buildSblocks in kernels_vruj.cu for the identities and layout.
//
//  Per iteration: two cublasSgemm calls (K*q and K*d) of 3*C^2 multiply-adds per
//  (particle, frequency) each, plus kernels that are linear in numComps.
//  Version 1 does about 18*C^2 per (particle, frequency) per iteration.
// ===========================================================================================

static void buildK_v2(cublasHandle_t h, workForGranger &wa, int numComps, int numFreqs)
{
  // M_f = Re(G_f^H G_f).
  // Viewed as plain floats, G_dev is a real column-major matrix with 2*C*F rows and C columns:
  // within each column the rows run over (frequency, row, real/imag). For frequency f, rows
  // [2Cf, 2Cf+2C) form a 2C x C real matrix X_f whose rows alternate real/imag parts, and
  // X_f^T X_f = Re(G_f^H G_f). cublasSgemmStridedBatched does all frequencies at once and
  // writes M_f straight into the M rows of K.
  const float one = 1.0f, zero = 0.0f;
  const int ldG = 2*numComps*numFreqs;
  const int ldK = 3*numComps*numFreqs;
  const float *Gf = reinterpret_cast<const float*>(wa.G_dev);
  cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N,
                            numComps, numComps, 2*numComps,
                            &one,
                            Gf,       ldG, 2*numComps,   // A = X_f   (transposed by OP_T)
                            Gf,       ldG, 2*numComps,   // B = X_f
                            &zero,
                            wa.K_dev, ldK, 3*numComps,   // C = M rows of frequency f in K
                            numFreqs);

  // Sr_f and Si_f, the real and imaginary parts of (G_f^T + G_f)/2.
  const int blk = 256;
  const int grd = (numComps*numFreqs*numComps + blk - 1)/blk;
  buildSblocks<<<grd,blk>>>(wa.K_dev, wa.G_dev, numComps, numFreqs);
}

static void applyK_v2(cublasHandle_t h, const workForGranger &wa, float *out, const float *x,
                      int numComps, int numFreqs, int numParticles)
{
  // out = K * x, with x a C x P column-major matrix (x[p*C + c], same layout as q).
  // Result: for each (p, f) the vector [M_f x_p; Sr_f x_p; Si_f x_p] at out[(p*F+f)*3C].
  const float one = 1.0f, zero = 0.0f;
  const int ldK = 3*numComps*numFreqs;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
              ldK, numParticles, numComps,
              &one,
              wa.K_dev, ldK,
              x,        numComps,
              &zero,
              out,      ldK);
}

static void launchBlockDots(float *out, const float *x, const float *Y,
                            int numComps, int numFreqs, int numParticles)
{
  const int blk = 256;                     // 8 warps per block, one warp per (particle, freq)
  const int warpsPerBlock = blk/32;
  const int grd = (numFreqs*numParticles + warpsPerBlock - 1)/warpsPerBlock;
  blockDots<<<grd,blk>>>(out, x, Y, numComps, numFreqs, numParticles);
}

static void iterate_v2(cublasHandle_t h, workForGranger &wa, int numSteps,
                       int numComps, int numFreqs, int numParticles)
{
  const int blk = 256;
  const int CP_grid = (numComps*numParticles + blk - 1)/blk;
  const int P_grid  = (numParticles + blk - 1)/blk;

  // 1. Y = K q  ->  Mq, Srq, Siq for every (particle, frequency)
  applyK_v2(h, wa, wa.Y_dev, wa.q_dev, numComps, numFreqs, numParticles);
  // 2. q.Mq (= ||Gq||^2), q.Srq and q.Siq (= real and imaginary parts of s = q^T G q)
  launchBlockDots(wa.qq_dev, wa.q_dev, wa.Y_dev, numComps, numFreqs, numParticles);
  // 3. Gradient summed over frequency (replaces makeGq x2, GstarGq, calcGradient, delgF)
  gradFromBlocks<<<CP_grid,blk>>>(wa.gradient_dev, wa.Y_dev, wa.qq_dev,
                                  numComps, numFreqs, numParticles);
  // 4. Project onto the tangent space of the sphere: d = (I - q q^T) grad
  projectQorth<<<CP_grid,blk>>>(wa.delF_proj_dev, wa.q_dev, wa.gradient_dev,
                                numComps, numParticles);
  // 5. Z = K d  ->  Md, Srd, Sid
  applyK_v2(h, wa, wa.Z_dev, wa.delF_proj_dev, numComps, numFreqs, numParticles);
  // 6. q.Md, q.Srd, q.Sid  and  d.Md, d.Srd, d.Sid
  launchBlockDots(wa.qd_dev, wa.q_dev,         wa.Z_dev, numComps, numFreqs, numParticles);
  launchBlockDots(wa.dd_dev, wa.delF_proj_dev, wa.Z_dev, numComps, numFreqs, numParticles);
  // 7. Try every step size, keep the best, update q in place and store its GC value
  //    (replaces numSteps x granger_vruj + parseTable)
  lineSearchUpdate<<<P_grid,blk>>>(wa.q_dev, wa.GCvals_dev, wa.delF_proj_dev,
                                   wa.qq_dev, wa.qd_dev, wa.dd_dev, wa.steps_dev, numSteps,
                                   numComps, numFreqs, numParticles);
}

void runFEHDstep_v2(std::vector<float> &q_best, std::vector<float> &L,
                    dataClass<float> dataArray,paramContainer params,int numComps)
{
  // Same inputs and outputs as runFEHDstep.
  srand((unsigned)time(0));
  MVAR<float> model = mkARGPU(dataArray,params);

  int maxLag = *std::max_element(model.lagList.begin(),model.lagList.end());
  int epochAdj = dataArray.getEpochPoints() - maxLag;
  dataClass<float> residuals(epochAdj,dataArray.getNumComps(),model.R,dataArray.getSampRate());
  std::vector<float> Dmat = PCA(residuals);

  MVAR<float> rModel = rotate_model(model,Dmat);
  L.resize(Dmat.size());
  std::copy(Dmat.begin(),Dmat.end(),L.begin());

  const int numFreqs = params.numFreqs;
  const int numParticles = params.numParticles;

  // Step sizes for the line search (the same values as version 1). With version 2 extra
  // values cost very little, because each one only needs O(numFreqs) work per particle.
  std::vector<float> steps = {0.0f,0.0001f,0.01f,0.05f,0.5f};
  const int numSteps = (int)steps.size();

  workForGranger wa;
  allocateParams_v2(wa,numComps,numParticles,params,numSteps);
  cudaMemcpy(wa.steps_dev,steps.data(),sizeof(float)*numSteps,cudaMemcpyHostToDevice);

  cudaMemcpy(wa.AR_dev,rModel.A.data(),sizeof(float)*params.numLags*numComps*numComps,cudaMemcpyHostToDevice);
  cudaMemcpy(wa.lagList_dev,rModel.lagList.data(),sizeof(int)*params.numLags,cudaMemcpyHostToDevice);

  const int blksize = 256;
  const float dt = 1.0f/float(params.sampRate);
  const int CCF_grid = (numComps*numComps*numFreqs + blksize - 1)/blksize;
  G_vruj<<<CCF_grid,blksize>>>(wa.G_dev,wa.AR_dev,wa.lagList_dev,numComps,params.freqLo,
                               params.freqHi,numFreqs,params.numLags,dt);

  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  // K = [M; Sr; Si] for every frequency. Computed once; the loop below only reads it.
  buildK_v2(cublasH, wa, numComps, numFreqs);

  // Random starting q's, normalised to unit length
  std::vector<float> q_host(numParticles*numComps);
  for(int k=0;k<numComps*numParticles;++k)
    q_host[k] = (float)((rand()%2000)/1000.0f-1.0f );
  cudaMemcpy(wa.q_tmp,q_host.data(),sizeof(float)*numComps*numParticles,cudaMemcpyHostToDevice);
  const int CP_grid = (numComps*numParticles + blksize - 1)/blksize;
  normalize_q<<<CP_grid,blksize>>>(wa.q_dev,wa.q_tmp,numComps,numParticles);

  const int numIts = 100;
  int minIndx;
  float val;
  for(int iter = 0;iter<numIts;iter++)
    {
      iterate_v2(cublasH, wa, numSteps, numComps, numFreqs, numParticles);

      if(params.verbose)
        {
          // GCvals_dev holds each particle's GC after this iteration's step. GC >= 0,
          // so the smallest absolute value (Isamin) is the smallest value.
          cublasIsamin(cublasH,numParticles,wa.GCvals_dev,1,&minIndx);
          minIndx = minIndx - 1;   // cuBLAS indices start at 1
          cudaMemcpy(&val,wa.GCvals_dev+minIndx,sizeof(float),cudaMemcpyDeviceToHost);
          std::cout << "Iteration: " << iter << " Particle: " << minIndx <<
            " Value: " << val << std::endl;
        }
    }

  // Best particle overall
  cublasIsamin(cublasH,numParticles,wa.GCvals_dev,1,&minIndx);
  minIndx = minIndx - 1;
  cudaMemcpy(q_best.data(),wa.q_dev+minIndx*numComps,sizeof(float)*numComps,
             cudaMemcpyDeviceToHost);

  cublasDestroy(cublasH);
  freeWorkArray(wa);
  return;
}

// ===========================================================================================
//  Version 3: fused kernels for numComps <= 32 (see the comment above v3_paddedComps in
//  kernels_vruj.cu). Two kernel launches per iteration and no per-(particle, frequency,
//  component) arrays.
// ===========================================================================================
void runFEHDstep_v3(std::vector<float> &q_best, std::vector<float> &L,
                    dataClass<float> dataArray,paramContainer params,int numComps)
{
  const int paddedComps = v3_paddedComps(numComps);
  if(paddedComps < 0)
    {
      std::cout << "runFEHDstep_v3: numComps = " << numComps << " > 32, use runFEHDstep_v2" << std::endl;
      return;
    }

  // Seed for the GPU random numbers (starting q's and recycling). Use a fixed number
  // instead of time(0) to make runs reproducible.
  const unsigned long long seed = (unsigned long long)time(0);

  // Recycling: a particle whose GC improves by no more than this in one iteration is
  // replaced by a random unit vector (its best result is kept). 0 = recycle only particles
  // that no step size can improve; a small positive value (e.g. 1e-6) also recycles ones
  // that are barely moving; -1 = never recycle.
  const float recycleTol = 0.0f;

  MVAR<float> model = mkARGPU(dataArray,params);

  int maxLag = *std::max_element(model.lagList.begin(),model.lagList.end());
  int epochAdj = dataArray.getEpochPoints() - maxLag;
  dataClass<float> residuals(epochAdj,dataArray.getNumComps(),model.R,dataArray.getSampRate());
  std::vector<float> Dmat = PCA(residuals);

  MVAR<float> rModel = rotate_model(model,Dmat);
  L.resize(Dmat.size());
  std::copy(Dmat.begin(),Dmat.end(),L.begin());

  const int numFreqs = params.numFreqs;
  const int numParticles = params.numParticles;

  // Line-search step sizes (same as version 1). Up to V3_MAXSTEPS (16); extra ones cost only
  // a few operations and one log per (particle, frequency).
  std::vector<float> steps = {0.0f,0.0001f,0.01f,0.05f,0.5f};
  const int numSteps = (int)steps.size();
  v3_setSteps(steps.data(), numSteps);

  workForGranger wa;
  allocateParams_v3(wa,numComps,numParticles,params,paddedComps);

  cudaMemcpy(wa.AR_dev,rModel.A.data(),sizeof(float)*params.numLags*numComps*numComps,cudaMemcpyHostToDevice);
  cudaMemcpy(wa.lagList_dev,rModel.lagList.data(),sizeof(int)*params.numLags,cudaMemcpyHostToDevice);

  const int blksize = 256;
  const float dt = 1.0f/float(params.sampRate);
  const int CCF_grid = (numComps*numComps*numFreqs + blksize - 1)/blksize;
  G_vruj<<<CCF_grid,blksize>>>(wa.G_dev,wa.AR_dev,wa.lagList_dev,numComps,params.freqLo,
                               params.freqHi,numFreqs,params.numLags,dt);

  // Padded K = [M; Sr; Si] per frequency. Computed once; every iteration only reads it.
  v3_buildK(wa.K_dev, wa.G_dev, numComps, numFreqs);

  // Random starting q's (uniform on the unit sphere), generated on the GPU, and
  // bestGC = +infinity.
  v3_initParticles(wa.q_dev, wa.bestGC_dev, numComps, numParticles, seed);

  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);   // only used for Isamin below

  v3_args args;
  args.q = wa.q_dev;               args.d = wa.delF_proj_dev;
  args.qq = wa.qq_dev;             args.GCvals = wa.GCvals_dev;
  args.Kpad = wa.K_dev;
  args.bestGC = wa.bestGC_dev;     args.bestQ = wa.bestQ_dev;
  args.recycleCount = params.verbose ? wa.recycleCount_dev : nullptr;  // count only if printing
  args.recycleTol = recycleTol;
  args.seed = seed;
  args.numSteps = numSteps;        args.numComps = numComps;
  args.numFreqs = numFreqs;        args.numParticles = numParticles;

  const int numIts = 100;
  int minIndx;
  float val;
  int numRecycled;
  for(int iter = 0;iter<numIts;iter++)
    {
      if(params.verbose) cudaMemset(wa.recycleCount_dev, 0, sizeof(int));

      // Gradient + direction, then line search + update (+ best-so-far and recycling).
      args.iter = iter;
      v3_iterate(args);

      if(params.verbose)
        {
          // bestGC >= 0, so the smallest absolute value (Isamin) is the smallest value.
          cublasIsamin(cublasH,numParticles,wa.bestGC_dev,1,&minIndx);
          minIndx = minIndx - 1;   // cuBLAS indices start at 1
          cudaMemcpy(&val,wa.bestGC_dev+minIndx,sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(&numRecycled,wa.recycleCount_dev,sizeof(int),cudaMemcpyDeviceToHost);
          std::cout << "Iteration: " << iter << " Particle: " << minIndx <<
            " Best value: " << val << " Recycled: " << numRecycled << std::endl;
        }
    }

  // Best result any particle reached, including ones later recycled.
  cublasIsamin(cublasH,numParticles,wa.bestGC_dev,1,&minIndx);
  minIndx = minIndx - 1;
  cudaMemcpy(q_best.data(),wa.bestQ_dev+minIndx*numComps,sizeof(float)*numComps,
             cudaMemcpyDeviceToHost);

  cublasDestroy(cublasH);
  freeWorkArray(wa);
  return;
}
