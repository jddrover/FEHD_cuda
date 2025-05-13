#include "GC.h"
#include "dataContainers.h"
#include "utility.h"
#include "timeSeriesOPs.h"
#include <iostream>
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
#include <random>
void granger(std::vector<float> angleArray,
	     std::vector<float> &GCvals, paramContainer params,
	     int numComps,workForGranger workArray)
{
  int blksize = 1024;
  int grdsize = (int)(params.numParticles+blksize-1)/blksize;
  const dim3 blockSize(blksize);
  const dim3 gridSize(grdsize);

  float2 alphaC;
  float2 betaC;
  alphaC.x=1.0f;
  alphaC.y=0.0f;
  betaC.x=0.0f;
  betaC.y=0.0f;

  float2 alphaC2;
  float2 betaC2;
  alphaC2.x=-1.0f;
  alphaC2.y=0.0f;
  betaC2.x=1.0f;
  betaC2.y=0.0f;
  
  int lwork = workArray.lworkVal;


  
  // cublas handle. Contains info about the system that routines need.
  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  // cusolver options - don't sort, don't compute eigenvectors, use
  // the upper triangle the matrix is Hermitian
  const int sort_eig = 0;
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

  syevjInfo_t syevj_params = NULL;
  
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);

  cusolverDnCreateSyevjInfo(&syevj_params);
  cusolverDnXsyevjSetSortEig(syevj_params, sort_eig);
  // Multiply A'*Q which will create the M*L x M*P matrix
  // [A_1TQ1T  ... A_1TQPT]
  // [...                 ]
  // [A_LTQ2T  ... A_LTQPT]

  // Recall Q was stored as transposed matrices, so we use the
  // following gemm call to do all particles and lags at once.

  const float alpha=1.0;
  const float beta=0.0;

  int grdsize2 =
  (int)(numComps*numComps*params.numLags*params.numParticles+blksize-1)/blksize; 

  const dim3 gridSize2(grdsize2);

  int grdsize3 =
  (int)(numComps*numComps*params.numParticles*params.numFreqs+blksize-1)/blksize;

  const dim3 blockSizeTF(blksize);
  const dim3 gridSizeTF(grdsize3);
  const int memsizetf = sizeof(float2)*blksize;
  float dt = 1.0f/(float)(params.sampRate);

  int grdsize4 =
  (int)((numComps*params.numFreqs*params.numParticles+blksize-1)/blksize);

  const dim3 gridSizeScale(grdsize4);


  int grdsize5 =
  (int)(params.numParticles*params.numFreqs*(numComps-1)*(numComps-1)+blksize-1)/blksize;

  const dim3 blockSizeShrink(blksize);
  const dim3 gridSizeShrink(grdsize5);

  int grdsize6 = (int)(params.numParticles*params.numFreqs+blksize-1)/blksize;
  const dim3 blockSizeProd(blksize);
  const dim3 gridSizeProd(grdsize6);
  
  const int memsizeEig = sizeof(float)*blksize;


  int grdsize7 = (int)(params.numParticles+blksize-1)/blksize;
  const dim3 blockSize_det2GC(blksize);
  const dim3 gridSize_det2GC(grdsize7);



  // I remember this being here because it was difficult otherwise, I do not remember
  // what made it difficult. I would like to place it externally. 3/3/25
  // Create an array and allocate space on the device to store the rotation matrices
  float *angles_dev;
  cudaMalloc((void**)&angles_dev, sizeof(float)*(numComps-1)*params.numParticles);

  cudaMemcpy(angles_dev,angleArray.data(),sizeof(float)*(numComps-1)*params.numParticles,
	     cudaMemcpyHostToDevice);


  // Using the angles in angles_dev, create the rotation matrices Q.
  generateRotationMatrices<<<gridSize,blockSize>>>(angles_dev,workArray.Qdev,numComps,params.numParticles);
  // Create
  // [A1^tQ1* A1^tQ2* ... A1^tQp*]
  // [A2^tQ1* ...                ]
  // [...                        ]
  // [AL^tQ1* ...     ... AL^tQp*]
  cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numComps*params.numLags,numComps*params.numParticles,
  	      numComps,&alpha,workArray.ARdev,numComps,workArray.Qdev,numComps,&beta,workArray.rotatedModels,
	      numComps*params.numLags);
  // Transpose each individual lag matrix
  // [Q1A1 Q2A1 ... ... QpA1]
  // [Q1A2 ...              ]
  // [...                   ]
  // [Q1AL ...  ... ... QpAL]
  transposeBlockMatrices<<<gridSize2,blockSize>>>(workArray.rotatedModels,workArray.wArray,numComps,params.numParticles,params.numLags);
  // Multiply, strided
  // [Q1A1]     [Q2A1]    ... [QpA1]
  // [ ...]Q1*  [... ]Q2* ... [... ]Qp*
  // [Q1AL]     [Q2AL]    ... [QpAL]
  cublasSgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
			    params.numLags*numComps,numComps,numComps,
			    &alpha,
			    workArray.wArray,numComps*params.numLags,params.numLags*numComps*numComps,
			    workArray.Qdev,numComps,numComps*numComps,
			    &beta,
			    workArray.rotatedModels,numComps*params.numLags,params.numLags*numComps*numComps,
			    params.numParticles);
  // Compute the inverse of the transfer function - numParticles * numFreqs complex matrices
  // [Tfp1f1^-1, Tfp1f2^-1, ... , Tfp1fF^-1, Tfp2f1^-1, ... TfppfF^-1]
  // See the function in kernels.cu for the details on how it works.
  compTransferFunc<<<gridSizeTF,blockSizeTF,memsizetf>>>(workArray.rotatedModels,workArray.Tf,workArray.lagList_DEVICE,numComps,
						       params.numParticles,params.freqLo,
						       params.freqHi,params.numFreqs,
						       params.numLags,dt);
  // Compute (Tf Tf*)^-1=Tf*^-1 Tf^-1 - the inverse of the variance in the neighborhood of each frequency
  // Collectively the inverse power spectrum of the model. 
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_C,CUBLAS_OP_N,
				   numComps,numComps,numComps,
				   &alphaC,
				   workArray.Tf,numComps,numComps*numComps,
				   workArray.Tf,numComps,numComps*numComps,
				   &betaC,
				   workArray.Swhole,numComps,numComps*numComps,
				   params.numParticles*params.numFreqs);

  // Our first goal is to obtain the sub m-1 x m-1 spectral matrix for each frequency.
  // We have avoided inversion, so we don't have the spectral matrix, we have its inverse.
  // Consider the block matrix inverse:
  // [A  B]^-1   [[A-CB/D]^-1 X]
  // [    ]    = [             ]
  // [C  D]      [ X          X]
  // This is the inverse of the spectral sub matrix. We're going to calculate its determinant, so
  // we will not need to invert it.
  // This function scales the mth column (B) in each submatrix by the m,m entry.
  // This is preparation for the gemm below. 
  scale_columns<<<gridSizeScale,blockSizeTF>>>(workArray.Swhole,numComps,params.numParticles,params.numFreqs);
  // Copy the entire spectral matrix to a temporary array (not really temporary)
  cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
	      workArray.Swhole,1,workArray.tmp,1);
  // GEMM does the above calculation.
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
				   numComps-1,numComps-1,1,
				   &alphaC2,
				   workArray.Swhole+(numComps-1)*numComps,numComps,numComps*numComps,
				   workArray.Swhole+(numComps-1),numComps,numComps*numComps, 
				   &betaC2,
				   workArray.tmp,numComps,numComps*numComps,
				   params.numParticles*params.numFreqs);
  // Copy back to Swhole array
  cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
	      workArray.tmp,1,workArray.Swhole,1);

  // Same trick, but we need the product of the sub-transfer functions.
  // We determine the sub-inverse as above, first by scaling:
  scale_columns<<<gridSizeScale,blockSizeTF>>>(workArray.Tf,numComps,params.numParticles,params.numFreqs);
  // Then copying to a temporary array
  cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
	      workArray.Tf,1,workArray.tmp,1);
  // Now we have the inverses of the sub transfer functions.
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
				   numComps-1,numComps-1,1,
				   &alphaC2,
				   workArray.Tf+(numComps-1)*numComps,numComps,numComps*numComps,
				   workArray.Tf+numComps-1,numComps,numComps*numComps,
				   &betaC2,
				   workArray.tmp,numComps,numComps*numComps,
				   params.numParticles*params.numFreqs);

  // And we multiply them to get the spectrum without the influence of the last component.
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_C,CUBLAS_OP_N,
				   numComps-1,numComps-1,numComps-1,
				   &alphaC,
				   workArray.tmp,numComps,numComps*numComps,
				   workArray.tmp,numComps,numComps*numComps,
				   &betaC,
				   workArray.Spartial,numComps-1,(numComps-1)*(numComps-1),
				   params.numParticles*params.numFreqs);
  // Cheev batched doesn't stride, so I shrink the whole spectrum arrays to the m-1 x m-1 size. 
  shrinkArrays<<<gridSizeShrink,blockSizeShrink>>>(workArray.Swhole, workArray.d_wholeSpec, numComps, params.numParticles, params.numFreqs);
  // Cholesky algorithm to determine the eigenvalues (we set it not to compute eigenvectors, it can)
  cusolverDnCheevjBatched(cusolverH,jobz,uplo,numComps-1,workArray.d_wholeSpec,numComps-1,
			  workArray.dev_W, workArray.d_work2,lwork,workArray.d_info, syevj_params, params.numFreqs*params.numParticles);
  // Multiply the eigenvalues together to get the determinant. 
  prodEigs<<<gridSizeProd,blockSizeProd,memsizeEig>>>(workArray.dev_W, workArray.det_whole, numComps-1, params.numParticles, params.numFreqs);
  // Repeat for the partial spectral matrices. 
  cusolverDnCheevjBatched(cusolverH,jobz,uplo,numComps-1,workArray.Spartial,numComps-1,
			  workArray.dev_W, workArray.d_work2,lwork,workArray.d_info, syevj_params, params.numFreqs*params.numParticles);
  // Compute the determinant.
  prodEigs<<<gridSizeProd,blockSizeProd,memsizeEig>>>(workArray.dev_W, workArray.det_partial, numComps-1, params.numParticles, params.numFreqs);      
  // Divides the determinants, takes the log, and adds to the integral. 
  det2GC<<<gridSize_det2GC,blockSize_det2GC>>>(workArray.det_partial, workArray.det_whole, workArray.dev_GC,params.numParticles,params.numFreqs);
  // Send the numParticles Granger causality values to the system memory.
  cudaMemcpy(GCvals.data(),workArray.dev_GC,sizeof(float)*params.numParticles,cudaMemcpyDeviceToHost);
  // Clean up (if you don't memory will leak).
  cusolverDnDestroy(cusolverH);
  cublasDestroy(cublasH);
  cudaFree(angles_dev);

  
  return;
}

void runFEHDstep(std::vector<float> &bestAngle, matrix &L, dataList dataArray ,paramContainer params,int numComps)
{
  
  // Determine the available memory on the GPU
  int id;
  size_t freemem,total;
  cudaGetDevice(&id);
 
  srand((unsigned)time(0));

  ARmodel A;
  dataList residuals;
  std::vector<int> lagList(params.lagList);

  std::sort(lagList.begin(),lagList.end());

  mkARGPU(dataArray, lagList, A, residuals);
  
  // Orthonormalize the residuals using the SVD.
  dataList ortho_residuals;

  // Obtain the transformation that orthonormalizes the residual time series.
  orthonormalizeR(residuals, ortho_residuals, L); // This function is in mkARGPU.h
  // Apply the transformations - LAL^-1
  rotate_model(A, L); // Also in mkARGPU.h

  // Convert the AR model format to a single vector so it can be copied etc.
  std::vector<float> AR(params.numLags*numComps*numComps,0);
  for(int lag=0;lag<params.numLags;lag++)
    for(int row=0;row<numComps;row++)
      for(int col=0;col<numComps;col++)
	{
	  AR[lag*numComps*numComps+col*numComps+row] = A.lagMatrices[lag].elements[col*numComps+row];
	}

  // Determine how to break up the analysis so that it fits on the GPU.
  int numBlocks, particleBlockSize;
  cudaMemGetInfo(&freemem, &total);

  if(params.verbose)
    {
      printf("memory free = %ld bytes \n",freemem);
      printf("total memory = %ld bytes \n",total);
    }
  
  computeBlocks(numBlocks,particleBlockSize,freemem,params,numComps);
  
  paramContainer paramsBLOCKED = params;
  paramsBLOCKED.numParticles = particleBlockSize;

  if(params.verbose)
    {
      printf("Number of blocks = %i \n",numBlocks);
      printf("Block size = %i \n",particleBlockSize);
    }

  std::vector<float> GCmin(numBlocks,0);
  std::vector<int> GCminIndex(numBlocks,0);

  // Allocate all of the arrays need for the GC function. 
  workForGranger workArray;
  allocateParams(workArray,numComps,particleBlockSize,params,lagList,AR);
   
  // Angle arrays.
  std::vector<std::vector<float>> angleArray;
  //std::vector<std::vector<float>> angleArray1;
  //std::vector<std::vector<float>> angleArray2;
  //std::vector<std::vector<float>> angleArray3;

  //std::vector<float> tmpAngle;
  std::vector<float> tmpAngle;
  std::random_device gen;
  std::default_random_engine generator(gen());
  std::uniform_real_distribution<float> distribution(-3.1415,3.1415);
  for(int block=0;block<numBlocks;block++)
    {
      for(int indx=0;indx<particleBlockSize*(numComps-1);indx++)
	tmpAngle.push_back(distribution(generator));
  
      angleArray.push_back(tmpAngle);
      tmpAngle.clear();
    }

  // GCvals arrays - these store the Granger causality and
  // are the value we wish to minimize.
  std::vector<std::vector<float>> GCvals;

  
  std::vector<float> GCtmp(particleBlockSize,0);
  
  for(int block=0;block<numBlocks;block++)
    GCvals.push_back(GCtmp);


  // gradient arrays for each block.
  //std::vector<std::vector<float>> gradient;
  //std::vector<float> gradientTmp(particleBlockSize*(numComps-1),0);

  //for(int block=0;block<numBlocks;block++)
  //  gradient.push_back(gradientTmp);
      
  for(int block=0;block<numBlocks;block++)
    {
      granger(angleArray[block],GCvals[block], paramsBLOCKED,numComps,workArray);
    }

  // At this point, we have the initialization. Translate to L, location value couple
  // This code block compiled successfully. Still needs to be checked for correctness.
  std::vector<particleObject> particleInfo(params.numParticles);
  
  for(int block=0;block<numBlocks;block++)
    {
      for(int part=0;part<paramsBLOCKED.numParticles;part++)
	{
	  particleInfo[block*paramsBLOCKED.numParticles+part].location.resize(numComps-1);
	  std::copy(angleArray[block].begin()+block*paramsBLOCKED.numParticles+
		    part*(numComps-1),
		    angleArray[block].begin()+block*paramsBLOCKED.numParticles+
		    part*(numComps-1)+(numComps-1),
		    particleInfo[block*paramsBLOCKED.numParticles+part].location.begin());
	  particleInfo[block*paramsBLOCKED.numParticles+part].value = GCvals[block][part];
	}
    }
  std::vector<particleObject> particleMIN(particleInfo);
  particleObject pGlobal;
  std::vector<float> v(params.numParticles);
  for(int part=0;part<params.numParticles;part++)
    {
      v[part]=particleInfo[part].value;
    }
  int globalMinIndex = std::min_element(v.begin(),v.end())-v.begin();
  std::vector<float> globalMinLocation (particleInfo[globalMinIndex].location);
  pGlobal.location = globalMinLocation;
  pGlobal.value = v[globalMinIndex];

      
  int STATIONARY_COUNT = 0;
  const int COUNTMAX = params.STUCKCOUNT;

  
  
  //for(int iter=0;iter<numIts;iter++)
  int iter = 0;
  float pGlobal_store = pGlobal.value;
  while(STATIONARY_COUNT < COUNTMAX)
    {
      PSOstep(particleInfo,particleMIN,pGlobal,paramsBLOCKED,numBlocks,numComps,workArray);
      if(pGlobal.value >= pGlobal_store)
	STATIONARY_COUNT++;
      else
	{
	  STATIONARY_COUNT=0;
	  pGlobal_store = pGlobal.value;
	}
      iter++;
      std::cout << "Iteration: " << iter << " Global value: " << pGlobal.value << " Exit count: " << STATIONARY_COUNT << std::endl;
      std::cout << "Location: " << std::endl;

      for (int comp=0;comp<numComps-1;comp++)
	  std::cout << pGlobal.location[comp] << std:: endl;
      std::cout << std::endl;
      std::cout << particleInfo[0].value << " " << particleMIN[0].value << std::endl;
      
	
    }
  
  freeWorkArray(workArray);
  bestAngle = pGlobal.location;
  
  return;
 
}
void compGradient(std::vector<float> &gradient ,std::vector<float> GCvalsBASE,std::vector<float> angleArray,paramContainer params, int numComps,
		  workForGranger workArray)
{
  const int numVars = numComps-1;
  const float  h_val = 0.001f; // This is for the gradient spacing.
  
  std::vector<float> angle(angleArray); // Copy this
  std::vector<float> GCvalsUTIL(params.numParticles,0);

  // These are all the same size, can I allocate the arrays here?
  // I will have to rename the file to .cu
  // Is passing device arrays done normally? Weird that I don't know this.
  
  for(int varIndex=0;varIndex<numVars;varIndex++)
    {
      
      for(int particle=0;particle<params.numParticles;particle++)
	{
	  angle[particle*numVars+varIndex] += h_val;
	}

      granger(angle,GCvalsUTIL,params,numComps,workArray);

      for(int particle=0;particle<params.numParticles;particle++)
	{
	  if(std::isnan(GCvalsUTIL[particle]))
	    {
	      printf("GC nan in gradient \n");
	      exit(0);
	    }
	  if(std::isnan(GCvalsBASE[particle]))
	    {
	      printf("It's in the base \n");
	      exit(0);
	    }
	}
      
      for(int particle=0;particle<params.numParticles;particle++)
	{
	  
	  gradient[particle*numVars+varIndex] = (GCvalsUTIL[particle]-
							GCvalsBASE[particle])/h_val;

	  if(std::isnan(gradient[particle*numVars+varIndex]))
	    {
	      printf("isnan after the calculation \n");
	      printf("GCvalsUTIL = %f, GCvalsBASE = %f \n",GCvalsUTIL[particle],GCvalsBASE[particle]);
	      exit(0);
	    }
	  
	  angle[particle*numVars+varIndex] -= h_val;
	  
	}      
    }  
}

void computeBlocks(int &numBlocks,int &particleBlockSize,size_t memval,paramContainer params,int numComps)
{
  int k;
  unsigned long int required=(unsigned long int)((unsigned long int)params.numParticles*
						 (unsigned long int)(sizeof(float)*
								     ((1+2*params.numLags)*numComps*numComps+ // Qdev and workArrays
								      2*params.numFreqs+1+params.numFreqs*(numComps-1)+numComps-1)+ // determinants, GCval,eigenvalues, anglearray
								     sizeof(float2)*
								     (3*params.numFreqs*numComps*numComps+ // Tf, Swhole, tmp
								      4*params.numFreqs*(numComps-1)*(numComps-1))+ // D_wholeSpec,Spartial,2x work array
								     sizeof(int)*params.numFreqs)+ // info array
						 (unsigned long int)(sizeof(int)*params.numLags+sizeof(float)*params.numLags*numComps*numComps));// laglist, ARmodel
  
  unsigned long int allowable = (unsigned long int)(memval);
  allowable = (unsigned long int)((double)allowable*0.9);

  if(params.verbose)
    {
      printf("memory requested = %lu \n",required);
      printf("memory available per block = %lu \n",allowable);
    }
  if(required <= allowable)
    {
      numBlocks = 1;
      particleBlockSize = params.numParticles;
    }
  else
    {
      k = ceil(log2f((float)((float)required/(float)allowable)));
      numBlocks = pow(2,k);
      particleBlockSize = ceil(params.numParticles/numBlocks);
    }

  return;
}

void PSOstep(std::vector<particleObject> &L,std::vector<particleObject> &Lmin,particleObject &Lglobal,paramContainer paramsBLOCKED,
	     int numBlocks,int numComps,workForGranger workArray)
{
  // Establish the minimum value and record location
  int P = L.size();
  int M = L[0].location.size();
  std::cout << "M=" << M << std::endl;
  float movement, noiseSize;
  float h=0.05; // h < 2/(alpha+beta)
  float alpha=0.25;
  float beta=1.0;
  int Pblock = paramsBLOCKED.numParticles;
  std::vector<std::vector<float>> angleArray;
  // Noise - reseed?
  // Noise has normal amplitude in each direction
  // This amplitude is scaled according to a parameter.
  std::random_device gen;
  std::default_random_engine generator(gen());
  //std::uniform_real_distribution<float> distribution(-3.14159,3.14159);
  for(int part=0;part<P;part++)
    {
      for(int k=0;k<M;k++)
	{
	  //movement = h*(Lmin[part].location[k]-L[part].location[k]+Lglobal.location[k]-L[part].location[k]);
	  movement = h*(alpha*(Lmin[part].location[k]-L[part].location[k])+beta*(Lglobal.location[k]-L[part].location[k]));
	  L[part].location[k] = L[part].location[k]+movement;
	  noiseSize = std::abs(movement);
	  std::normal_distribution<float> distribution(0.0,noiseSize*0.1+0.01);
	  L[part].location[k] = L[part].location[k]+movement+distribution(generator);
	}
    }
  // Turn L back into the angle arrays.
  std::vector<float> tmpVec;
  for(int block=0;block<numBlocks;block++)
    {
      for(int part=0;part<Pblock;part++)
	{
	  for(int indx=0;indx<numComps-1;indx++)
	    tmpVec.push_back(L[block*Pblock+part].location[indx]);
	}
      angleArray.push_back(tmpVec);
    }
  // Create GCvals
  std::vector<float> GCvals(Pblock);

  
  
  for(int block=0;block<numBlocks;block++)
    {      
      granger(angleArray[block],GCvals, paramsBLOCKED,numComps,workArray);

      for(int part=0;part<Pblock;part++)
	L[block*Pblock+part].value = GCvals[part];      
    }
  
  int globalParticle;
  for(int part=0;part<P;part++)
    {
      if(L[part].value < Lmin[part].value)
	{
	  Lmin[part].value = L[part].value;
	  Lmin[part].location = L[part].location;
	}
      if(L[part].value < Lglobal.value)
	{
	  Lglobal.location = L[part].location;
	  Lglobal.value = L[part].value;
	  globalParticle = part;
	}
    }
  std::cout << "global particle: " << part
  return;  
}
  
  
