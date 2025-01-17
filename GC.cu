#include "GC.h"
#include "dataContainers.h"
#include "utility.h"
#include "timeSeriesOPs.h"
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

void granger(float *ARdev, std::vector<float> angleArray,
	     std::vector<float> &GCvals, paramContainer params, int
	     numComps,int *lagList_DEVICE, float *Qdev,float
	     *rotatedModels,float *workArray,float2 *Tf,float2
	     *Swhole,float2 *tmp,float2 *Spartial,float2 *d_wholeSpec,
	     float *dev_W,int *d_info,int &lworkVal,float2
	     *d_work2,float *det_whole,float *det_partial,float
	     *dev_GC)
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
  
  int lwork = lworkVal;

  
  
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



  
  // Create an array and allocate space on the device to store the rotation matrices
  float *angles_dev;
  cudaMalloc((void**)&angles_dev, sizeof(float)*(numComps-1)*params.numParticles);

  cudaMemcpy(angles_dev,angleArray.data(),sizeof(float)*(numComps-1)*params.numParticles,
	     cudaMemcpyHostToDevice);


  // Using the angles in angles_dev, create the rotation matrices Q.
  generateRotationMatrices<<<gridSize,blockSize>>>(angles_dev,Qdev,numComps,params.numParticles);
  // Create
  // [A1^tQ1* A1^tQ2* ... A1^tQp*]
  // [A2^tQ1* ...                ]
  // [...                        ]
  // [AL^tQ1* ...     ... AL^tQp*]
  cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numComps*params.numLags,numComps*params.numParticles,
  	      numComps,&alpha,ARdev,numComps,Qdev,numComps,&beta,rotatedModels,
	      numComps*params.numLags);
  // Transpose each individual lag matrix
  // [Q1A1 Q2A1 ... ... QpA1]
  // [Q1A2 ...              ]
  // [...                   ]
  // [Q1AL ...  ... ... QpAL]
  transposeBlockMatrices<<<gridSize2,blockSize>>>(rotatedModels,workArray,numComps,params.numParticles,params.numLags);
  // Multiply, strided
  // [Q1A1]     [Q2A1]    ... [QpA1]
  // [ ...]Q1*  [... ]Q2* ... [... ]Qp*
  // [Q1AL]     [Q2AL]    ... [QpAL]
  cublasSgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
			    params.numLags*numComps,numComps,numComps,
			    &alpha,
			    workArray,numComps*params.numLags,params.numLags*numComps*numComps,
			    Qdev,numComps,numComps*numComps,
			    &beta,
			    rotatedModels,numComps*params.numLags,params.numLags*numComps*numComps,
			    params.numParticles);
  // Compute the inverse of the transfer function - numParticles * numFreqs complex matrices
  // [Tfp1f1^-1, Tfp1f2^-1, ... , Tfp1fF^-1, Tfp2f1^-1, ... TfppfF^-1]
  // See the function in kernels.cu for the details on how it works.
  compTransferFunc<<<gridSizeTF,blockSizeTF,memsizetf>>>(rotatedModels,Tf,lagList_DEVICE,numComps,
						       params.numParticles,params.freqLo,
						       params.freqHi,params.numFreqs,
						       params.numLags,dt);
  // Compute (Tf Tf*)^-1=Tf*^-1 Tf^-1 - the inverse of the variance in the neighborhood of each frequency
  // Collectively the inverse power spectrum of the model. 
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_C,CUBLAS_OP_N,
				   numComps,numComps,numComps,
				   &alphaC,
				   Tf,numComps,numComps*numComps,
				   Tf,numComps,numComps*numComps,
				   &betaC,
				   Swhole,numComps,numComps*numComps,
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
  scale_columns<<<gridSizeScale,blockSizeTF>>>(Swhole,numComps,params.numParticles,params.numFreqs);
  // Copy the entire spectral matrix to a temporary array (not really temporary)
  cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
	      Swhole,1,tmp,1);
  // GEMM does the above calculation.
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
				   numComps-1,numComps-1,1,
				   &alphaC2,
				   Swhole+(numComps-1)*numComps,numComps,numComps*numComps,
				   Swhole+(numComps-1),numComps,numComps*numComps, 
				   &betaC2,
				   tmp,numComps,numComps*numComps,
				   params.numParticles*params.numFreqs);
  // Copy back to Swhole array
  cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
	      tmp,1,Swhole,1);

  // Same trick, but we need the product of the sub-transfer functions.
  // We determine the sub-inverse as above, first by scaling:
  scale_columns<<<gridSizeScale,blockSizeTF>>>(Tf,numComps,params.numParticles,params.numFreqs);
  // Then copying to a temporary array
  cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
	      Tf,1,tmp,1);
  // Now we have the inverses of the sub transfer functions.
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
				   numComps-1,numComps-1,1,
				   &alphaC2,
				   Tf+(numComps-1)*numComps,numComps,numComps*numComps,
				   Tf+numComps-1,numComps,numComps*numComps,
				   &betaC2,
				   tmp,numComps,numComps*numComps,
				   params.numParticles*params.numFreqs);

  // And we multiply them to get the spectrum without the influence of the last component.
  cublasCgemmStridedBatched(cublasH,CUBLAS_OP_C,CUBLAS_OP_N,
				   numComps-1,numComps-1,numComps-1,
				   &alphaC,
				   tmp,numComps,numComps*numComps,
				   tmp,numComps,numComps*numComps,
				   &betaC,
				   Spartial,numComps-1,(numComps-1)*(numComps-1),
				   params.numParticles*params.numFreqs);
  // Cheev batched doesn't stride, so I shrink the whole spectrum arrays to the m-1 x m-1 size. 
  shrinkArrays<<<gridSizeShrink,blockSizeShrink>>>(Swhole, d_wholeSpec, numComps, params.numParticles, params.numFreqs);
  // Cholesky algorithm to determine the eigenvalues (we set it not to compute eigenvectors, it can)
  cusolverDnCheevjBatched(cusolverH,jobz,uplo,numComps-1,d_wholeSpec,numComps-1,
			  dev_W, d_work2,lwork,d_info, syevj_params, params.numFreqs*params.numParticles);
  // Multiply the eigenvalues together to get the determinant. 
  prodEigs<<<gridSizeProd,blockSizeProd,memsizeEig>>>(dev_W, det_whole, numComps-1, params.numParticles, params.numFreqs);
  // Repeat for the partial spectral matrices. 
  cusolverDnCheevjBatched(cusolverH,jobz,uplo,numComps-1,Spartial,numComps-1,
			  dev_W, d_work2,lwork,d_info, syevj_params, params.numFreqs*params.numParticles);
  // Compute the determinant.
  prodEigs<<<gridSizeProd,blockSizeProd,memsizeEig>>>(dev_W, det_partial, numComps-1, params.numParticles, params.numFreqs);      
  // Divides the determinants, takes the log, and adds to the integral. 
  det2GC<<<gridSize_det2GC,blockSize_det2GC>>>(det_partial, det_whole, dev_GC,params.numParticles,params.numFreqs);
  // Send the numParticles Granger causality values to the system memory.
  cudaMemcpy(GCvals.data(),dev_GC,sizeof(float)*params.numParticles,cudaMemcpyDeviceToHost);
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


  // Number of iterations
  int numIts = 50;
  // The step-sizes to check along the (-)gradient.
  std::vector<float> h = {0.01f, 1.0f, 10.0f};

  std::vector<float> candidates(4,0);
  int minIndx;
  float allBlockMin;

  int minBlockNumber;
  // For recycling
  unsigned long int minimumGC;
  unsigned long int allBlockParticle;
  std::vector<int> resetList;
  std::vector<float> angleArrayReset;

  std::vector<float> GCvalsReset;

  paramContainer paramsReset = params;

  // Determine how to break up the analysis so that it fits on the GPU.
  int numBlocks, particleBlockSize;
  cudaMemGetInfo(&freemem, &total);

  if(params.verbose)
    {
      printf("memory free = %ld bytes \n",freemem);
      printf("total memory = %ld bytes \n",total);
    }
  
  computeBlocks(numBlocks,particleBlockSize,freemem,params,numComps); // Need to write this.
  
  paramContainer paramsBLOCKED = params;
  paramsBLOCKED.numParticles = particleBlockSize;

  if(params.verbose)
    {
      printf("Number of blocks = %i \n",numBlocks);
      printf("Block size = %i \n",particleBlockSize);
    }

  std::vector<float> GCmin(numBlocks,0);
  std::vector<int> GCminIndex(numBlocks,0);
  // Allocate a whole bunch of stuff on the GPU.
  float *Qdev;
  cudaMalloc((void**)&Qdev, sizeof(float)*particleBlockSize*numComps*numComps);
  float *rotatedModels;
  float *workArray;
  cudaMalloc((void**)&rotatedModels, sizeof(float)*numComps*numComps*particleBlockSize*params.numLags);
  cudaMalloc((void**)&workArray, sizeof(float)*numComps*numComps*particleBlockSize*params.numLags);
  float2 *Tf;
  cudaMalloc((void**)&Tf,sizeof(float2)*numComps*numComps*particleBlockSize*params.numFreqs);
  float2 *Swhole;
  cudaMalloc((void**)&Swhole,sizeof(float2)*numComps*numComps*params.numFreqs*particleBlockSize);
  float2 *tmp;
  cudaMalloc(&tmp,sizeof(float2)*particleBlockSize*params.numFreqs*numComps*numComps);
  float2 *Spartial; // Make partial one size down
  cudaMalloc((void**)&Spartial,sizeof(float2)*(numComps-1)*(numComps-1)*params.numFreqs*particleBlockSize);
  float2 *d_wholeSpec;
  cudaMalloc(&d_wholeSpec,sizeof(float2)*params.numFreqs*particleBlockSize*(numComps-1)*(numComps-1));
  float *dev_W;
  cudaMalloc(&dev_W,sizeof(float)*particleBlockSize*params.numFreqs*(numComps-1));
  int *d_info;
  cudaMalloc(&d_info,sizeof(int)*particleBlockSize*params.numFreqs);
  int lworkVal = particleBlockSize*params.numFreqs*2*(numComps-1)*(numComps-1); 
  float2 *d_work2;
  cudaMalloc(&d_work2,sizeof(float2)*lworkVal);
  float *det_whole;
  cudaMalloc(&det_whole,sizeof(float)*particleBlockSize*params.numFreqs);
  float *det_partial;
  cudaMalloc(&det_partial,sizeof(float)*particleBlockSize*params.numFreqs);
  float *dev_GC;
  cudaMalloc(&dev_GC,sizeof(float)*particleBlockSize);
  int *lagList_DEVICE;
  cudaMalloc(&lagList_DEVICE,sizeof(int)*lagList.size());
  cudaMemcpy(lagList_DEVICE,lagList.data(),sizeof(int)*lagList.size(),cudaMemcpyHostToDevice);
  float *ARdev;
  cudaMalloc(&ARdev,sizeof(float)*AR.size());
  cudaMemcpy(ARdev,AR.data(),sizeof(float)*AR.size(),cudaMemcpyHostToDevice);

  // Angle arrays.
  std::vector<std::vector<float>> angleArray;
  std::vector<std::vector<float>> angleArray1;
  std::vector<std::vector<float>> angleArray2;
  std::vector<std::vector<float>> angleArray3;

  std::vector<float> tmpAngle;
  
  for(int block=0;block<numBlocks;block++)
    {
      for(int indx=0;indx<particleBlockSize*(numComps-1);indx++)
	tmpAngle.push_back((float)(rand()%314-157)/100.0f);
  
      angleArray.push_back(tmpAngle);
      angleArray1.push_back(tmpAngle);
      angleArray2.push_back(tmpAngle);
      angleArray3.push_back(tmpAngle);
			    
      tmpAngle.clear();
    }

  // GCvals arrays - these store the Granger causality and
  // are the value we wish to minimize.
  std::vector<std::vector<float>> GCvals;
  std::vector<std::vector<float>> GCvals1;
  std::vector<std::vector<float>> GCvals2;
  std::vector<std::vector<float>> GCvals3;
  
  std::vector<float> GCtmp(particleBlockSize,0);
  
  for(int block=0;block<numBlocks;block++)
    {
      GCvals.push_back(GCtmp);
      GCvals1.push_back(GCtmp);
      GCvals2.push_back(GCtmp);
      GCvals3.push_back(GCtmp);
    }

  // gradient arrays for each block.
  std::vector<std::vector<float>> gradient;
  std::vector<float> gradientTmp(particleBlockSize*(numComps-1),0);

  for(int block=0;block<numBlocks;block++)
    gradient.push_back(gradientTmp);
      
  for(int block=0;block<numBlocks;block++)
    {
      granger(ARdev,angleArray[block],GCvals[block], paramsBLOCKED,numComps,lagList_DEVICE,Qdev,rotatedModels,workArray,Tf,Swhole,tmp,
	      Spartial,d_wholeSpec,dev_W,d_info,lworkVal,d_work2,det_whole,det_partial,dev_GC);
    }
  // Here is the iterator - adjustments occur here.
  // while STATIONARY_COUNT < COUNTMAX
  for(int iter=0;iter<numIts;iter++)
    {
      // Get a bunch of gradients
      for(int block=0;block<numBlocks;block++)
	compGradient(ARdev,gradient[block],GCvals[block],angleArray[block],paramsBLOCKED,numComps,lagList_DEVICE,Qdev,rotatedModels,
		     workArray,Tf,Swhole,tmp,Spartial,d_wholeSpec,dev_W,d_info,lworkVal,d_work2,det_whole,det_partial,dev_GC);

      // Assign values to the angles accordning to the gradient.
      for(int block=0;block<numBlocks;block++)
	{
	  angleArray1[block]=angleArray[block];
	  angleArray2[block]=angleArray[block];
	  angleArray3[block]=angleArray[block];
	  cblas_saxpy(particleBlockSize*(numComps-1), -h[0], gradient[block].data(), 1, angleArray1[block].data(),1);
	  cblas_saxpy(particleBlockSize*(numComps-1), -h[1], gradient[block].data(), 1, angleArray2[block].data(),1);
	  cblas_saxpy(particleBlockSize*(numComps-1), -h[2], gradient[block].data(), 1, angleArray3[block].data(),1);
	}

      // Evaluate the minimization candidates.
      for(int block=0;block<numBlocks;block++)
	{
	  granger(ARdev,angleArray1[block],GCvals1[block],paramsBLOCKED,numComps,lagList_DEVICE,Qdev,rotatedModels,workArray,Tf,Swhole,tmp,Spartial,d_wholeSpec,dev_W,
		  d_info,lworkVal,d_work2,det_whole,det_partial,dev_GC);
	  granger(ARdev,angleArray2[block],GCvals2[block],paramsBLOCKED,numComps,lagList_DEVICE,Qdev,rotatedModels,workArray,Tf,Swhole,tmp,Spartial,d_wholeSpec,dev_W,
		  d_info,lworkVal,d_work2,det_whole,det_partial,dev_GC);
	  granger(ARdev,angleArray3[block],GCvals3[block],paramsBLOCKED,numComps,lagList_DEVICE,Qdev,rotatedModels,workArray,Tf,Swhole,tmp,Spartial,d_wholeSpec,dev_W,
		  d_info,lworkVal,d_work2,det_whole,det_partial,dev_GC);
	}

      
      // Determine the minimum value its location for each of the blocks
      // Recycle the particles that are local minima.
      for(int block=0;block<numBlocks;block++)
	{

	  minimumGC = std::distance(GCvals[block].begin(),std::min_element(GCvals[block].begin(),GCvals[block].end()));

	  resetList.clear();
	  GCvalsReset.clear();
	  angleArrayReset.clear();
	  
	  for(int particle=0;particle<particleBlockSize;particle++)
	    {	  	  	  
	      candidates[0] = GCvals[block][particle];
	      candidates[1] = GCvals1[block][particle];
	      candidates[2] = GCvals2[block][particle];
	      candidates[3] = GCvals3[block][particle];
	      
	      minIndx = std::distance(candidates.begin(),min_element(candidates.begin(),candidates.end()));

	      if(minIndx == 0) // Recycle these
		if(minimumGC != particle)
		  {
		    resetList.push_back(particle); // Store the particle numbers to be reset.
		    GCvalsReset.push_back(0.0); // This just adjusts the size, used below.
		    for(int comp=0;comp<numComps-1;comp++) // Reset the angle array, and make a copy for the reset run.
		      {
		        angleArray[block][particle*(numComps-1)+comp] = (float)(rand()%314-157)/100.0f;
			angleArrayReset.push_back(angleArray[block][particle*(numComps-1)+comp]);
		      }
		  }
	    
	      if(minIndx == 1)
		{
		  GCvals[block][particle] = GCvals1[block][particle];
		  std::copy(angleArray1[block].data()+particle*(numComps-1),angleArray1[block].data()+particle*(numComps-1)+numComps-1,
			    angleArray[block].data()+particle*(numComps-1));
		}
	      if(minIndx == 2)
		{
		  GCvals[block][particle] = GCvals2[block][particle];
		  std::copy(angleArray2[block].data()+particle*(numComps-1),angleArray2[block].data()+particle*(numComps-1)+numComps-1,
			    angleArray[block].data()+particle*(numComps-1));
		}
	      if(minIndx == 3)
		{
		  GCvals[block][particle] = GCvals3[block][particle];
		  std::copy(angleArray3[block].data()+particle*(numComps-1),angleArray3[block].data()+particle*(numComps-1)+numComps-1,
			    angleArray[block].data()+particle*(numComps-1));
		}
	    }

	  GCminIndex[block]=std::min_element(GCvals[block].begin(),GCvals[block].end())-GCvals[block].begin();
	  GCmin[block]=GCvals[block][GCminIndex[block]];
	  
	  paramsReset.numParticles = GCvalsReset.size();
       
	  if(paramsReset.numParticles>=1)
	    {
	      granger(ARdev,angleArrayReset,GCvalsReset,paramsReset,numComps,lagList_DEVICE,
		      Qdev,rotatedModels,workArray,Tf,Swhole,tmp,Spartial,d_wholeSpec,dev_W, d_info,
		      lworkVal,d_work2,det_whole,det_partial,dev_GC);
	      for(int resetParticle=0;resetParticle<paramsReset.numParticles;resetParticle++)
		GCvals[block][resetList[resetParticle]]=GCvalsReset[resetParticle];
	    }
	}

      // Find the minimum over all of the blocks
      minBlockNumber = std::min_element(GCmin.begin(),GCmin.end())-GCmin.begin();
      allBlockMin = GCmin[minBlockNumber];
      allBlockParticle = minBlockNumber*particleBlockSize+GCminIndex[minBlockNumber];

      
      
      if(params.verbose)
      	printf("iteration = %i, particle = %li, value = %e \n",
	       iter,allBlockParticle,allBlockMin);
  
    }
  
  // Return the best angle.

  long unsigned int indexVal = GCminIndex[minBlockNumber];

  //printf("%li \n",indexVal);

  std::copy(angleArray[minBlockNumber].data()+indexVal*(numComps-1),angleArray[minBlockNumber].data()+indexVal*(numComps-1)+numComps-1,bestAngle.begin());

  cudaFree(Qdev);
  cudaFree(rotatedModels);
  cudaFree(workArray);
  cudaFree(Tf);
  cudaFree(Swhole);
  cudaFree(tmp);
  cudaFree(Spartial);
  cudaFree(d_wholeSpec);
  cudaFree(dev_W);
  cudaFree(d_info);
  cudaFree(d_work2);
  cudaFree(det_whole);
  cudaFree(det_partial);
  cudaFree(dev_GC);
  cudaFree(lagList_DEVICE);
  cudaFree(ARdev);
  
  return;
 
}
void compGradient(float *ARdev, std::vector<float> &gradient ,std::vector<float> GCvalsBASE,std::vector<float> angleArray,paramContainer params, int numComps,
		  int *lagList_DEVICE,float *Qdev,float *rotatedModels,float *workArray,float2 *Tf,float2 *Swhole,float2 *tmp,float2 *Spartial,
		  float2 *d_wholeSpec,float *dev_W,int *d_info,int lworkVal,float2 *d_work2,float *det_whole,float *det_partial,float *dev_GC)
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

      granger(ARdev,angle,GCvalsUTIL,params,numComps,lagList_DEVICE,Qdev,rotatedModels,workArray,Tf,Swhole,tmp,Spartial,d_wholeSpec,
	      dev_W,d_info,lworkVal,d_work2,det_whole,det_partial,dev_GC);

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


  
  
