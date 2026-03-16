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

  int grdNEW = (int)(params.numParticles+blksize-1)/blksize;
  const dim3 grdBLOCKSIZE(blksize);
  const dim3 grdGRIDSIZE(grdNEW);
  
  const int memsizeEig = sizeof(float)*blksize;


  int grdsize7 = (int)(params.numParticles+blksize-1)/blksize;
  const dim3 blockSize_det2GC(blksize);
  const dim3 gridSize_det2GC(grdsize7);
  // Put error checking on all of these. Problems do arise.
  // I remember this being here because it was difficult otherwise, I do not remember
  // what made it difficult. I would like to place it externally. 3/3/25
  // Create an array and allocate space on the device to store the rotation matrices
  float *angles_dev;
  if(cudaMalloc((void**)&angles_dev, sizeof(float)*(numComps-1)*params.numParticles) != cudaSuccess)
    {
      std::cout << "cudaMalloc failed in GC - angles_dev" << std::endl;
      exit(1);
    }

  if(cudaMemcpy(angles_dev,angleArray.data(),sizeof(float)*(numComps-1)*params.numParticles,
		cudaMemcpyHostToDevice) != cudaSuccess)
    {
      std::cout << "cudamemcpy failed in GC - angles_dev" << std::endl;
      exit(1);
    }
  
  
  // Using the angles in angles_dev, create the rotation matrices Q.
  generateRotationMatrices<<<gridSize,blockSize>>>(angles_dev,workArray.Qdev,numComps,params.numParticles);
  if(cudaGetLastError() != cudaSuccess)
    {
      std::cout << "cuda kernel generateRotationMatrices failed" << std::endl;
      exit(1);
    }

  // Create
  // [A1^tQ1* A1^tQ2* ... A1^tQp*]
  // [A2^tQ1* ...                ]
  // [...                        ]
  // [AL^tQ1* ...     ... AL^tQp*]
  if(cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numComps*params.numLags,numComps*params.numParticles,
		 numComps,&alpha,workArray.ARdev,numComps,workArray.Qdev,numComps,&beta,workArray.rotatedModels,
		 numComps*params.numLags) != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "cublas gemm failed in GC" << std::endl;
      exit(1);
    }
  
  // Transpose each individual lag matrix
  // |Q1A1 Q2A1 ... ... QpA1|
  // |Q1A2 ...              |
  // |...                   |
  // |Q1AL ...  ... ... QpAL|
  transposeBlockMatrices<<<gridSize2,blockSize>>>(workArray.rotatedModels,workArray.wArray,numComps,params.numParticles,params.numLags);
  if(cudaGetLastError() != cudaSuccess)
    {
      std::cout << "cuda kernel transposeBlockMatrices failed" << std::endl;
      exit(1);
    }
 
  // Multiply, strided
  // [Q1A1]     [Q2A1]    ... [QpA1]
  // [ ...]Q1*  [... ]Q2* ... [... ]Qp*
  // [Q1AL]     [Q2AL]    ... [QpAL]
  if(cublasSgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
			       params.numLags*numComps,numComps,numComps,
			       &alpha,
			       workArray.wArray,numComps*params.numLags,params.numLags*numComps*numComps,
			       workArray.Qdev,numComps,numComps*numComps,
			       &beta,
			       workArray.rotatedModels,numComps*params.numLags,params.numLags*numComps*numComps,
			       params.numParticles) != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "strided batch gemm failed" << std::endl;
      exit(1);
    }
  // Compute the inverse of the transfer function - numParticles * numFreqs complex matrices
  // [Tfp1f1^-1, Tfp1f2^-1, ... , Tfp1fF^-1, Tfp2f1^-1, ... TfppfF^-1]
  // See the function in kernels.cu for the details on how it works.
  compTransferFunc<<<gridSizeTF,blockSizeTF,memsizetf>>>(workArray.rotatedModels,workArray.Tf,workArray.lagList_DEVICE,numComps,
						       params.numParticles,params.freqLo,
						       params.freqHi,params.numFreqs,
						       params.numLags,dt);
  if(cudaGetLastError() != cudaSuccess)
    {
      std::cout << "cuda kernel compTransferFunc failed" << std::endl;
      exit(1);
    }
  // Compute (Tf Tf*)^-1=Tf*^-1 Tf^-1 - the inverse of the variance in the neighborhood of each frequency
  // Collectively the inverse power spectrum of the model. 
  if(cublasCgemmStridedBatched(cublasH,CUBLAS_OP_C,CUBLAS_OP_N,
			       numComps,numComps,numComps,
			       &alphaC,
			       workArray.Tf,numComps,numComps*numComps,
			       workArray.Tf,numComps,numComps*numComps,
			       &betaC,
			       workArray.Swhole,numComps,numComps*numComps,
			       params.numParticles*params.numFreqs) != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "strided batch gemm failed" << std::endl;
      exit(1);			    
    }			    
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
  if(cudaGetLastError() != cudaSuccess)
    {
      std::cout << "cuda kernel scale_columns failed" << std::endl;
      exit(1);
    }


  // Copy the entire spectral matrix to a temporary array (not really temporary)
  if(cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
		 workArray.Swhole,1,workArray.tmp,1) != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "complex copy failed" << std::endl;
      exit(1);
    }
  // GEMM does the above calculation.
  if(cublasCgemmStridedBatched(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
			       numComps-1,numComps-1,1,
			       &alphaC2,
			       workArray.Swhole+(numComps-1)*numComps,numComps,numComps*numComps,
			       workArray.Swhole+(numComps-1),numComps,numComps*numComps, 
			       &betaC2,
			       workArray.tmp,numComps,numComps*numComps,
			       params.numParticles*params.numFreqs) != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "strided batch failed" << std::endl;
      exit(1);
    }
  // Copy back to Swhole array
  cublasCcopy(cublasH,params.numParticles*params.numFreqs*numComps*numComps,
	      workArray.tmp,1,workArray.Swhole,1);

  // Same trick, but we need the product of the sub-transfer functions.
  // We determine the sub-inverse as above, first by scaling:
  scale_columns<<<gridSizeScale,blockSizeTF>>>(workArray.Tf,numComps,params.numParticles,params.numFreqs);
  if(cudaGetLastError() != cudaSuccess)
    {
      std::cout << "cuda kernel scale_columns failed" << std::endl;
      exit(1);
    }
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
  if(cudaGetLastError() != cudaSuccess)
    {
      std::cout << "cuda kernel shrinkArrays failed" << std::endl;
      exit(1);
    }

  // These each use the same dev_W, so it gets written twice.
  
  // Cholesky algorithm to determine the eigenvalues (we set it not to compute eigenvectors, it can)
  if(cusolverDnCheevjBatched(cusolverH,jobz,uplo,numComps-1,workArray.d_wholeSpec,numComps-1,
			     workArray.dev_W_whole, workArray.d_work2,lwork,workArray.d_info,
			     syevj_params, params.numFreqs*params.numParticles) != CUSOLVER_STATUS_SUCCESS)
    {
      std::cout << "cheev failed" <<  std::endl;
      exit(1);
    }
				     
  // Multiply the eigenvalues together to get the determinant. 
  //prodEigs<<<gridSizeProd,blockSizeProd,memsizeEig>>>(workArray.dev_W, workArray.det_whole, numComps-1, params.numParticles, params.numFreqs);
  // Repeat for the partial spectral matrices. 
  if(cusolverDnCheevjBatched(cusolverH,jobz,uplo,numComps-1,workArray.Spartial,numComps-1,
			     workArray.dev_W_partial, workArray.d_work2,lwork,workArray.d_info,
			     syevj_params, params.numFreqs*params.numParticles) != CUSOLVER_STATUS_SUCCESS)
    {
      std::cout << "cheev failed" << std::endl;
      exit(1);
    }
  
  // Compute the determinant.
  //prodEigs<<<gridSizeProd,blockSizeProd,memsizeEig>>>(workArray.dev_W, workArray.det_partial, numComps-1, params.numParticles, params.numFreqs);

  compGC<<<grdGRIDSIZE,grdBLOCKSIZE>>>(workArray.dev_GC,workArray.dev_W_partial,workArray.dev_W_whole,numComps-1,params.numParticles,params.numFreqs);
  
  if(cudaGetLastError() != cudaSuccess)
    {
      std::cout << "cuda kernel prodEigs failed" << std::endl;
      exit(1);
    }
  // Divides the determinants, takes the log, and adds to the integral. 
  //det2GC<<<gridSize_det2GC,blockSize_det2GC>>>(workArray.det_partial, workArray.det_whole, workArray.dev_GC,params.numParticles,params.numFreqs);
  //if(cudaGetLastError() != cudaSuccess)
  //  {
  //    std::cout << "cuda kernel det2GC failed" << std::endl;
  //    exit(1);
  //  }
  // Send the numParticles Granger causality values to the system memory.
  cudaMemcpy(GCvals.data(),workArray.dev_GC,sizeof(float)*params.numParticles,cudaMemcpyDeviceToHost);
  // Clean up (if you don't memory will leak).
  cusolverDnDestroy(cusolverH);
  cublasDestroy(cublasH);
  cudaFree(angles_dev);

  
  return;
}

void runFEHDstep(std::vector<float> &bestAngle, std::vector<float> &L, dataClass<float> dataArray ,paramContainer params,int numComps)
{

  // Determine the available memory on the GPU
  int id;
  size_t freemem,total;
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
  /*for(int row=0;row<rModel.numComps;row++)
    {
      for(int col=0;col<rModel.A.size()/rModel.numComps;col++)
  	std::cout << rModel.A[col*rModel.numComps+row] << " ";
     std::cout << std::endl;
     }*/
  // Copy to the vector that is returned (passed by reference, for now).
  std::copy(Dmat.begin(),Dmat.end(),L.begin());

  // This assumes that Dmat is the correct size - could it not be?
  // If Dmat is too small, this is the breaking point for filtered data.

  
  // The step-sizes to check along the (-)gradient.
  std::vector<float> h = {0.001f, 0.01f, 0.1f};

  // This stuff is all used below and should be put below.
  // A lot of it should be in for loop scope only. 
  std::vector<float> candidates(4,0);
  int minIndx;
  float allBlockMin=10000.0; // Just needs a somewhat large value. Will be set below.

  int minBlockNumber;
  // For recycling
  unsigned long int minimumGC;
  unsigned long int allBlockParticle;
  std::vector<int> resetList;
  std::vector<float> angleArrayReset;

  std::vector<float> GCvalsReset;

  paramContainer paramsReset = params;

  // Determine how to break up the analysis so that it fits on the GPU.
  // Query the GPU for memory situation
  int numBlocks, particleBlockSize;
  cudaMemGetInfo(&freemem, &total);

  if(params.verbose)
    {
      printf("memory free = %ld bytes \n",freemem);
      printf("total memory = %ld bytes \n",total);
    }
  // Add up all of the memory we need, and divide into equally sized blocks
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
  allocateParams(workArray,numComps,particleBlockSize,params,model.lagList,rModel.A);
   
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
      granger(angleArray[block],GCvals[block], paramsBLOCKED,numComps,workArray);
    }

  //for(int indx=0;indx<angleArray[0].size();indx++)
  //  std::cout << angleArray[0][indx] << std::endl;
  // Here is the iterator - adjustments occur here.
  // while STATIONARY_COUNT < COUNTMAX

  int STATIONARY_COUNT = 0;
  const int COUNTMAX = params.STUCKCOUNT;
  
  //for(int iter=0;iter<numIts;iter++)
  int iter = 0;
  while(STATIONARY_COUNT < COUNTMAX)
    {
      // Get a bunch of gradients
      //std::cout << " Heading in" << std::endl;
      for(int block=0;block<numBlocks;block++)
	compGradient(gradient[block],GCvals[block],angleArray[block],paramsBLOCKED,numComps,workArray);
      //std::cout << "Made it out" << std::endl;
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
	  granger(angleArray1[block],GCvals1[block],paramsBLOCKED,numComps,workArray);
	  granger(angleArray2[block],GCvals2[block],paramsBLOCKED,numComps,workArray);
	  granger(angleArray3[block],GCvals3[block],paramsBLOCKED,numComps,workArray);
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
	      granger(angleArrayReset,GCvalsReset,paramsReset,numComps,workArray);

	      for(int resetParticle=0;resetParticle<paramsReset.numParticles;resetParticle++)
		GCvals[block][resetList[resetParticle]]=GCvalsReset[resetParticle];
	    }
	}

      // Find the minimum over all of the blocks
      minBlockNumber = std::min_element(GCmin.begin(),GCmin.end())-GCmin.begin();
      if(allBlockMin <= GCmin[minBlockNumber])
	STATIONARY_COUNT++;
      else
	STATIONARY_COUNT = 0;
      
      allBlockMin = GCmin[minBlockNumber];
      allBlockParticle = minBlockNumber*particleBlockSize+GCminIndex[minBlockNumber];
            
      if(params.verbose)
      	printf("iteration = %i, particle = %li, value = %e, exit count = %i \n",
	       iter,allBlockParticle,allBlockMin,STATIONARY_COUNT);
      iter++;
    }
  
  // Return the best angle.

  long unsigned int indexVal = GCminIndex[minBlockNumber];

  //printf("%li \n",indexVal);

  std::copy(angleArray[minBlockNumber].data()+indexVal*(numComps-1),angleArray[minBlockNumber].data()+indexVal*(numComps-1)+numComps-1,bestAngle.begin());


  freeWorkArray(workArray);
  
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


  
  
