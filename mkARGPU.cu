#include <iostream>
#include "dataContainers.h"
#include "mkARGPU.h"
#include <vector>
#include <cblas.h>
#include <lapacke.h>
#include <algorithm>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "timeSeriesOPs.h"
void mkARGPU(dataList dataArray, std::vector<int> lagList, ARmodel &A, dataList &R)
{
  int numEpochs = dataArray.epochArray.size();
  int epochPts = dataArray.epochArray[0].timePointArray.size();
  int numComps = dataArray.epochArray[0].timePointArray[0].dataVector.size();
  //int numPoints = numEpochs*epochPts;
  
  //matrix LHS;
  //matrix RHS;

  // We want maxLag, not numLags
  
  int maxLag = *std::max_element(lagList.begin(),lagList.end());
  int tpoint;
  // Sort the lag list in reverse order, so everything matches up without changing a lot.
  //printf("max lag= %i \n",maxLag);
  std::sort(lagList.begin(),lagList.end(), std::greater<int>());


  // Change this so that it is the transpose of this.

  std::vector<float> RHS((epochPts-maxLag)*numEpochs*numComps,0.0);
  std::vector<float> LHS((epochPts-maxLag)*numEpochs*numComps*lagList.size(),0.0);
  
  int epochAdj = epochPts-maxLag;
  int tpUse;
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      for(int tp=maxLag;tp<epochPts;tp++)
	{
	  tpUse = tp - maxLag;
	  for(int comp=0;comp<numComps;comp++)
	    {
	      RHS[epoch*epochAdj+tpUse+comp*numEpochs*epochAdj]=
		dataArray.epochArray[epoch].timePointArray[tp].dataVector[comp];
	    }
	}
      for(int lagIndx=0;lagIndx<lagList.size();lagIndx++)
	{
	  for(int tp=0;tp<epochAdj;tp++)
	    {
	      tpUse = tp+maxLag-lagList[lagIndx];
	      for(int comp=0;comp<numComps;comp++)
		{
		  LHS[epoch*epochAdj+tp+lagIndx*numComps*numEpochs*epochAdj+
		      comp*numEpochs*epochAdj]=
		    dataArray.epochArray[epoch].timePointArray[tpUse].dataVector[comp];
		}
	    }
	}
    }

  int numLags = lagList.size();
  
  int mval = numLags*numComps;
  int nval = numLags*numComps;
  int kval = (epochPts-maxLag)*numEpochs;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  float *LHS_DEVICE = nullptr;
  float *RHS_DEVICE = nullptr;
  float *LHS_DEVICE_BACKUP = nullptr; // gesvd destroys the input, and we need it later.

  cudaMalloc(&LHS_DEVICE,sizeof(float)*LHS.size());
  cudaMalloc(&RHS_DEVICE,sizeof(float)*RHS.size());
  cudaMalloc(&LHS_DEVICE_BACKUP,sizeof(float)*LHS.size());
  cudaMemcpy(LHS_DEVICE,LHS.data(),sizeof(float)*LHS.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(RHS_DEVICE,RHS.data(),sizeof(float)*RHS.size(),cudaMemcpyHostToDevice);

  cublasScopy(LHS.size(),LHS_DEVICE,1,LHS_DEVICE_BACKUP,1);
  // Create svd with this.

  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
    
  int Lwork=0;
  // This is already written to use a symmetric matrix, so the output from above
  // Can be brought down directly.

  cusolverDnSgesvd_bufferSize(cusolverH,numEpochs*epochAdj,numLags*numComps,&Lwork);
   
  float *Workspace;
  cudaMalloc(&Workspace,sizeof(float)*Lwork);

  int m = numEpochs*epochAdj;
  int n = numLags*numComps;

  float *d_rwork = nullptr; // This is an option.
  int *devInfo = nullptr;
  cudaMalloc(&devInfo,sizeof(int));

  float *U = nullptr;
  float *S = nullptr;
  float *VT = nullptr;

  cudaMalloc(&U,sizeof(float)*m*n);
  cudaMalloc(&S,sizeof(float)*n);
  cudaMalloc(&VT,sizeof(float)*n*n);

  
  
  cusolverDnSgesvd(cusolverH,'S','S',m,n,
		   LHS_DEVICE,m,Sdiag,U,m,VT,n,
		   Workspace,Lwork,d_rwork,devInfo);
		   
  float *UTb = nullptr;
  cudaMalloc(&UTb,sizeof(float)*n*numComps);

  cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,n,m,numComps,
	      &alphaY,U,m,RHS_device,m,
	      &betaY,UTb,n);

  int blksize = 1024;
  int grdsize = (int)(m*n+blksize-1)/blksize;
  const dim3 blockSize(blksize);
  const dim3 gridSize(grdsize);

  scaleByS<<<gridSize,blockSize>>>(S,UTb,n,numComps);

  float *Adev = nullptr;
  cudaMalloc(&Adev,sizeof(float)*n*numComps); 
  cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,n,n,numComps,
	      &alphaY,VT,n,UTb,n,
	      &betaY,Adev,n);

  std::vector<float> A_result(numLags*numComps*numComps,0);
  cudaMemcpy(A_result.data(),Adev,sizeof(float)*n*numComps,cudaMemcpyDeviceToHost);

  const float alphaRes = -1.0f;
  const float betaRes = 1.0f;
  
  cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,m,n,numComps,
	      &alphaRes,LHS_DEVICE_BACKUP,m, Adev,n,
	      &betaRes, RHS_DEVICE,m);
  // RHS_DEVICE now contains the residuals.
  
  std::vector<float> residualsTmp(m*numComps,0);

  cudaMemcpy(residualsTmp.data(),RHS_DEVICE,sizeof(float)*numComps*m,cudaMemcpyDeviceToHost);

  // This data list stuff is a pain in the ass. 
  convertRawArrayToDataList(residualsTmp.data(),R,numComps,epochPts-maxLag, numEpochs); 

  matrix lagTmp;

  for(int lag=numLags-1;lag>=0;lag--)
    {
      for(int col=0;col<numComps;col++)
	for(int row=0;row<numComps;row++)
	  {
	    // Transpose and copy.
	    lagTmp.elements.push_back(A_result[row*numLags*numComps+col+lag*numComps]);
	  }

      A.lagMatrices.push_back(lagTmp);
      lagTmp.elements.clear();
    }


  cudaFree(LHSvec_DEVICE);
  cudaFree(RHSvec_DEVICE);
  cudaFree(LHScov_DEVICE);
  cudaFree(RHScov_DEVICE);
  cudaFree(Workspace);
  cudaFree(devInfo);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  return;
}

void orthonormalizeR(dataList residuals, dataList &ortho_residuals, matrix &L)
{
  PCA(residuals, ortho_residuals, L);
}

void rotate_model(ARmodel &A, matrix L)
{

  int M = sqrt(L.elements.size()); // Matrix dimension
  int numLags = A.lagMatrices.size(); // number of lags
  int info1,info2; // Error checking
  matrix LBAK; // The original will be changing, it is easiest to just make a copy. 
  LBAK.elements = L.elements;

  // Invert the matrix
  // Uses LAPACK to LU=A (trf) and back solve (tri)
  std::vector<int> ipiv(M,0);
  
  info1 = LAPACKE_sgetrf(LAPACK_COL_MAJOR,M,M,L.elements.data(),M,ipiv.data());
  info2 = LAPACKE_sgetri(LAPACK_COL_MAJOR,M,L.elements.data(),M,ipiv.data());
  if(info1 != 0 || info2 != 0)
    {
      // Put some diagnostics here - 

      printf("Error inverting the residual transformation matrix \n");
      exit(0);
    }
  /*printf("The inverse \n");
  for(int row=0;row<M;row++)
    {
      for(int col=0; col<M;col++)
	{
	  printf("%f ",L.elements[col*M+row]);

	}
      printf("\n");
      }*/
  
  // Multiply L A L^-1 for each lag matrix in the AR model
  const float alpha=1.0f;
  const float beta=0.0f;

  std::vector<float> tmp(M*M,0);
  for(int lag=0; lag<numLags; lag++)
    {
      
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  M,M,M,alpha,LBAK.elements.data(), M,
		  A.lagMatrices[lag].elements.data(),M,
		  beta, tmp.data(),M);
      

      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  M,M,M, alpha,tmp.data(),M,
		  L.elements.data(),M,
		  beta, A.lagMatrices[lag].elements.data(),M);
    }
  
  
  return;
}
	
