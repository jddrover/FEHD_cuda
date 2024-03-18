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
  
  matrix LHS;
  matrix RHS;

  // We want maxLag, not numLags
  
  int maxLag = *std::max_element(lagList.begin(),lagList.end());
  int tpoint;
  // Sort the lag list in reverse order, so everything matches up without changing a lot.
  //printf("max lag= %i \n",maxLag);
  std::sort(lagList.begin(),lagList.end(), std::greater<int>());
  
  for(int epoch=0;epoch<numEpochs;epoch++)
      {
      for(int tp=maxLag;tp<epochPts;tp++)
	{
	  RHS.elements.insert(RHS.elements.end(),
			      dataArray.epochArray[epoch].timePointArray[tp].dataVector.begin(),
			      dataArray.epochArray[epoch].timePointArray[tp].dataVector.end());
	}

      for(int tp=0;tp<epochPts-maxLag;tp++)
	{
	  for(int lagIndx=0;lagIndx<lagList.size();lagIndx++)
	    {
	      tpoint = tp + maxLag-lagList[lagIndx];
	      LHS.elements.insert(LHS.elements.end(),
				  dataArray.epochArray[epoch].timePointArray[tpoint].dataVector.begin(),
				  dataArray.epochArray[epoch].timePointArray[tpoint].dataVector.end());
	    }
	}

      }

  // Make some copies (probably some of them are unnecessary.
  std::vector<float> LHSvec(LHS.elements);
  std::vector<float> RHSvec(RHS.elements);
  //std::vector<float> LHSvecBAK(LHS.elements);
  //std::vector<float> RHSvecBAK(RHS.elements);

  int numLags = lagList.size();
  
  int mval = numLags*numComps;
  int nval = numLags*numComps;
  int kval = (epochPts-maxLag)*numEpochs;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  float *LHSvec_DEVICE;
  float *RHSvec_DEVICE;
  float *LHScov_DEVICE;
  float *RHScov_DEVICE;

  cudaMalloc(&LHSvec_DEVICE,sizeof(float)*LHSvec.size());
  cudaMalloc(&RHSvec_DEVICE,sizeof(float)*RHSvec.size());
  cudaMemcpy(LHSvec_DEVICE,LHSvec.data(),sizeof(float)*LHSvec.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(RHSvec_DEVICE,RHSvec.data(),sizeof(float)*RHSvec.size(),cudaMemcpyHostToDevice);
  
  cudaMalloc(&LHScov_DEVICE,sizeof(float)*numComps*numComps*numLags*numLags);
  cudaMalloc(&RHScov_DEVICE,sizeof(float)*numComps*numComps*numLags);

  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  //cublasStatus_t errChk; // Put this in if there is a problem.
  // This can be changed to a rank update function - make sure uplo is upper.

  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_T,mval,nval,kval,
	      &alpha,LHSvec_DEVICE,mval,LHSvec_DEVICE,mval,
	      &beta,LHScov_DEVICE,mval);


  // This one needs to stay the same.
  cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_T,mval,numComps,kval,
	      &alpha, LHSvec_DEVICE,mval, RHSvec_DEVICE,numComps,
	      &beta, RHScov_DEVICE,mval);


  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
    
  int Lwork=0;
  // This is already written to use a symmetric matrix, so the output from above
  // Can be brought down directly. 
  cusolverDnSpotrf_bufferSize(cusolverH, uplo, numComps, LHScov_DEVICE, numComps, &Lwork);
	
  float *Workspace;
  cudaMalloc(&Workspace,sizeof(float)*Lwork);
  int *devInfo;
  cudaMalloc(&devInfo,sizeof(int));
  // Need some error checking here.
  cusolverDnSpotrf(cusolverH, uplo, mval, LHScov_DEVICE, mval, Workspace, Lwork, devInfo);
  
  cusolverDnSpotrs(cusolverH, uplo, mval, numComps, LHScov_DEVICE, mval, RHScov_DEVICE, mval, devInfo);

  std::vector<float> A_result(numLags*numComps*numComps,0);
  cudaMemcpy(A_result.data(),RHScov_DEVICE,sizeof(float)*numComps*numComps*numLags,cudaMemcpyDeviceToHost);

  const float alphaRes = 1.0f;
  const float betaRes = -1.0f;
  
  cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numComps,kval,mval,
	      &alphaRes, RHScov_DEVICE, mval, LHSvec_DEVICE, mval,
	      &betaRes, RHSvec_DEVICE, numComps);

  std::vector<float> residualsTmp(numComps*numEpochs*(epochPts-maxLag),0);

  cudaMemcpy(residualsTmp.data(),RHSvec_DEVICE,sizeof(float)*numComps*numEpochs*(epochPts-maxLag),cudaMemcpyDeviceToHost);

  
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
	
