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
#include <cmath>
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
  /*for(int row=0;row<numComps*lagList.size();row++)
    {
      for(int col=0;col<(epochPts-maxLag)*numEpochs;col++)
	{
	  std::cout << LHSvec[col*numComps*lagList.size()+row] << " ";
	}
      std::cout << std::endl;
      }*/


  std::vector<float> RHSvec(RHS.elements);
  //std::vector<float> LHSvecBAK(LHS.elements);
  //std::vector<float> RHSvecBAK(RHS.elements);
  std::cout << "moo up top" << std::endl;
    for (int checker=0;checker<RHSvec.size();checker++)
    {
      if(std::isnan(RHSvec[checker]))
	{
	  std::cout << "found a nan" << std::endl;
	}
    }
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

  cublasStatus_t status;
  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  //cublasStatus_t errChk; // Put this in if there is a problem.
  // This can be changed to a rank update function - make sure uplo is upper.

  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  status = cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_T,mval,nval,kval,
	      &alpha,LHSvec_DEVICE,mval,LHSvec_DEVICE,mval,
	      &beta,LHScov_DEVICE,mval);
  if(status != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "cublasSgemm failed" << std::endl;
      exit(0);
    }
  std::vector<float> host_checker(numComps*numComps*numLags*numLags);
  cudaMemcpy(host_checker.data(),LHScov_DEVICE,sizeof(float)*numComps*numComps*numLags*numLags,cudaMemcpyDeviceToHost);
  /*for(int row=0;row<numComps*lagList.size();row++)
    {
      for(int col=0;col<numComps*lagList.size();col++)
	{
	  std::cout << host_checker[col*numComps*lagList.size()+row] << " ";
	}
      std::cout << std::endl;
      }*/
  
  // I want to know what this covariance looks like in terms of nan.
  

  status = cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_T,mval,numComps,kval,
	      &alpha, LHSvec_DEVICE,mval, RHSvec_DEVICE,numComps,
	      &beta, RHScov_DEVICE,mval);
  if(status != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "cublasSgemm failed" << std::endl;
      exit(0);
    }

  // I want to look for NaNs.

  
  cusolverDnHandle_t cusolverH = NULL;
  cusolverStatus_t csstatus;
  cusolverDnCreate(&cusolverH);
  int initVal = 10;// a dummy to avoid a warning without suppressing it.
  int *devCheck = &initVal;
  int Lwork=0;


  std::vector<double> LHS_cov_DOUBLE(host_checker.begin(),host_checker.end());
  std::cout << LHS_cov_DOUBLE.size() << std::endl;

  double *LHS_double_dev = new double[LHS_cov_DOUBLE.size()];
  cudaMalloc(&LHS_double_dev,sizeof(double)*numComps*numComps*lagList.size()*lagList.size());
  cudaMemcpy(LHS_double_dev,LHS_cov_DOUBLE.data(),sizeof(double)*numComps*numComps*lagList.size()*lagList.size(),
  	     cudaMemcpyHostToDevice);
  csstatus = cusolverDnDpotrf_bufferSize(cusolverH,uplo,numComps,LHS_double_dev,numComps, &Lwork);
  //delete(LHS_double_dev);
  //exit(0);
  // This is already written to use a symmetric matrix, so the output from above
  // Can be brought down directly. 
  //csstatus = cusolverDnSpotrf_bufferSize(cusolverH, uplo, numComps, LHScov_DEVICE, numComps, &Lwork);
    

/*if(csstatus != CUSOLVER_STATUS_SUCCESS)
    {
      std::cout << "cusolverDnSpotrf_buffersize failed" << std::endl;
      exit(0);
      }*/
 
  double *Workspace;
  cudaMalloc(&Workspace,sizeof(double)*Lwork);
  int *devInfo;
  cudaMalloc(&devInfo,sizeof(int));
  // Need some error checking here.
 
  csstatus = cusolverDnDpotrf(cusolverH, uplo, mval, LHS_double_dev, mval, Workspace, Lwork, devInfo);
  if(csstatus != CUSOLVER_STATUS_SUCCESS)
    {
      std::cout << "cusolveDnSpotrf failed" << std::endl;
      exit(0);
    }
  cudaMemcpy(devCheck,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
  if(*devCheck != 0)
    {
      std::cerr << "cusolverDnSpotrf passed info = " << *devCheck << std::endl;
      exit(0);
    }
   
  //csstatus = cusolverDnDpotrs(cusolverH, uplo, mval, numComps, LHS_double_dev, mval, RHScov_DEVICE, mval, devInfo);
  //if(csstatus != CUSOLVER_STATUS_SUCCESS)
  //  {
  //    std::cout << "cusolveDnSpotrs failed" << std::endl;
  //    exit(0);
  //  }
  // Check if devinfo gives anything
  
  //cudaMemcpy(devCheck,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
  //std::cout << "devCheck = " << *devCheck << std::endl;
  
  std::vector<float> A_result(numLags*numComps*numComps,0);
  cudaMemcpy(A_result.data(),RHScov_DEVICE,sizeof(float)*numComps*numComps*numLags,cudaMemcpyDeviceToHost);

  const float alphaRes = -1.0f;
  const float betaRes = 1.0f;
  
  status = cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numComps,kval,mval,
	      &alphaRes, RHScov_DEVICE, mval, LHSvec_DEVICE, mval,
	      &betaRes, RHSvec_DEVICE, numComps);
  if(status != CUBLAS_STATUS_SUCCESS)
    {
      std::cout << "cublasSgemm failed" << std::endl;
      exit(0);
    }
  std::vector<float> residualsTmp(numComps*numEpochs*(epochPts-maxLag),0);
  
  cudaMemcpy(residualsTmp.data(),RHSvec_DEVICE,sizeof(float)*numComps*numEpochs*(epochPts-maxLag),cudaMemcpyDeviceToHost);
  
  
  convertRawArrayToDataList(residualsTmp.data(),R,numComps,epochPts-maxLag, numEpochs);
  std::cout << "moo" << std::endl;
  std::cout << residualsTmp[0] << std::endl;
  
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
  std::cout << "orthonormalization" << std::endl;
  std::cout << residuals.epochArray[0].timePointArray[0].dataVector[0] << std::endl;
  PCA(residuals, ortho_residuals, L);
}

void rotate_model(ARmodel &A, matrix L)
{

  int M = sqrt(L.elements.size()); // Matrix dimension
  int numLags = A.lagMatrices.size(); // number of lags
  int info1,info2; // Error checking
  matrix LBAK; // The original will be changing, it is easiest to just make a copy. 
  LBAK.elements = L.elements;

  for(int row=0;row<M;row++)
    {
      for(int col=0;col<M;col++)
	{
	  std::cout << L.elements[col*M+row];
	}
      std::cout << "\n";
    }



  
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
	
