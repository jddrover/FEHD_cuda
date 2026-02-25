#include <iostream>
#include "dataContainers.h"
#include "mkARGPU.h"
#include "kernels.h"
#include <vector>
#include <cblas.h>
#include <lapacke.h>
#include <algorithm>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "timeSeriesOPs.h"
#include "dataClass.h"

MVAR<float> mkARGPU(dataClass<float> dataArray,paramContainer params)
{
  // I have the means to pass parameters, I find these objects I made intrusive.

  int numEpochs = dataArray.getNumEpochs();
  int epochPts = dataArray.getEpochPoints();
  int numComps = dataArray.getNumComps();
  
  // Determine the maximum lag and sort the lags into increasing order.
  std::vector<int> lagList(params.lagList);
  int maxLag = *std::max_element(lagList.begin(),lagList.end());
  std::sort(lagList.begin(),lagList.end());
  int numLags = lagList.size();

  int epochAdj = epochPts-maxLag; // The lagged epochs are maxLag shorter.
  std::vector<float> RHS(epochAdj*numEpochs*numComps,0.0);
  std::vector<float> LHS(epochAdj*numEpochs*numComps*numLags,0.0);

  for(int epoch=0;epoch<numComps;epoch++)
    {
      std::vector<float> epochData = dataArray.isoEpoch(epoch).dataArray();
      std::copy(epochData.begin()+numComps*maxLag,epochData.end(),RHS.begin()+epoch*epochAdj*numComps);
      for(int tp=maxLag;tp<epochPts;tp++)
	for(int lagindx=0;lagindx<lagList.size();lagindx++)
	  std::copy(epochData.begin()+(tp-lagList[lagindx])*numComps,
		    epochData.begin()+(tp-lagList[lagindx]+1)*numComps,
		    LHS.begin()+(epoch*epochAdj+tp-maxLag)*numComps*numLags+lagindx*numComps);
    }
  // Transpose these
  std::vector<float> LHST(LHS.size());
  for(int rowindx=0;rowindx<epochAdj*numEpochs;rowindx++)
    for(int colindx=0;colindx<numComps*numLags;colindx++)
      LHST[rowindx+colindx*epochAdj*numEpochs] = LHS[colindx+rowindx*numComps*numLags];
      
  std::vector<float> RHST(RHS.size());
  for(int rowindx=0;rowindx<epochAdj*numEpochs;rowindx++)
    for(int colindx=0;colindx<numComps;colindx++)
      RHST[rowindx+colindx*epochAdj*numEpochs] = RHS[colindx+rowindx*numComps];
  
  float *LHS_DEVICE = nullptr;
  float *RHS_DEVICE = nullptr;
  float *LHS_DEVICE_BACKUP = nullptr; // gesvd destroys the input, and we need it later.

  cublasHandle_t cublasH = 0;
  cublasCreate(&cublasH);

  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  
  cudaMalloc(&LHS_DEVICE,sizeof(float)*LHST.size());
  cudaMalloc(&RHS_DEVICE,sizeof(float)*RHST.size());
  cudaMalloc(&LHS_DEVICE_BACKUP,sizeof(float)*LHST.size());
  cudaMemcpy(LHS_DEVICE,LHST.data(),sizeof(float)*LHST.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(RHS_DEVICE,RHST.data(),sizeof(float)*RHST.size(),cudaMemcpyHostToDevice);
  // Fill in the backup (I am guessing that this is faster than another cudaMemcpy. 
  cublasScopy(cublasH,LHST.size(),LHS_DEVICE,1,LHS_DEVICE_BACKUP,1);

  // Determine the size of the work array needed, and create it.
  int Lwork=0;
  cusolverDnSgesvd_bufferSize(cusolverH,numEpochs*epochAdj,numLags*numComps,&Lwork);   
  float *Workspace = nullptr;
  cudaMalloc(&Workspace,sizeof(float)*Lwork);
  // Problem dimensions, to save typing.
  int m = numEpochs*epochAdj;
  int n = numLags*numComps;

  float *d_rwork = nullptr; // This is an option. I really don't want to deal with it.
  int *devInfo = nullptr;
  cudaMalloc(&devInfo,sizeof(int));
  //SVD matrices - returns V transposed.
  float *U = nullptr;
  float *S = nullptr;
  float *VT = nullptr;

  cudaMalloc(&U,sizeof(float)*m*n);
  cudaMalloc(&S,sizeof(float)*n);
  cudaMalloc(&VT,sizeof(float)*n*n);
  // Compute the SVD
  cusolverDnSgesvd(cusolverH,'S','S',m,n,
		   LHS_DEVICE,m,S,U,m,VT,n,
		   Workspace,Lwork,d_rwork,devInfo);

  // Bounce out to the host.
  // Doing this whole thing on the GPU instead of the CPU might be stupid.
  std::vector<float> Shost(n);
  cudaMemcpy(Shost.data(),S,sizeof(float)*n,cudaMemcpyDeviceToHost);

  if(params.verbose)
    std::cout << "Smallest Singular Value: " << Shost[n-1] << std::endl;

  float tol;

  float Ptol = params.Ptol;
  float P = 0.0;
  float singValSum = 0.0;
  if(Ptol < 1.0)
    {
      for(int indx=0;indx<n;indx++)
	singValSum += Shost[indx];
      for(int indx=0;indx<n;indx++)
	{
	  P += Shost[indx]/singValSum;
	  if(P>Ptol)
	    {
	      tol = Shost[indx];
	      break;
	    }
	}
    }
  else
    tol = 0.0;
  
  if(params.verbose)
    std::cout << "Tolerance is : " << tol << std::endl;
  
  float *UTb = nullptr;
  cudaMalloc(&UTb,sizeof(float)*n*numComps);
  
  const float alphaY = 1.0;
  const float betaY = 0.0;
  
  cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,n,numComps,m,
	      &alphaY,U,m,RHS_DEVICE,m,
	      &betaY,UTb,n);
    
  int blksize = 1024;
  int grdsize = (int)(m*n+blksize-1)/blksize;
  const dim3 blockSize(blksize);
  const dim3 gridSize(grdsize);

  // How should this tol be chosen?
  scaleByS<<<gridSize,blockSize>>>(S,UTb,n,numComps,tol);
  std::vector<float> UTbhost(n*numComps);
  //cudaMemcpy(UTbhost.data(),UTb,sizeof(float)*n*numComps,cudaMemcpyDeviceToHost);
  //  for(int indx=0;indx<n*numComps;indx++)
  //std::cout << UTbhost1[indx] << " " << UTbhost[indx] << std::endl;

  float *Adev = nullptr;
  cudaMalloc(&Adev,sizeof(float)*n*numComps); 
  cublasSgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,n,numComps,n,
	      &alphaY,VT,n,UTb,n,
	      &betaY,Adev,n);

  
  
  const float alphaRes = -1.0f;
  const float betaRes = 1.0f;
  
  cublasSgemm(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,m,numComps,n,
	      &alphaRes,LHS_DEVICE_BACKUP,m, Adev,n,
	      &betaRes, RHS_DEVICE,m);
  // RHS_DEVICE now contains the residuals.
  
  std::vector<float> residualsTmp(m*numComps,0);
  std::vector<float> Rout(m*numComps,0);
  cudaMemcpy(residualsTmp.data(),RHS_DEVICE,sizeof(float)*numComps*m,cudaMemcpyDeviceToHost);

  for(int row=0;row<numComps;row++)
    for(int col=0;col<m;col++)
      Rout[col*numComps+row] = residualsTmp[row*m+col];

  std::vector<float> Aout(numComps*numComps*numLags);
  std::vector<float> Ahost(numLags*numComps*numComps,0);
  cudaMemcpy(Ahost.data(),Adev,sizeof(float)*n*numComps,cudaMemcpyDeviceToHost);
  for(int row=0;row<numComps;row++)
    for(int col=0;col<numComps*numLags;col++)
      Aout[col*numComps+row] = Ahost[row*numComps*numLags+col];
 

  MVAR<float> toReturn(Aout,Rout,numComps,lagList);
  
  cudaFree(LHS_DEVICE);
  cudaFree(RHS_DEVICE);
  cudaFree(LHS_DEVICE_BACKUP);
  cudaFree(U);
  cudaFree(S);
  cudaFree(VT);
  cudaFree(UTb);
  cudaFree(Adev);
  cudaFree(Workspace);
  cudaFree(devInfo);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  return toReturn;
}

void orthonormalizeR(dataList residuals, dataList &ortho_residuals, matrix &L)
{
  PCA(residuals, ortho_residuals, L);
}

MVAR<float> rotate_model(MVAR<float> model, std::vector<float> L)
{
  int numComps = model.numComps;
  int numLags = model.lagList.size();
  // Check that the vector L is the correct size
  int N = model.R.size()/numComps;
  if(L.size() != numComps*numComps)
    {
      std::cout << "When transforming the model to have orthonormal residuals, the transformation matrix is the wrong size" << std::endl;
      exit(1);
    }
  
  int info1,info2; // Error checking
  std::vector<float> LBAK(L);

  // Invert the matrix
  // Uses LAPACK to LU=A (trf) and back solve (tri)
  std::vector<int> ipiv(numComps,0);
  
  info1 = LAPACKE_sgetrf(LAPACK_COL_MAJOR,numComps,numComps,L.data(),numComps,ipiv.data());
  info2 = LAPACKE_sgetri(LAPACK_COL_MAJOR,numComps,L.data(),numComps,ipiv.data());
  if(info1 != 0 || info2 != 0)
    {
      std::cout << "When transforming the model to have orthonormal residuals, the transformation matrix did not invert" << std::endl;
      exit(1);
    }
  
  // Multiply L A L^-1 for each lag matrix in the AR model
  const float alpha=1.0f;
  const float beta=0.0f;

  std::vector<float> Aout(model.A);
  for(int lag=0; lag<numLags; lag++)
    {
      std::vector<float> lagMat = model.getLag(lag);
      std::vector<float> tmp(numComps*numComps,0.0);
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  numComps,numComps,numComps,alpha,LBAK.data(),numComps,
		  lagMat.data(),numComps,// Left here.
		  beta, tmp.data(),numComps);
      
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  numComps,numComps,numComps, alpha,tmp.data(),numComps,
		  L.data(),numComps,
		  beta,Aout.data()+lag*numComps*numComps,numComps);
    }
  std::vector<float> Rout(model.R);
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
	      numComps,N,numComps,
	      alpha,LBAK.data(),numComps,model.R.data(),numComps,
	      beta,Rout.data(),numComps);

  
  
  MVAR<float> ARout(Aout,Rout,numComps,model.lagList);
  return
}
	
