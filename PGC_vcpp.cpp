#include <cblas.h>
#include <lapacke.h>
#include <random>
#include <vector>
#include <iostream>
#include <complex>
//#include <cmath>
#include <algorithm>
#include "utility.h"
#include <numbers>


void PGC(std::vector<float> dataArray, paramContainer params)
{
  // Size of large arrays 
  int numEpochs = params.numEpochs;
  int numComps = params.numChannels;
  
  std::vector<float> freq(params.numFreqs);
  for(int findx=0;findx<params.numFreqs;findx++)
    freq[findx] = (params.freqLo-params.freqHi)/(params.numFreqs-1)*float(findx)+params.freqLo;

  // Check that this is an integer
  int epochPts = params.epochPts;
  std::vector<int> lagList(params.lagList);
  std::sort(lagList.begin(),lagList.end());
  int numLags = lagList.size();
  int maxLag = lagList[numLags-1];
  std::vector<float> RHS(numComps*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> LHS(numComps*numLags*numEpochs*(epochPts-maxLag),0.0);
  
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      std::copy(dataArray.begin()+(epoch*epochPts+maxLag)*numComps,dataArray.begin()+(epoch+1)*epochPts*numComps,
		RHS.begin()+epoch*(epochPts-maxLag)*numComps);

      for(int lag=0;lag<numLags;lag++)
      {
        for(int tp=0;tp<epochPts-maxLag;tp++)
          {
            std::copy(dataArray.begin()+epoch*numComps*epochPts+(maxLag-lagList[lag])*numComps+tp*numComps,
      		dataArray.begin()+epoch*numComps*epochPts+(maxLag-lagList[lag])*numComps+(tp+1)*numComps,
      		LHS.begin()+epoch*numComps*numLags*(epochPts-maxLag)+lag*numComps+tp*numComps*numLags);
          }
      }
    }
  float dt = 1.0/((float)params.sampRate);
  float argmtBASE = -2.0*M_PI*dt;
  std::complex<float> argmt;
  std::vector<float> LS(2*numLags*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> RS(2*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> RS2(2*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> LCOV(2*numLags*2*numLags,0.0);
  std::vector<float> RCOV(2*numLags*2,0.0);
  std::vector<float> A(4*numLags,0.0);
  std::vector<float> resCOV(4,0.0);
  std::vector<float> Pmat(4,0.0);
  std::vector<float> Pinv(4,0.0);
  std::vector<float> resCOVinv(4,0.0);
  std::vector<std::complex<float>> RI(4,std::complex<float>(0.0,0.0));
  std::vector<std::complex<float>> Tf(4*params.numFreqs,std::complex<float>(0.0,0.0));
  std::complex<float> alphaC(1.0,0.0);
  std::complex<float> betazero(0.0,0.0);
  std::vector<std::complex<float>> Swhole(4*params.numFreqs,std::complex<float>(0.0,0.0));
  std::vector<std::complex<float>> Stmp(4,std::complex<float>(0.0,0.0));
  float tmp;
  int info;
  int IPIV[2*numLags];
  const float alpha=1.0;
  const float oneoverN = 1.0/((float)(numEpochs*(epochPts-maxLag)));
  float wholeSpec;
  for(int pair1=0;pair1<numComps;pair1++)
    for(int pair2=0;pair2<numComps;pair2++)
      {
	if(pair1==pair2)
	  continue;

	cblas_scopy(numLags*numEpochs*(epochPts-maxLag),LHS.data()+pair1,numComps,LS.data(),2);
	cblas_scopy(numLags*numEpochs*(epochPts-maxLag),LHS.data()+pair2,numComps,LS.data()+1,2);

	cblas_scopy(numEpochs*(epochPts-maxLag),RHS.data()+pair1,numComps,RS.data(),2);
	cblas_scopy(numEpochs*(epochPts-maxLag),RHS.data()+pair2,numComps,RS.data()+1,2);

	cblas_ssyrk(CblasColMajor,CblasUpper,CblasNoTrans,2*numLags,numEpochs*(epochPts-maxLag),
		    alpha,LS.data(),2*numLags,0.0,LCOV.data(),2*numLags);

	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2*numLags,2,numEpochs*(epochPts-maxLag),
		    alpha,LS.data(),2*numLags,RS.data(),2,0.0,RCOV.data(),2*numLags);
	
	info = LAPACKE_ssysv(LAPACK_COL_MAJOR,'U',2*numLags,2,LCOV.data(),2*numLags,IPIV,
			     RCOV.data(),2*numLags);
	
	for(int col=0;col<numLags*2;col++)
	  {
	    cblas_scopy(2,RCOV.data()+col,2*numLags,A.data()+col*2,1);
	  }
	
	// Compute the residuals
	// RHS-A*LHS

	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,numEpochs*(epochPts-maxLag),2*numLags,
		    -1.0,A.data(),2,LS.data(),2*numLags,1.0,RS.data(),2);

	
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2,2,numEpochs*(epochPts-maxLag),
		    oneoverN,RS.data(),2,RS.data(),2,0.0,resCOV.data(),2);
	
	Pmat[0]=1.0;
	Pmat[1]=-resCOV[2]/resCOV[0];
	Pmat[2]=0.0;
	Pmat[3]=1.0;
	
	Pinv[0]=1.0;
	Pinv[1]=-Pmat[1];
	Pinv[2]=0.0;
	Pinv[3]=Pmat[3];

	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2*numLags,2,
		    1.0,Pmat.data(),2,A.data(),2,0.0,RCOV.data(),2);

	for(int lag=0;lag<numLags;lag++)
	  {
	    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2,2,
			1.0,RCOV.data()+lag*4,2,Pinv.data(),2,0.0,A.data()+lag*4,2);
	  }

	// Apply PR
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,numEpochs*(epochPts-maxLag),2,
		    1.0,Pmat.data(),2,RS.data(),2,0.0,RS2.data(),2);
	// Compute covariance.
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2,2,numEpochs*(epochPts-maxLag),
		    oneoverN,RS2.data(),2,RS2.data(),2,0.0,resCOV.data(),2); // Reusing resCOV
	//std::cout << "resCov matrix" << std::endl;
	//std::cout << resCOV[0] << " " << resCOV[2] << std::endl;
	//std::cout << resCOV[1] << " " << resCOV[3] << std::endl;
	// Need to invert this
	tmp = resCOV[0]*resCOV[3]-resCOV[1]*resCOV[2];
	resCOVinv[0] = resCOV[3]/tmp;
	resCOVinv[1] = -resCOV[1]/tmp;
	resCOVinv[2] = -resCOV[2]/tmp;
	resCOVinv[3] = resCOV[0]/tmp;
	RI[0] = std::complex<float>(resCOVinv[0],0.0);
	RI[1] = std::complex<float>(resCOVinv[1],0.0);
	RI[2] = std::complex<float>(resCOVinv[2],0.0);
	RI[3] = std::complex<float>(resCOVinv[3],0.0);
	//std::cout << "resCovinv matrix" << std::endl;
	//std::cout << resCOVinv[0] << " " << resCOVinv[2] << std::endl;
	//std::cout << resCOVinv[1] << " " << resCOVinv[3] << std::endl;
	
	
	for(int findx=0;findx<params.numFreqs;findx++)
	  {
	    Tf[findx*4]=std::complex<float>(1.0,0.0);
	    Tf[findx*4+3]=std::complex<float>(1.0,0.0);

	    for(int lag=0;lag<params.numLags;lag++)
	      {
		argmt = -std::exp(std::complex<float>(0.0,argmtBASE*(float)lagList[lag]*freq[findx]));
		cblas_caxpy(4,&argmt,A.data()+lag*4,1,Tf.data()+findx*4,1);
	      }
	    //std::cout << "Tf matrix " << std::endl;
	    //std::cout << Tf[0] << " " << Tf[2] << std::endl;
	    //std::cout << Tf[1] << " " << Tf[3] << std::endl;
	    cblas_cgemm(CblasColMajor,CblasConjTrans,CblasNoTrans,2,2,2,
			&alphaC,Tf.data()+findx*4,2,RI.data(),2,// Might want a complex array.
			&betazero,Stmp.data(),2);
	    //std::cout << "tmp matrix " << std::endl;
	    //std::cout << Stmp[0] << " " << Stmp[2] << std::endl;
	    //std::cout << Stmp[1] << " " << Stmp[3] << std::endl;
	    cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2,2,
			&alphaC,Stmp.data(),2,Tf.data()+4*findx,2,
			&betazero,Swhole.data()+4*findx,2);

	    //std::cout << "whole mat " << std::endl;
	    //std::cout << Swhole[0] << " " << Swhole[2] << std::endl;
	    //std::cout << Swhole[1] << " " << Swhole[3] << std::endl;
	    wholeSpec = (Swhole[findx*4]-Swhole[findx*4+2]*Swhole[findx*4+1]/Swhole[findx*4+3]).real();
	    //std::cout << "whole spec = " << wholeSpec << std::endl;
	  }

	

	/*	std::cout << "Tf f1" << std::endl;
     
	std::cout << Tf[0] << " " << Tf[2] << std::endl;
	std::cout << Tf[1] << " " << Tf[3] << std::endl;
	*/

      }
  return;
  }

int main()
{
  int numComps = 16;
  int numEpochs = 200;
  int epochPts = 750;
  int numLags = 30;
  int N = numEpochs*epochPts;
  std::random_device gen;
  std::default_random_engine generator(gen());
  std::normal_distribution<float> normDist(0.0,1.0);

  std::vector<float> dataArray(numComps*N,0.0);
  for(int indx=0;indx<numComps*N;indx++)
    
      dataArray[indx] = normDist(generator);
  
  // Create a dummy data array
  // Create parameter structure.
  paramContainer params;
  params.epochPts=epochPts;
  params.numLags =numLags;
  std::vector<int> lagList(numLags,0);
  for(int lag=0;lag<numLags;lag++)
    lagList[lag] = lag+1;
  params.numChannels = 16;
  params.lagList = lagList;
  params.numEpochs = numEpochs;
  params.numFreqs = 13;
  params.freqLo = 8.0;
  params.freqHi = 12.0;
  params.sampRate = 500;
  PGC(dataArray,params);
  return 0;
}

  
   
