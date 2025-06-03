#include <cblas.h>
#include <lapacke.h>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include "utility.h"


void PGC(std::vector<float> dataArray, paramContainer params)
{
  // Size of large arrays 
  int numEpochs = params.numEpochs;
  int numComps = params.numChannels;
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
    
  std::vector<float> LS(2*numLags*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> RS(2*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> LCOV(2*numLags*2*numLags,0.0);
  std::vector<float> RCOV(2*numLags*2,0.0);
  int info;
  int IPIV[2*numLags];
  const float alpha=1.0;
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
	
	// I only have the upper triangular part of the covariance matrix. 
	// This should be ok.

	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2*numLags,2,numEpochs*(epochPts-maxLag),
		    alpha,LS.data(),2*numLags,RS.data(),2,0.0,RCOV.data(),2*numLags);
		    
	
	info = LAPACKE_ssysv(LAPACK_COL_MAJOR,'U',2*numLags,2,LCOV.data(),2*numLags,IPIV,
			     RCOV.data(),2*numLags);
	
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

  PGC(dataArray,params);
  return 0;
}

  
   
