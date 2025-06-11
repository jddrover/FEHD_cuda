#include <cblas.h>
#include <lapacke.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <complex>
#include <algorithm>
#include "utility.h"
#include <string>

float GCgen(std::vector<float> dataArray, std::vector<int> caused, paramContainer params)
{
  float GC;

  std::vector<float> freq(params.numFreqs,0.0);
  for(int findx=0;findx<params.numFreqs;findx++)
    freq[findx] = (params.freqHi-params.freqLo)/(params.numFreqs-1)*float(findx)+params.freqLo;

  std::sort(lagList.begin(),lagList.end());
  int maxLag = lagList[params.numLags-1];
  std::vector<float> RHS(params.numChannels*params.numEpochs*(params.epochPts-maxLag),0.0);
  std::vector<float> LHS(params.numChannels*params.numEpochs*params.numLags*(params.epochPts-maxLag),0.0);

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
      
  cblas_ssyrk(CblasColMajor,CblasUpper,CblasNoTrans,params.numChannels*params.numLags,
	      params.numEpochs*(params.epochPts-maxLag),
	      1.0,LHS.data(),params.numLags*params.numChannels,
	      0.0,LCOV.data(),params.numLags*params.numChannels);

  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,params.numChannels*params.numLags,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),
	      1.0,LHS.data(),params.numLags*params.numChannels,
	      RHS.data(),params.numChannels,
	      0.0,RCOV.data(),params.numChannels*params.numLags);

  info = LAPACKE_ssysv(LAPACK_COL_MAJOR,'U',params.numChannels*params.numLags,params.numChannels,
		       LCOV.data(),params.numChannels*params.numLags,IPIV,
			     RCOV.data(),params.numChannels*params.numLags);

  
