#include <iostream>

#include "FEHD.h"
#include "dataContainers.h"
#include "mkARGPU.h"
#include <algorithm>
#include <math.h>
#include <fstream>
#include "GC.h"
#include "utility.h"
#include <chrono>
#include <cblas.h>
#include "timeSeriesOPs.h"
// Main call. Executes the FEHD algorithm
void runFEHD(dataList dataArray, std::vector<float> &Lmat, paramContainer params)
{
  // Set the parameters for sgemm. 
  float alpha=1.0f;
  float beta=0.0f;

  
  std::vector<float> bestAngle;
  matrix Rdecor; // Another example - it's the pca function.
  std::vector<float> Q; // The rotation matrix, resized at each step.
  std::vector<float> T(params.numPCs*params.numPCs,0); // The "work" transformation
  std::vector<float> oneArrayData; // Holds the data without epoch boundaries.
  std::vector<float> transformedData; // Holds the new data without epoch boundaries.  
  std::vector<float> newTrans(params.numPCs*params.numChannels); // Another worker.
  // Start at numPCs, work down to 2, removing (straight up, it is now an n-1 dimensional system)
  // the least causal component at each stage.
  for(int numComps = params.numPCs;numComps>1;numComps--)
    {
      
      Rdecor.elements.clear();
      bestAngle.resize(numComps-1);
      // Find the angle that results smallest upward causality.
      runFEHDstep(bestAngle, Rdecor, dataArray, params, numComps);
      // Local Q, going to assemble from the angles.
      Q.resize(numComps*numComps);
      
      singleQ(Q,bestAngle);

      std::fill(T.begin(),T.end(), 0.0f);

      for(int i=numComps;i<params.numPCs;i++)
	T[i*params.numPCs+i] = 1.0f;

      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  numComps,numComps,numComps,
		  alpha,Q.data(),numComps,
		  Rdecor.elements.data(),numComps,
		  beta,T.data(),params.numPCs);

      oneArrayData.resize(numComps*params.numEpochs*params.epochPts);
      convertDataListToRawArray(dataArray, oneArrayData.data());

      // Transform the data
      
      transformedData.resize(numComps*params.numEpochs*params.epochPts);

      cblas_sgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,
		  numComps,params.numEpochs*params.epochPts,numComps,
		  alpha,T.data(),params.numPCs,
		  oneArrayData.data(), numComps,
		  beta,transformedData.data(), numComps);

      // Convert it back into dataArray

      convertRawArrayToDataList(transformedData.data(), dataArray, numComps, params.epochPts, params.numEpochs);

      // Remove the last component
      removeComponent(dataArray, numComps-1);

      // Update the transformation

      cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		  params.numPCs,params.numChannels,params.numPCs,
		  alpha, T.data(),params.numPCs,
		  Lmat.data(),params.numPCs,
		  beta, newTrans.data(), params.numPCs);

      Lmat = newTrans;
    }
  
  return;
}



void singleQ(std::vector<float> &Q, std::vector<float> angle)
{

  int numVars = angle.size();
  
  float Qcol1;
  float Qcol2;

  float sinVal;
  float cosVal;

  // Create an identity matrix.
  for(int indx=0;indx<(numVars+1)*(numVars+1);indx++)
    {
      Q[indx] = 0;
    }
  for(int row=0;row<(numVars+1);row++)
    {
      Q[row*(numVars+1)+row] = 1.0f;
    }

  // Multiply the individual rotations together. 
  for(int varIndx=0; varIndx<numVars; varIndx++) // Cycle through the angles. 
    {
      sinVal = sinf(angle[varIndx]);// Assign the cos and sin to variables.
      cosVal = cosf(angle[varIndx]);

      for(int k=0;k<numVars+1;k++) // Do the matrix multiplication 
	{
	  Qcol1 = Q[k*(numVars+1)+varIndx]; //Q(particle,row i, column k
	  Qcol2 = Q[k*(numVars+1)+numVars];//(particle,row M-1,column k
	  Q[k*(numVars+1)+varIndx] = cosVal*Qcol1-sinVal*Qcol2;
	  Q[k*(numVars+1)+numVars] = sinVal*Qcol1+cosVal*Qcol2;
	}

    }
  
  return;
  
}
