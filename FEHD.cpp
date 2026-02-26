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
#include "dataClass.h"
#include "dataManip.h"
//#include "timeSeriesOPs.h"
// Main call. Executes the FEHD algorithm
void runFEHD(dataClass<float> dataArray, std::vector<float> &Lmat, paramContainer params)
{
  // Set the parameters for sgemm. 
  float alpha=1.0f;
  float beta=0.0f;

  dataClass<float> dataiter = dataArray;

  // Start at numPCs, work down to 2, determining the least causal combination at each stage.
  for(int numComps = params.numPCs;numComps>1;numComps--)
    {
      std::vector<float> Rdecor(numComps*numComps);
      std::vector<float> bestAngle(numComps-1);
      
      // Find the angle that results smallest upward causality.
      
      runFEHDstep(bestAngle, Rdecor, dataiter, params, numComps);

      // Local Q, going to assemble from the angles.
      std::vector<float> Q(numComps*numComps);      
      singleQ(Q,bestAngle);
      /*
      for(int row=0;row<numComps;row++)
	{
	  for(int col=0;col<numComps;col++)
	    std::cout << Q[col*numComps+row] << " ";
	  std::cout << std::endl;
	}
      */
      
      std::vector<float> T(numComps*numComps);
      
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  numComps,numComps,numComps,
		  alpha,Q.data(),numComps,
		  Rdecor.data(),numComps,
		  beta,T.data(),numComps);

      /*
      for(int row=0;row<numComps;row++)
	{
	  for(int col=0;col<numComps;col++)
	    std::cout << T[col*numComps+row] << " ";
	  std::cout << std::endl;
	}
      */
      
      // Transform the data      

      dataClass<float> Ldata = linearTrans(dataiter,T);
      Ldata.removeComponent(numComps-1);
      //std::cout << Ldata.dataArray().size() << std::endl;
      //std::cout << Ldata.getNumComps() << std::endl;
      //std::cout << Ldata.getTotalPoints() << std::endl;
      dataiter = Ldata;
      
      // Update the transformation

      std::vector<float> largeT(params.numPCs*params.numPCs,0.0);
      for(int indx=numComps;indx<params.numPCs;indx++)
	largeT[indx*params.numPCs+indx] = 1.0;
      for(int colindx=0;colindx<numComps;colindx++)
	for(int rowindx=0;rowindx<numComps;rowindx++)
	  largeT[colindx*params.numPCs+rowindx] = T[colindx*numComps+rowindx];
      std::cout << Lmat.size() << std::endl;// This might be a mess.
      std::vector<float> newTrans(Lmat);
      cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		  params.numPCs,params.numChannels,params.numPCs,
		  alpha, largeT.data(),params.numPCs,
		  newTrans.data(),params.numPCs,
		  beta, Lmat.data(), params.numPCs);

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
