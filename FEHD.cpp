#include <iostream>
#include "FEHD.h"
//#include "dataContainers.h"
#include "mkARGPU.h"
#include <algorithm>
#include <math.h>
#include <fstream>
#include "GC.h"
#include "utility.h"
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
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
      std::vector<float> q(numComps);
      
      // Find the angle that results smallest upward causality.
      
      runFEHDstep_v3(q, Rdecor, dataiter, params, numComps);

      // Orthogonal Q (column-major) whose last column is +-q, built directly as a
      // Householder reflection:
      //    v = q + s*e_last,  s = sign(q_last) (+1 if q_last = 0),
      //    Q = I - 2 v v^T / (v^T v).
      // Then Q*q = -s*e_last; Q is symmetric and Q*Q = I, so its last column (and row) is
      // -s*q, and the other columns are an orthonormal basis of the directions orthogonal
      // to q. v^T v = 2 + 2|q_last| >= 2, so there is no cancellation for any q.
      // (The previous QR of I - q q^T only works if q_last is not near 0; with q_last = 0
      // its last column comes out orthogonal to q, so the wrong component would be removed.)
      // The sign of the last row does not matter: that component is removed below.
      // Assumes q has unit length, which runFEHDstep_* guarantees.
      std::vector<float> Q(numComps*numComps);
      {
        const int last = numComps-1;
        const float s = (q[last] >= 0.0f) ? 1.0f : -1.0f;
        std::vector<float> v(q);
        v[last] += s;
        float vtv = 0.0f;
        for(int i=0;i<numComps;i++) vtv += v[i]*v[i];
        for(int col=0;col<numComps;col++)
          for(int row=0;row<numComps;row++)
            Q[col*numComps+row] = ((row == col) ? 1.0f : 0.0f) - 2.0f*v[row]*v[col]/vtv;
      }

      std::vector<float> T(numComps*numComps);
      
      cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		  numComps,numComps,numComps,
		  alpha,Q.data(),numComps,
		  Rdecor.data(),numComps,
		  beta,T.data(),numComps);
      
      // Transform the data      

      dataClass<float> Ldata = linearTrans(dataiter,T);
      std::vector<int> compsToRemove = {numComps-1};
      Ldata.removeComponents(compsToRemove);

      dataiter = Ldata;
      
      // Update the transformation

      std::vector<float> largeT(params.numPCs*params.numPCs,0.0);
      for(int indx=numComps;indx<params.numPCs;indx++)
	largeT[indx*params.numPCs+indx] = 1.0;
      for(int colindx=0;colindx<numComps;colindx++)
	for(int rowindx=0;rowindx<numComps;rowindx++)
	  largeT[colindx*params.numPCs+rowindx] = T[colindx*numComps+rowindx];
      
      std::vector<float> newTrans(Lmat);
      cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		  params.numPCs,params.numChannels,params.numPCs,
		  alpha, largeT.data(),params.numPCs,
		  newTrans.data(),params.numPCs,
		  beta, Lmat.data(), params.numPCs);
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
