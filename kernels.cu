#include "kernels.h"
#include <stdio.h>
#include <cuComplex.h>

// Do the scaling for the S, including removing singular values that are too small.

__global__
void scaleByS(float *S,float *X,int m,int n)
{
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= m*n) {return;}
  
  int rowIndx = idx-int(idx/m)*m;
  
  X[idx] = X[idx]/S[rowIndx];    
}


// Transpose individual block matrices. 
__global__
void transposeBlockMatrices(float *A, float *B, int M, int P, int L)
{
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  // Trim any extra threads
  if(idx >= M*M*P*L) {return;}

  //printf("%d \n",idx);
  // Determine the local coordinates
  const int global_column = int(idx/(L*M));
  const int global_row = idx-global_column*L*M;
  const int particle = int(global_column/M);
  const int lag = int(global_row/M);
  const int local_col = global_column-particle*M;
  const int local_row = global_row-lag*M;
  const int locate_matrix = particle*L*M*M+lag*M;
 
  B[idx]=A[locate_matrix+local_row*L*M+local_col];
  return; 
}

// Angles as input, determine the rotation matrix for each particle.
__global__
void generateRotationMatrices(float *angleVars,float *Q_Array,int M,int numParticles){

  const int particle = threadIdx.x+blockDim.x*blockIdx.x;

  if(particle >= numParticles) {return;}
  
  float sinVal;
  float cosVal;

  float Qcol1;
  float Qcol2;
  
  // Zero out the Q matrices
  for(int i=0; i<M; i++)
    {
      for (int j=0;j<M;j++)
	{
	  Q_Array[particle*M*M+i*M+j] = 0.0f;
	}
    }
  

 // Put 1's down the diagonal.
  for(int i=0; i<M; i++)
    {
      Q_Array[particle*M*M+i*(M+1)]=1.0f;
    }
  
  for(int i=M-2; i>=0; i--) // Cycle through the angles. 
    {
      sinVal = sinf(angleVars[particle*(M-1)+i]);// Assign the cos and sin to variables.
      cosVal = cosf(angleVars[particle*(M-1)+i]);

      //printf("%f %f \n",sinVal,cosVal);
      for(int k=0;k<M;k++) // Do the matrix multiplication 
	{
	  
	  Qcol1 = Q_Array[particle*M*M+k*M+i]; //Q(particle,row i, column k
	  Qcol2 = Q_Array[particle*M*M+k*M+M-1];//(particle,row M-1,column k
	  Q_Array[particle*M*M+i+k*M] = cosVal*Qcol1+sinVal*Qcol2;
	  Q_Array[particle*M*M+(M-1)+k*M] = -sinVal*Qcol1+cosVal*Qcol2;
	  
	  /*
	  Qcol1 = Q_Array[particle*M*M+k*M+i]; //Q(particle,row i, column k
	  Qcol2 = Q_Array[particle*M*M+k*M+M-1];//(particle,row M-1,column k
	  Q_Array[particle*M*M+i+k*M] = cosVal*Qcol1-sinVal*Qcol2;
	  Q_Array[particle*M*M+(M-1)+k*M] = sinVal*Qcol1+cosVal*Qcol2;
	  */
	  
	}
    }
  return;
}
// Computes the inverse of the transfer function. 
__global__
void compTransferFunc(float *ARmodels,float2 *Tf,int *lagList,int numComps,int numParticles,float fMin,float fMax,int numFreqs,int numLags,float deltaT){

  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  
  if(idx >=numFreqs*numComps*numComps*numParticles){return;}

  
  extern __shared__ float2 sArray[];

  // Establish location in the array based on thread id (idx)
  
  const int global_column = (int)(idx/numComps);
  const int global_row = idx-global_column*numComps;
  const int particle = (int)(global_column/(numComps*numFreqs));
  const int freq_indx = (int)((global_column-particle*numComps*numFreqs)/numComps);  
  const int local_col = global_column-particle*numComps*numFreqs-freq_indx*numComps;
  const int local_row = global_row;
  const float frequency = (fMax-fMin)/((float)(numFreqs-1))*(float)freq_indx+fMin;

  // Assemble the exponent, saving the lag dependent part for the loop.
  const float argmtBASE = -2.0f*3.1415f*deltaT*frequency;
  float argmt;

  // Set up the shared memory
  const int sidx = threadIdx.x;
  
  sArray[sidx].x = 0.0f;
  sArray[sidx].y = 0.0f;

  // Loop over lags
  for(int lag = 0;lag<numLags;lag++)
    {
      // Finish the exponential
      argmt = argmtBASE*((float)(lagList[lag]));

      // Do the sum(A exp(arg))
      sArray[sidx].x -= ARmodels[particle*numComps*numComps*numLags+
				 lag*numComps+local_col*numComps*numLags+local_row]*cosf(argmt);
      sArray[sidx].y -= ARmodels[particle*numComps*numComps*numLags+
				 lag*numComps+local_col*numComps*numLags+local_row]*sinf(argmt);
    }

  // Add the identity matrix
  if(local_row == local_col)
    {
      sArray[sidx].x += 1.0f;
    }
  // Back to global memory to be returned
  Tf[idx] = sArray[sidx];

   
  return;
}
// Scale the last column of many matrices. 
__global__
void scale_columns(float2 *TF,int numComps, int numParticles, int numFreqs)
{
  // This function scales the last column in each matrix by the lowest diagonal entry.
  // This is just a gadget to make the upcoming blas calls more efficient.

  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx>=numComps*numFreqs*numParticles) {return;}

  const int column = (int)(idx/numComps);
  const int row = idx-column*numComps;
  if(row == numComps-1) return;
  
  const float2 divisor = TF[column*numComps*numComps+numComps*numComps-1];
  const int element = column*numComps*numComps+numComps*(numComps-1)+row;
  const float2 tmpVal=TF[element];
  
  TF[element].x = (tmpVal.x*divisor.x+tmpVal.y*divisor.y)/(divisor.x*divisor.x+divisor.y*divisor.y);
  TF[element].y = (tmpVal.y*divisor.x-tmpVal.x*divisor.y)/(divisor.x*divisor.x+divisor.y*divisor.y);

  return;
}
// Shrinks an array of matrices so that the matrices size decreases by one.
// Simply removes the last column and row of each matrix.
__global__
void shrinkArrays(float2 *S1, float2 *S2, int numComps, int numParticles, int numFreqs)
{
  // Removes the last column and row from each matrix.
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx>=(numComps-1)*(numComps-1)*numFreqs*numParticles) {return;}

  S2[idx].x=0.0f;
  S2[idx].y=0.0f;
  const int matrixNum = (int)(idx/((numComps-1)*(numComps-1)));
  const int col = (int)((idx-matrixNum*(numComps-1)*(numComps-1))/(numComps-1));
  const int row = idx - col*(numComps-1)-matrixNum*(numComps-1)*(numComps-1);
  
  S2[idx].x = S1[matrixNum*numComps*numComps+col*numComps+row].x;
  S2[idx].y = S1[matrixNum*numComps*numComps+col*numComps+row].y;

  return;
}
// Multiplies eigenvalues together to get determinant.
__global__
void prodEigs(float *W, float *det, int numComps, int numParticles, int numFreqs)
{
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  
  if(idx>=numFreqs*numParticles) {return;}

  extern __shared__ float s2[];

  const int sidx = threadIdx.x;
  
  
  s2[sidx] = 1.0;

  for(int comp=0;comp<numComps;comp++)
    {
      if(W[idx*numComps+comp]<=0.0){
	printf("particle %i, comp %i \n",idx,comp);
	printf("diagonal = %e \n",W[idx*numComps+comp]);
	}
      s2[sidx] = s2[sidx]*W[idx*numComps+comp];
    }
  det[idx]=s2[sidx];

  return;

}

__global__
void det2GC(float *d_partial, float *d_whole, float *GC, int numParticles, int numFreqs)
{
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx>=numParticles){return;}
  GC[idx] = 0.0f;

  float ratio;
  for(int freq=0;freq<numFreqs;freq++)
    {
      ratio = d_partial[idx*numFreqs+freq]/d_whole[idx*numFreqs+freq];
      
      if(ratio > 1.0f)
	GC[idx]=GC[idx]+logf(ratio);
      
      else if(ratio < 0.9f)
      	{	  
      	  printf("determinants are real close to zero, and the ratio is off. \n");
      	  printf("idx: %i, upper determinant: %e, lower determinant: %e \n",idx,d_partial[idx*numFreqs+freq], d_whole[idx*numFreqs+freq]);
	}
    }

  return;
}
