#include "kernels_vruj.h"
#include <cuComplex.h>


__global__
void manyDots(cuComplex *qtGq,float *q,cuComplex *Gq,int numComps,int numFreqs,int numParticles)
{
  // Multiplies (dot product) q, as a row vector, times the previously computed Gq. 

  // The input q has size numComps x numParticles
  // The input Gq has size numComps x numFreqs x numParticles
  // The output qtGq has size numFreqs x numParticles
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numFreqs*numParticles) {return;}

  int idx_particle = int(idx/numFreqs);
  int idx_freq = idx - idx_particle*numFreqs;
  
  cuComplex sum = cuCmplx(0.0f,0.0f);

  for(int k=0;k<numComps;++k)
    {
      sum.x += q[idx_particle*numComps+k]*Gq[idx_particle*numComps*numFreqs+idx_freq*numComps+k].x;
      sum.y += q[idx_particle*numComps+k]*Gq[idx_particle*numComps*numFreqs+idx_freq*numComps+k].y;
    }
  qtGq[idx] = sum;  
}


__global__
void normalize_q(float *qout,float *q, int numComps,int numParticles)
{
  // Scales a bunch of q's to unit length

  // Input q has size numComps x numParticles
  // Output qout has size numComps x numParticles
  
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numParticles) {return;}

  float sum = 0.0f;
  for(int comp=0;comp<numComps;++comp)
    sum += q[idx*numComps+k]*q[idx*numComps+k];
  sum = rsqrtf(sum);
  for(int comp=0;comp<numComps;++comp)
    qout[idx*numComps+k] = q[idx*numComps+k]*sum;
}

__global__
void norm2Gq(float *Gq2,cuComplex *Gq,int numComps,int numFreqs, int numParticles)
{
  // Input Gq is (numFreqs x numComps) x numParticles
  // Output Gq2 is numFreqs x numParticles

  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx >= numFreqs*numParticles) {return;}

  idx_particle = int(idx/(numComps*numFreqs));
  idx_freq = idx/numComps - idx_particle*numFreqs;

  float nrm2 = 0.0f;
  for(int k=0;k<numComps;++k)
    {
      nrm2 += Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].x*
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].x +
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].y*
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].y;
    }
  nrm2 = sqrtf(nrm2);
  Gq2[idx] = nrm2;
}

__global__
void computeGC(float *GCvals,float *Gq2,float *qtGq,float *freq, int numComps,int numFreqs,int numParticles)
{
}

__global__
void G_vruj(cuComplex *G,float *AR,float *f,int *lagList,int numComps,int numFreqs,int numLags)
{
  // Computes the inverse of the transfer function tf(f)=I-sum(A_k exp(-2pi*i*dt*f*lag[k]))
  
  // Input AR is numComps x (numComps x numLags)
  // Input f is numFreqs
  // Input lagList is numLags
  // Output G is (numFreqs x numComps) x numComps
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx >= numFreqs*numComps*numComps) {return;}

  int idx_column = int(idx/(numComps*numFreqs));
  int idx_freq = int((idx/numComps-idx_column*(numFreqs)));
  int idx_row = idx-idx_column*(numFreqs*numComps)-idx_freq*numComps;

  float argmtBASE = -2.0f*3.1415f*dt*f[idx_freq];

  G[idx].x = 0.0f;
  G[idx].y = 0.0f;
   
  for(int lag=0;lag<numLags;++lag)
    {
      float argmt = argmtBase*((float)(lagList[lag]));
      G[idx].x -= AR[lag*numComps*numComps+idx_column*numComps+idx_row]*cosf(argmt);
      G[idx].y -= AR[lag*numComps*numComps+idx_column*numComps+idx_row]*sinf(argmt);
    }

  if(idx_column == idx_row)
    G[idx].x += 1.0f;    
}

  
