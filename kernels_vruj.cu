#include "kernels_vruj.h"
#include <cuComplex.h>

__global__
void manyDots(float *qtGq,float *q,float *Gq,int numComps,int numFreqs,int numParticles)
{
  // The input q has size numComps x numParticles
  // The input Gq has size numComps x numFreqs x numParticles
  // The output qtGq has size numFreqs x numParticles
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numFreqs*numParticles) {return;}

  int idx_particle = int(idx/numFreqs);
  int idx_freq = idx - idx_particle*numFreqs;
  
  float sum = 0.0f;
  for(int k=0;k<numComps;++k)
    sum += q[idx_particle*numComps+k]*Gq[idx_particle*numComps*numFreqs+idx_freq*numComps+k];
  qtGq[idx] = sum;  
}

__global__
void normalize_q(float *qout,float *q, int numComps,int numParticles)
{
  // The input q has size numComps x numParticles
  // The output qout has size numComps x numParticles
  
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numComps*numParticles) {return;}

  int idx_particle = int(idx/numComps);
  int idx_comp = idx - idx_particle*numComps;

  
