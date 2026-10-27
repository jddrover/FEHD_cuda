#include <stdio.h>
#include "kernels_vruj.h"
#include <cuComplex.h>

__global__
void GstarGq(cuComplex *GsGq,const cuComplex *G,const cuComplex *Gq,int numComps,int numFreqs,int numParticles)
{
  // Calculate G*Gq 

  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numParticles*numFreqs*numComps){return;}
  
  int idx_particle = int(idx/(numComps*numFreqs));
  int idx_freq = int(idx/numComps-idx_particle*numFreqs); 
  int idx_comp = idx-idx_particle*numComps*numFreqs-idx_freq*numComps;

  cuComplex cmplx_counter = make_cuComplex(0.0f,0.0f);
  for(int k=0;k<numComps;k++)
    { // Something wrong here...
      cmplx_counter.x += G[idx_comp*numFreqs*numComps+idx_freq*numComps+k].x * // a
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].x - // c
	-G[idx_comp*numFreqs*numComps+idx_freq*numComps+k].y * // -b (conj)
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].y; // d
      cmplx_counter.y += G[idx_comp*numFreqs*numComps+idx_freq*numComps+k].x *
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].y +
	-G[idx_comp*numFreqs*numComps+idx_freq*numComps+k].y *
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].x;
    }
  //printf("%f + %f i",cmplx_counter.x,cmplx_counter.y);
  GsGq[idx] = cmplx_counter;
}
	
__global__
void makeGq(cuComplex *Gq,const cuComplex *G,const float *q,int numComps,int numFreqs,int numParticles)
{
  // Calculate Gq
  
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx >= numParticles*numFreqs*numComps){return;}

  int idx_column = int(idx/(numComps*numFreqs));
  int idx_row = idx-idx_column*numComps*numFreqs;
  cuComplex summer = make_cuFloatComplex(0.0f,0.0f);
  
  for(int k=0;k<numComps;++k)
    {
      summer.x += G[k*numFreqs*numComps+idx_row].x*q[idx_column*numComps+k];
      summer.y += G[k*numFreqs*numComps+idx_row].y*q[idx_column*numComps+k];
    }
  Gq[idx] = summer;
}


__global__
void calcGradient(float *delF,const cuComplex *Gsq,const cuComplex *s,const float *Gq2,
		  const cuComplex *GstarGq,const float *q,
		  int numComps,int numFreqs,int numParticles)
{
  // Compute the gradient at q (not projected onto sphere surface).
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx >= numParticles*numFreqs*numComps){return;}

  int idx_particle = int(idx/(numComps*numFreqs));
  int idx_freq = int(idx/numComps-idx_particle*numFreqs);
  int idx_comp = idx - idx_freq*numComps-idx_particle*numComps*numFreqs;

  // First term
  float delF_element1 = 0.0f;

  delF_element1 = 2.0*GstarGq[idx_particle*numComps*numFreqs+idx_freq*numComps+idx_comp].x /
    Gq2[idx_particle*numFreqs+idx_freq];

  // Second term
  float delF_element2 = 0.0f;

  delF_element2 = 4.0*(Gsq[idx].x*s[idx_particle*numFreqs+idx_freq].x -
		       Gsq[idx].y*s[idx_particle*numFreqs+idx_freq].y) /
    (s[idx_particle*numFreqs+idx_freq].x*s[idx_particle*numFreqs+idx_freq].x +
     s[idx_particle*numFreqs+idx_freq].y*s[idx_particle*numFreqs+idx_freq].y);

  delF[idx] = delF_element1-delF_element2;
  
}

__global__
void GTplusG(cuComplex *GTpGd2,const cuComplex *G,int numComps,int numFreqs)
{
  // Compute (G^T+G)/2
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx >= numComps*numFreqs*numComps){return;}

  int idx_column = int(idx/(numFreqs*numComps));
  int idx_freq = int(idx/numComps-idx_column*numFreqs);
  int idx_row = idx - idx_column*numFreqs*numComps - idx_freq*numComps;

  cuComplex tmp = make_cuComplex(0.0f,0.0f);
  tmp.x = (G[idx_column*numComps*numFreqs+idx_freq*numComps+idx_row].x +
	   G[idx_row*numComps*numFreqs+idx_freq*numComps+idx_column].x)/2;
  tmp.y = (G[idx_column*numComps*numFreqs+idx_freq*numComps+idx_row].y +
	   G[idx_row*numComps*numFreqs+idx_freq*numComps+idx_column].y)/2;

  GTpGd2[idx] = tmp;
}


__global__
void qTGq(cuComplex *qtGq,const float *q,const cuComplex *Gq,int numComps,int numFreqs,int numParticles)
{
  // Multiplies (dot product) q, as a row vector, times the previously computed Gq. 

  // The input q has size numComps x numParticles
  // The input Gq has size numComps x numFreqs x numParticles
  // The output qtGq has size numFreqs x numParticles
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numFreqs*numParticles) {return;}

  int idx_particle = int(idx/numFreqs);
  int idx_freq = idx - idx_particle*numFreqs;
  
  cuComplex sum = make_cuComplex(0.0f,0.0f);

  for(int k=0;k<numComps;++k)
    {
      sum.x += q[idx_particle*numComps+k]*Gq[idx_particle*numComps*numFreqs+idx_freq*numComps+k].x;
      sum.y += q[idx_particle*numComps+k]*Gq[idx_particle*numComps*numFreqs+idx_freq*numComps+k].y;
    }
  qtGq[idx] = sum;
  
}


__global__
void magnitude2(float *absqtGq, const cuComplex *qtGq, int numFreqs,int numParticles)
{
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numFreqs*numParticles) {return;}

  float mag = 0.0f;
  mag = qtGq[idx].x*qtGq[idx].x + qtGq[idx].y*qtGq[idx].y;
  absqtGq[idx] = mag;
}
    

__global__
void normalize_q(float *qout,const float *q, int numComps,int numParticles)
{
  // Scales a bunch of q's to unit length

  // Input q has size numComps x numParticles
  // Output qout has size numComps x numParticles
  
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx >= numParticles) {return;}

  float sum = 0.0f;
  for(int k=0;k<numComps;++k)
    sum += q[idx*numComps+k]*q[idx*numComps+k];
  sum = rsqrtf(sum);
  for(int k=0;k<numComps;++k)
    qout[idx*numComps+k] = q[idx*numComps+k]*sum;
}

__global__
void norm2Gq(float *Gq2,cuComplex *Gq,int numComps,int numFreqs, int numParticles)
{
  // Input Gq is (numFreqs x numComps) x numParticles
  // Output Gq2 is numFreqs x numParticles

  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx >= numFreqs*numParticles) {return;}

  int idx_particle = int(idx/numFreqs);
  int idx_freq = idx - idx_particle*numFreqs;

  float nrm2 = 0.0f;
  for(int k=0;k<numComps;k++)
    {
      nrm2 += Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].x*
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].x +
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].y*
	Gq[idx_particle*numFreqs*numComps+idx_freq*numComps+k].y;
    }
  //nrm2 = sqrtf(nrm2);
  Gq2[idx] = nrm2;
}

__global__
void computeGC(float *GCvals,float *Gq2,float *absqtGq,int numFreqs,int numParticles)
{
  // Input Gq2 is numFreqs x numParticles
  // Input absqtGq is numFreqs x numParticles
  // Output GCvals is numParticles

  const int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx>=numParticles) {return;}
  // Trapezoids for integration.
  float GC = 0.0f;

  GC += 0.5*logf(Gq2[idx*numFreqs]/absqtGq[idx*numFreqs]);
  for(int findx=1;findx<numFreqs-1;++findx)
    GC += logf(Gq2[idx*numFreqs+findx]/absqtGq[idx*numFreqs+findx]);
  GC += 0.5*logf(Gq2[idx*numFreqs+numFreqs-1]/absqtGq[idx*numFreqs+numFreqs-1]);

  GCvals[idx] = GC;
}

__global__
void G_vruj(cuComplex *G,float *AR,int *lagList,int numComps,float fMin,float fMax,int numFreqs,int numLags,float dt)
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

  float freq = idx_freq*(fMax-fMin)/float(numFreqs-1)+fMin;
  
  float argmtBASE = -2.0f*3.1415f*dt*freq;

  // Switch this to a single value.
  cuComplex Gent = make_cuFloatComplex(0.0f,0.0f);
  for(int lag=0;lag<numLags;++lag)
    {
      float argmt = argmtBASE*((float)(lagList[lag]));
      Gent.x -= AR[lag*numComps*numComps+idx_column*numComps+idx_row]*cosf(argmt);
      Gent.y -= AR[lag*numComps*numComps+idx_column*numComps+idx_row]*sinf(argmt);
    }

  if(idx_column == idx_row)
    Gent.x += 1.0f;

  G[idx] = Gent;
}

__global__
void make_q_complex(cuComplex *qcmplx,float *q,int numComps,int numParticles)
{
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx >= numComps*numParticles) {return;}

  qcmplx[idx] = make_cuFloatComplex(q[idx],0.0f);
}

__global__
void projectQorth(float *qorth,float *q,float *delF,int numComps,int numFreqs,int numParticles)
{
  const int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=numComps*numFreqs*numParticles) {return;}

  int idx_particle = int(idx/(numComps*numFreqs));
  int idx_freq = int(idx/numComps-idx_particle*numFreqs);
  int idx_comp = idx - idx_particle*numComps*numFreqs - idx_freq*numComps;

  float summer = 0.0f;
  for(int k=0;k<numComps;++k)
    {
      float qo = -q[idx_particle*numComps+idx_comp]*q[idx_particle*numComps+k];
      if(k==idx_comp)
	qo += 1.0;
      summer += qo*delF[idx_particle*numComps*numFreqs+idx_freq*numComps+k];
    }

  qorth[idx] = summer;
}

__global__
void parseTable(float *q_out,float *GCtable,float *qStore,int numComps,int numParticles,int numCandidates)
{

  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numParticles){return;}

  int bestIndx = 0;
  for(int k=1;k<numCandidates;k++)
    {
      if(GCtable[idx*numCandidates+k] < GCtable[idx*numCandidates])
	{
	  GCtable[idx*numCandidates] = GCtable[idx*numCandidates+k];
	  bestIndx = k;
	}
    }

  for(int k=0;k<numComps;k++)
      q_out[idx*numComps+k] = qStore[bestIndx*numComps*numParticles+idx*numComps+k];

  return;  
}
