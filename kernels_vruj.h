#ifndef KERNELS_VRUJ_H
#define KERNELS_VRUJ_H

#include <cuComplex.h>

__global__ void GstarGq(cuComplex *,const cuComplex *,const cuComplex *,int,int,int);
__global__ void makeGq(cuComplex *,const cuComplex *,const float *,int,int,int);
__global__ void calcGradient(float *,const cuComplex *,const cuComplex *,const float *,
			     const cuComplex *,const float *,
			     int,int,int);
__global__ void GTplusG(cuComplex *,const cuComplex *,int,int);
__global__ void qTGq(cuComplex *,const float *,const cuComplex *,int,int,int);
__global__ void magnitude2(float *,const cuComplex *,int,int);
__global__ void normalize_q(float *,const float *,int,int);
__global__ void norm2Gq(float *,cuComplex *,int,int,int);
__global__ void computeGC(float *,float *,float *,int,int);
__global__ void G_vruj(cuComplex *,float *,int *,int,float,float,int,int,float);
__global__ void make_q_complex(cuComplex *,float *,int,int);
__global__ void projectQorth(float *qorth,float *q,float *delF,int numComps,int numParticles);
__global__ void parseTable(float *q_out,float *GCtable,float *qStore, int numComps,
			   int numParticles,int numCandidates);
__global__ void delgF(float *,float*,int,int,int);
#endif
