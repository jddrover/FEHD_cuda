#ifndef KERNELS_VRUJ_H
#define KERNELS_VRUJ_H

#include <cuComplex.h>

// __restrict__ is a promise that the pointer does not overlap any other pointer argument
// of the same kernel. Never pass the same buffer twice to a kernel marked this way.
// normalize_q is left unrestricted because it may be called in place (qout == q).

__global__ void GstarGq(cuComplex * __restrict__, const cuComplex * __restrict__,
                        const cuComplex * __restrict__, int, int, int);
__global__ void makeGq(cuComplex * __restrict__, const cuComplex * __restrict__,
                       const float * __restrict__, int, int, int);
__global__ void calcGradient(float * __restrict__, const cuComplex * __restrict__,
                             const cuComplex * __restrict__, const float * __restrict__,
                             const cuComplex * __restrict__, const float * __restrict__,
                             int, int, int);
__global__ void GTplusG(cuComplex * __restrict__, const cuComplex * __restrict__, int, int);
__global__ void qTGq(cuComplex * __restrict__, const float * __restrict__,
                     const cuComplex * __restrict__, int, int, int);
__global__ void magnitude2(float * __restrict__, const cuComplex * __restrict__, int, int);
__global__ void normalize_q(float *, const float *, int, int);
__global__ void norm2Gq(float * __restrict__, const cuComplex * __restrict__, int, int, int);
__global__ void computeGC(float * __restrict__, const float * __restrict__,
                          const float * __restrict__, int, int);
__global__ void G_vruj(cuComplex * __restrict__, const float * __restrict__,
                       const int * __restrict__, int, float, float, int, int, float);
__global__ void make_q_complex(cuComplex * __restrict__, const float * __restrict__, int, int);
__global__ void projectQorth(float * __restrict__ qorth, const float * __restrict__ q,
                             const float * __restrict__ delF, int numComps, int numParticles);
__global__ void parseTable(float * __restrict__ q_out, float * __restrict__ GCtable,
                           const float * __restrict__ qStore, int numComps,
                           int numParticles, int numCandidates);
__global__ void delgF(float * __restrict__, const float * __restrict__, int, int, int);

// ---- Version 2 (stacked K = [M; Sr; Si]) ----
__global__ void buildSblocks(float * __restrict__ K, const cuComplex * __restrict__ G,
                             int numComps, int numFreqs);
__global__ void blockDots(float * __restrict__ out, const float * __restrict__ x,
                          const float * __restrict__ Y, int numComps, int numFreqs,
                          int numParticles);
__global__ void gradFromBlocks(float * __restrict__ grad, const float * __restrict__ Y,
                               const float * __restrict__ qq, int numComps, int numFreqs,
                               int numParticles);
__global__ void lineSearchUpdate(float *q, float * __restrict__ GCvals,
                                 const float * __restrict__ d, const float * __restrict__ qq,
                                 const float * __restrict__ qd, const float * __restrict__ dd,
                                 const float * __restrict__ steps, int numSteps,
                                 int numComps, int numFreqs, int numParticles);

// ---- Version 3 (fused, numComps <= 32) ----
// Host functions; the templated kernels they launch live in kernels_vruj.cu.
#define V3_MAXSTEPS 16
int  v3_paddedComps(int numComps);   // 8, 16, 24 or 32; -1 if numComps > 32
void v3_setSteps(const float *steps_host, int numSteps);   // numSteps <= V3_MAXSTEPS
void v3_buildK(float *Kpad, const cuComplex *G, int numComps, int numFreqs);

// Random unit vector per particle (uniform on the sphere) and bestGC = +infinity.
void v3_initParticles(float *q, float *bestGC, int numComps, int numParticles,
                      unsigned long long seed);

// Everything one iteration needs, gathered in a struct so the argument list stays readable.
struct v3_args
{
  float *q;              // numComps x numParticles, updated in place
  float *d;              // numComps x numParticles, work space (projected gradient)
  float *qq;             // 3 x numFreqs x numParticles, work space
  float *GCvals;         // numParticles: GC found by this iteration's line search
  const float *Kpad;     // from v3_buildK
  float *bestGC;         // numParticles: best GC each particle has reached
  float *bestQ;          // numComps x numParticles: the q that reached it
  int   *recycleCount;   // device int incremented once per recycled particle, or nullptr
  float recycleTol;      // recycle when an iteration improves GC by <= this; < 0 = never
  unsigned long long seed;
  int iter;              // iteration number, so every iteration draws different vectors
  int numSteps, numComps, numFreqs, numParticles;
};
void v3_iterate(const v3_args &a);
#endif
