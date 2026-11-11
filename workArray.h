#ifndef WORKARRAY_H
#define WORKARRAY_H
#include "utility.h"
#include <vector>
#include <cuComplex.h>

// Every pointer starts as nullptr, so freeWorkArray can free whichever buffers a version
// allocated (cudaFree(nullptr) does nothing).
// Sizes: C = numComps, F = numFreqs, P = numParticles, L = numLags.
struct workForGranger
{
  float *q_dev = nullptr; // CP*4
  float *q_tmp = nullptr; // CP*4
  cuComplex *G_dev = nullptr; // CCF*8
  cuComplex *s_dev = nullptr; // FP*8
  float *absqtGq_dev = nullptr; // FP*4
  cuComplex *Gq_dev = nullptr; // CFP*8
  float *Gq2_dev = nullptr; // FP*4
  float *GCvals_dev = nullptr; // P*4
  float *AR_dev = nullptr; // CCL*4
  float *gradient_dev = nullptr; // CP*4
  cuComplex *GsGq_dev = nullptr; // CFP*8
  cuComplex *Gs_dev = nullptr; // CF*8
  cuComplex *Gsq_dev = nullptr; // CFP*8
  float *delF_dev = nullptr; // CFP*4
  float *delF_proj_dev = nullptr; // CP*4
  cuComplex *GstarGq_dev = nullptr; // CFP*8  (allocated by allocateParams but never used)
  int *lagList_dev = nullptr; // L

  // ---- Version 2 only ----
  float *K_dev = nullptr;     // v2: 3CCF*4 stacked [M_f; Sr_f; Si_f]   v3: 3*Pad*Pad*F*4 (padded)
  float *Y_dev = nullptr;     // 3CFP*4  K*q   -> [Mq; Srq; Siq] per (particle, freq)
  float *Z_dev = nullptr;     // 3CFP*4  K*d   -> [Md; Srd; Sid] per (particle, freq)
  float *qq_dev = nullptr;    // 3FP*4   q.Mq, q.Srq, q.Siq  (v2 and v3)
  float *qd_dev = nullptr;    // 3FP*4   q.Md, q.Srd, q.Sid
  float *dd_dev = nullptr;    // 3FP*4   d.Md, d.Srd, d.Sid
  float *steps_dev = nullptr; // numSteps  line-search step sizes

  // ---- Version 3 recycling ----
  float *bestGC_dev = nullptr;    // P*4   best GC each particle has reached
  float *bestQ_dev = nullptr;     // CP*4  the q that reached it
  int *recycleCount_dev = nullptr; // 1 int, particles recycled (verbose output only)
};

//void allocateParams(workForGranger &,int numComps,int particleBlockSize,paramContainer params,std::vector<int> lagList,std::vector<float> AR);
void allocateParams(workForGranger &,int numComps,int particleBlockSize,paramContainer params);
void allocateParams_v2(workForGranger &,int numComps,int particleBlockSize,paramContainer params,
                       int numSteps);
void allocateParams_v3(workForGranger &,int numComps,int particleBlockSize,paramContainer params,
                       int paddedComps);
void freeWorkArray(workForGranger &);
#endif
