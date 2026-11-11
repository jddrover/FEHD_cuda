#include <stdio.h>
#include "kernels_vruj.h"
#include <cuComplex.h>
#include <math.h>            // INFINITY
#include <curand_kernel.h>   // cuRAND device API (header only, no extra library to link)

// Tier 1 changes throughout:
//  - float literals (2.0f, 0.5f, ...) so no arithmetic is silently promoted to double
//  - __restrict__ on pointers that never overlap, so the compiler may keep values in
//    registers and use the read-only cache
//  - each array element is loaded once into a local variable instead of re-indexed
//  - const on every input that is only read

__global__
void GstarGq(cuComplex * __restrict__ GsGq, const cuComplex * __restrict__ G,
             const cuComplex * __restrict__ Gq, int numComps, int numFreqs, int numParticles)
{
  // Calculate G^H * Gq (conjugate transpose of G, times Gq), per frequency and particle.
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numParticles*numFreqs*numComps){return;}

  const int idx_particle = idx/(numComps*numFreqs);
  const int idx_freq     = idx/numComps - idx_particle*numFreqs;
  const int idx_comp     = idx - idx_particle*numComps*numFreqs - idx_freq*numComps;

  // G[comp][freq][k] is G(row=k, col=comp): column 'comp' of G at this frequency.
  const cuComplex *Gcol = G  + idx_comp*numFreqs*numComps     + idx_freq*numComps;
  const cuComplex *v    = Gq + idx_particle*numFreqs*numComps + idx_freq*numComps;

  cuComplex acc = make_cuComplex(0.0f, 0.0f);
  for(int k=0;k<numComps;k++)
    {
      const cuComplex g = Gcol[k];   // a + ib
      const cuComplex w = v[k];      // c + id
      acc.x += g.x*w.x + g.y*w.y;    // Re(conj(g)*w) = ac + bd
      acc.y += g.x*w.y - g.y*w.x;    // Im(conj(g)*w) = ad - bc
    }
  GsGq[idx] = acc;
}

__global__
void makeGq(cuComplex * __restrict__ Gq, const cuComplex * __restrict__ G,
            const float * __restrict__ q, int numComps, int numFreqs, int numParticles)
{
  // Calculate G*q for every frequency and particle.
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numParticles*numFreqs*numComps){return;}

  const int idx_column = idx/(numComps*numFreqs);         // particle
  const int idx_row    = idx - idx_column*numComps*numFreqs; // freq*numComps + comp

  const float *qp = q + idx_column*numComps;

  cuComplex summer = make_cuComplex(0.0f, 0.0f);
  for(int k=0;k<numComps;++k)
    {
      const cuComplex g  = G[k*numFreqs*numComps + idx_row];
      const float     qk = qp[k];
      summer.x += g.x*qk;
      summer.y += g.y*qk;
    }
  Gq[idx] = summer;
}

__global__
void calcGradient(float * __restrict__ delF, const cuComplex * __restrict__ Gsq,
                  const cuComplex * __restrict__ s, const float * __restrict__ Gq2,
                  const cuComplex * __restrict__ GstarGq, const float * __restrict__ q,
                  int numComps, int numFreqs, int numParticles)
{
  // Compute the gradient at q (not projected onto sphere surface).
  // (q is unused here; kept so the signature and call sites stay the same.)
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numParticles*numFreqs*numComps){return;}

  // idx = (particle*numFreqs + freq)*numComps + comp, so one division gives the
  // (particle, freq) index used by s and Gq2.
  const int idx_pf = idx/numComps;

  // First term: 2 Re(G^H Gq)_comp / ||Gq||^2.
  // GstarGq is laid out exactly like delF, so its element for this thread is GstarGq[idx].
  const float delF_element1 = 2.0f*GstarGq[idx].x / Gq2[idx_pf];

  // Second term: 4 Re(conj(s) Gsq) / |s|^2
  const cuComplex sv  = s[idx_pf];
  const cuComplex gsq = Gsq[idx];
  const float s2 = sv.x*sv.x + sv.y*sv.y;
  const float delF_element2 = 4.0f*(gsq.x*sv.x + gsq.y*sv.y) / s2;

  delF[idx] = delF_element1 - delF_element2;
}

__global__
void GTplusG(cuComplex * __restrict__ GTpGd2, const cuComplex * __restrict__ G,
             int numComps, int numFreqs)
{
  // Compute (G^T+G)/2
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numComps*numFreqs*numComps){return;}

  const int idx_column = idx/(numFreqs*numComps);
  const int idx_freq   = idx/numComps - idx_column*numFreqs;
  const int idx_row    = idx - idx_column*numFreqs*numComps - idx_freq*numComps;

  const cuComplex a = G[idx_column*numComps*numFreqs + idx_freq*numComps + idx_row];
  const cuComplex b = G[idx_row*numComps*numFreqs    + idx_freq*numComps + idx_column];

  GTpGd2[idx] = make_cuComplex(0.5f*(a.x + b.x), 0.5f*(a.y + b.y));
}

__global__
void qTGq(cuComplex * __restrict__ qtGq, const float * __restrict__ q,
          const cuComplex * __restrict__ Gq, int numComps, int numFreqs, int numParticles)
{
  // Multiplies (dot product) q, as a row vector, times the previously computed Gq.

  // The input q has size numComps x numParticles
  // The input Gq has size numComps x numFreqs x numParticles
  // The output qtGq has size numFreqs x numParticles
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numFreqs*numParticles) {return;}

  const int idx_particle = idx/numFreqs;
  const int idx_freq     = idx - idx_particle*numFreqs;

  const float     *qp = q  + idx_particle*numComps;
  const cuComplex *v  = Gq + idx_particle*numComps*numFreqs + idx_freq*numComps;

  cuComplex sum = make_cuComplex(0.0f, 0.0f);
  for(int k=0;k<numComps;++k)
    {
      const float     qk = qp[k];
      const cuComplex w  = v[k];
      sum.x += qk*w.x;
      sum.y += qk*w.y;
    }
  qtGq[idx] = sum;
}

__global__
void magnitude2(float * __restrict__ absqtGq, const cuComplex * __restrict__ qtGq,
                int numFreqs, int numParticles)
{
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numFreqs*numParticles) {return;}

  const cuComplex z = qtGq[idx];
  absqtGq[idx] = z.x*z.x + z.y*z.y;
}

__global__
void normalize_q(float *qout, const float *q, int numComps, int numParticles)
{
  // Scales a bunch of q's to unit length

  // Input q has size numComps x numParticles
  // Output qout has size numComps x numParticles

  // No __restrict__ here on purpose: this kernel works correctly in place
  // (qout == q), and __restrict__ would make that undefined behaviour.

  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numParticles) {return;}

  float sum = 0.0f;
  for(int k=0;k<numComps;++k)
    sum += q[idx*numComps+k]*q[idx*numComps+k];
  const float inv = rsqrtf(sum);
  for(int k=0;k<numComps;++k)
    qout[idx*numComps+k] = q[idx*numComps+k]*inv;
}

__global__
void norm2Gq(float * __restrict__ Gq2, const cuComplex * __restrict__ Gq,
             int numComps, int numFreqs, int numParticles)
{
  // Input Gq is (numFreqs x numComps) x numParticles
  // Output Gq2 is numFreqs x numParticles
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numFreqs*numParticles) {return;}

  // idx = particle*numFreqs + freq, so this vector starts at idx*numComps.
  const cuComplex *v = Gq + idx*numComps;

  float nrm2 = 0.0f;
  for(int k=0;k<numComps;k++)
    {
      const cuComplex w = v[k];
      nrm2 += w.x*w.x + w.y*w.y;
    }
  Gq2[idx] = nrm2;
}

__global__
void computeGC(float * __restrict__ GCvals, const float * __restrict__ Gq2,
               const float * __restrict__ absqtGq, int numFreqs, int numParticles)
{
  // Input Gq2 is numFreqs x numParticles
  // Input absqtGq is numFreqs x numParticles
  // Output GCvals is numParticles
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx>=numParticles) {return;}

  const float *a = Gq2     + idx*numFreqs;
  const float *b = absqtGq + idx*numFreqs;

  // Trapezoids for integration.
  float GC = 0.5f*logf(a[0]/b[0]);
  for(int findx=1;findx<numFreqs-1;++findx)
    GC += logf(a[findx]/b[findx]);
  GC += 0.5f*logf(a[numFreqs-1]/b[numFreqs-1]);

  GCvals[idx] = GC;
}

__global__
void G_vruj(cuComplex * __restrict__ G, const float * __restrict__ AR,
            const int * __restrict__ lagList, int numComps, float fMin, float fMax,
            int numFreqs, int numLags, float dt)
{
  // Computes the transfer function tf(f) = I - sum_k A_k exp(-2 pi i dt f lag[k])

  // Input AR is numComps x (numComps x numLags)
  // Input lagList is numLags
  // Output G is (numFreqs x numComps) x numComps
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numFreqs*numComps*numComps) {return;}

  const int idx_column = idx/(numComps*numFreqs);
  const int idx_freq   = idx/numComps - idx_column*numFreqs;
  const int idx_row    = idx - idx_column*(numFreqs*numComps) - idx_freq*numComps;

  const float freq = idx_freq*(fMax-fMin)/float(numFreqs-1) + fMin;

  // Phase in units of pi: exp(-2 pi i dt f lag) = exp(i pi x) with x = -2 dt f lag.
  // sincospif(x) returns sin(pi x) and cos(pi x) together and is more accurate than
  // multiplying by an approximate pi (the old code used 3.1415f).
  const float xBASE = -2.0f*dt*freq;

  cuComplex Gent = make_cuComplex(0.0f, 0.0f);
  for(int lag=0;lag<numLags;++lag)
    {
      const float a = AR[lag*numComps*numComps + idx_column*numComps + idx_row];
      float sn, cs;
      sincospif(xBASE*(float)lagList[lag], &sn, &cs);
      Gent.x -= a*cs;
      Gent.y -= a*sn;
    }

  if(idx_column == idx_row)
    Gent.x += 1.0f;

  G[idx] = Gent;
}

__global__
void make_q_complex(cuComplex * __restrict__ qcmplx, const float * __restrict__ q,
                    int numComps, int numParticles)
{
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numComps*numParticles) {return;}

  qcmplx[idx] = make_cuComplex(q[idx], 0.0f);
}

__global__
void projectQorth(float * __restrict__ qorth, const float * __restrict__ q,
                  const float * __restrict__ delF, int numComps, int numParticles)
{
  // Project delF onto the tangent space of the unit sphere at q:
  //   (I - q q^T) delF = delF - q (q . delF)
  // Same result as building row 'comp' of (I - q q^T), with half the arithmetic.
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx>=numComps*numParticles) {return;}

  const int idx_particle = idx/numComps;

  const float *qp = q    + idx_particle*numComps;
  const float *dp = delF + idx_particle*numComps;

  float dot = 0.0f;
  for(int k=0;k<numComps;++k)
    dot += qp[k]*dp[k];

  // idx == idx_particle*numComps + idx_comp, so q[idx] and delF[idx] are this component.
  qorth[idx] = delF[idx] - q[idx]*dot;
}

__global__
void parseTable(float * __restrict__ q_out, float * __restrict__ GCtable,
                const float * __restrict__ qStore, int numComps, int numParticles,
                int numCandidates)
{
  // For each particle, pick the candidate with the smallest GC value and copy its q.
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numParticles){return;}

  float *row = GCtable + idx*numCandidates;

  // Track the running minimum in a register instead of rewriting global memory.
  // Same comparison (strict <), so ties and NaNs resolve exactly as before.
  float best = row[0];
  int bestIndx = 0;
  for(int k=1;k<numCandidates;k++)
    {
      const float v = row[k];
      if(v < best)
        {
          best = v;
          bestIndx = k;
        }
    }

  // The old code left the minimum in GCtable[idx*numCandidates]; keep doing that
  // (a single write) in case the host reads it.
  row[0] = best;

  for(int k=0;k<numComps;k++)
    q_out[idx*numComps+k] = qStore[bestIndx*numComps*numParticles + idx*numComps + k];
}

__global__
void delgF(float * __restrict__ delF_summed, const float * __restrict__ delF_dev,
           int numComps, int numFreqs, int numParticles)
{
  // Trapezoidal sum of the per-frequency gradient over frequency.
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numComps*numParticles){return;}

  const int idx_particle = idx/numComps;
  const int idx_comp     = idx - idx_particle*numComps;

  const float *d = delF_dev + idx_particle*numFreqs*numComps + idx_comp;

  float summer = 0.5f*d[0];
  for(int k=1;k<numFreqs-1;k++)
    summer += d[k*numComps];
  summer += 0.5f*d[(numFreqs-1)*numComps];

  delF_summed[idx] = summer;
}

// ===========================================================================================
//  Version 2 kernels: work with the stacked real matrix K = [M_f; Sr_f; Si_f] (all f).
//
//  For real q:
//    ||G q||^2      = q^T M q          with M = Re(G^H G)       (real symmetric)
//    q^T G q        = q^T S q          with S = (G^T + G)/2 = Sr + i Si
//    Re(G^H G q)    = M q
//    (G^T+G)/2 q    = Sr q + i Si q
//
//  Layout of K (column-major, as cuBLAS expects):
//    (3*numComps*numFreqs) rows x numComps columns, leading dimension ldK = 3*numComps*numFreqs.
//    Rows for frequency f start at f*3*numComps:
//       rows [0, nC)      -> M_f
//       rows [nC, 2nC)    -> Sr_f
//       rows [2nC, 3nC)   -> Si_f
//
//  Y = K * Q (one cublasSgemm) then holds, for each (particle p, frequency f), the contiguous
//  3*numComps vector [M_f q ; Sr_f q ; Si_f q] starting at Y[(p*numFreqs + f)*3*numComps].
// ===========================================================================================

__global__
void buildSblocks(float * __restrict__ K, const cuComplex * __restrict__ G,
                  int numComps, int numFreqs)
{
  // Fill the Sr and Si blocks of K. (The M blocks are filled by cublasSgemmStridedBatched.)
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numComps*numFreqs*numComps){return;}

  // Same index order as G: idx = col*(numFreqs*numComps) + f*numComps + row
  const int col = idx/(numFreqs*numComps);
  const int f   = idx/numComps - col*numFreqs;
  const int row = idx - col*numFreqs*numComps - f*numComps;

  const cuComplex g1 = G[idx];                                    // G(row, col, f)
  const cuComplex g2 = G[row*numFreqs*numComps + f*numComps + col]; // G(col, row, f)

  const size_t ldK = (size_t)3*numComps*numFreqs;
  float *Kcol = K + col*ldK + (size_t)f*3*numComps;  // start of frequency f's rows in column col

  Kcol[numComps   + row] = 0.5f*(g1.x + g2.x);   // Sr_f(row, col)
  Kcol[2*numComps + row] = 0.5f*(g1.y + g2.y);   // Si_f(row, col)
}

__global__
void blockDots(float * __restrict__ out, const float * __restrict__ x,
               const float * __restrict__ Y, int numComps, int numFreqs, int numParticles)
{
  // For each (particle p, frequency f), dot x_p with each of the three blocks of Y:
  //   out[0*FP + pf] = x_p . Y_pf[0:nC]      (e.g. q^T M q)
  //   out[1*FP + pf] = x_p . Y_pf[nC:2nC]    (e.g. q^T Sr q)
  //   out[2*FP + pf] = x_p . Y_pf[2nC:3nC]   (e.g. q^T Si q)
  // where pf = p*numFreqs + f and FP = numFreqs*numParticles.
  //
  // One WARP (32 threads) per (p, f): lane L handles components L, L+32, L+64, ...
  // so neighbouring lanes read neighbouring addresses (coalesced). The 32 partial sums are
  // then combined with __shfl_down_sync, which passes values between the registers of
  // threads in the same warp without using shared memory.
  //
  // Launch with blockDim.x a multiple of 32 and at least numFreqs*numParticles warps in total.
  const int pf   = (threadIdx.x + blockDim.x*blockIdx.x) / 32;  // same for all lanes of a warp
  const int lane = threadIdx.x & 31;                            // 0..31
  const int FP   = numFreqs*numParticles;
  if(pf >= FP){return;}   // whole warp leaves together, so the shuffles below see all 32 lanes

  const int p = pf / numFreqs;
  const float *xp = x + (size_t)p*numComps;
  const float *y  = Y + (size_t)pf*3*numComps;

  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
  for(int c = lane; c < numComps; c += 32)
    {
      const float xc = xp[c];
      s0 += xc*y[c];
      s1 += xc*y[numComps + c];
      s2 += xc*y[2*numComps + c];
    }

  // Tree reduction across the warp: after the loop, lane 0 holds the full sums.
  for(int offset = 16; offset > 0; offset >>= 1)
    {
      s0 += __shfl_down_sync(0xffffffff, s0, offset);
      s1 += __shfl_down_sync(0xffffffff, s1, offset);
      s2 += __shfl_down_sync(0xffffffff, s2, offset);
    }

  if(lane == 0)
    {
      out[pf]        = s0;
      out[FP + pf]   = s1;
      out[2*FP + pf] = s2;
    }
}

__device__ __forceinline__
float trapWeight(int f, int numFreqs)
{
  // Trapezoid weights, matching computeGC and delgF (0.5 at both ends, 1 inside).
  return (numFreqs > 1 && (f == 0 || f == numFreqs-1)) ? 0.5f : 1.0f;
}

__global__
void gradFromBlocks(float * __restrict__ grad, const float * __restrict__ Y,
                    const float * __restrict__ qq, int numComps, int numFreqs, int numParticles)
{
  // Unprojected gradient, already summed over frequency (replaces calcGradient + delgF):
  //   grad_c = sum_f w_f [ 2 (Mq)_c / (q^T M q)  -  4 (sr (Sr q)_c + si (Si q)_c) / (sr^2 + si^2) ]
  // with sr = q^T Sr q, si = q^T Si q (so s = q^T G q = sr + i si).
  // One thread per (particle, component); neighbouring threads read neighbouring components.
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numComps*numParticles){return;}

  const int p = idx/numComps;
  const int c = idx - p*numComps;
  const int FP = numFreqs*numParticles;

  float sum = 0.0f;
  for(int f = 0; f < numFreqs; ++f)
    {
      const int pf = p*numFreqs + f;
      const float qMq = qq[pf];
      const float sr  = qq[FP + pf];
      const float si  = qq[2*FP + pf];
      const float s2  = sr*sr + si*si;

      const float *y = Y + (size_t)pf*3*numComps;
      const float g = 2.0f*y[c]/qMq - 4.0f*(sr*y[numComps + c] + si*y[2*numComps + c])/s2;
      sum += trapWeight(f, numFreqs)*g;
    }
  grad[idx] = sum;
}

__device__
float gcAtStep(float t, int p, int numFreqs, int FP,
               const float *qq, const float *qd, const float *dd,
               float qn, float qdot, float dn)
{
  // GC at q_hat = v/||v||, v = q - t d, using only precomputed scalars:
  //   v^T M v  = qMq - 2t qMd + t^2 dMd
  //   v^T S v  = qSq - 2t qSd + t^2 dSd      (real and imaginary parts separately)
  //   ||v||^2  = q.q - 2t q.d + t^2 d.d
  //   GC = sum_f w_f log( v^T M v * ||v||^2 / |v^T S v|^2 )
  const float vv = qn - 2.0f*t*qdot + t*t*dn;
  float GC = 0.0f;
  for(int f = 0; f < numFreqs; ++f)
    {
      const int pf = p*numFreqs + f;
      const float vMv = qq[pf]        - 2.0f*t*qd[pf]        + t*t*dd[pf];
      const float vSr = qq[FP + pf]   - 2.0f*t*qd[FP + pf]   + t*t*dd[FP + pf];
      const float vSi = qq[2*FP + pf] - 2.0f*t*qd[2*FP + pf] + t*t*dd[2*FP + pf];
      GC += trapWeight(f, numFreqs)*logf(vMv*vv/(vSr*vSr + vSi*vSi));
    }
  return GC;
}

__global__
void lineSearchUpdate(float *q, float * __restrict__ GCvals, const float * __restrict__ d,
                      const float * __restrict__ qq, const float * __restrict__ qd,
                      const float * __restrict__ dd, const float * __restrict__ steps,
                      int numSteps, int numComps, int numFreqs, int numParticles)
{
  // For each particle: evaluate GC at q_hat = (q - t d)/||q - t d|| for every step size t in
  // 'steps', keep the smallest (first one wins ties, as in parseTable), then replace q by the
  // winning q_hat and store its GC value. Replaces granger_vruj x numSteps + parseTable.
  //
  // q is updated in place, so it is deliberately not __restrict__.
  // qq, qd, dd each hold three planes of numFreqs*numParticles values (see blockDots):
  //   qq: q^T M q, q^T Sr q, q^T Si q      qd: q^T M d, ...      dd: d^T M d, ...
  const int p = threadIdx.x + blockDim.x*blockIdx.x;
  if(p >= numParticles){return;}

  const int FP = numFreqs*numParticles;
  float *qp = q + (size_t)p*numComps;
  const float *dp = d + (size_t)p*numComps;

  float qn = 0.0f, qdot = 0.0f, dn = 0.0f;
  for(int c = 0; c < numComps; ++c)
    {
      const float qc = qp[c], dc = dp[c];
      qn   += qc*qc;
      qdot += qc*dc;
      dn   += dc*dc;
    }

  int   bestK  = 0;
  float bestGC = gcAtStep(steps[0], p, numFreqs, FP, qq, qd, dd, qn, qdot, dn);
  for(int k = 1; k < numSteps; ++k)
    {
      const float GC = gcAtStep(steps[k], p, numFreqs, FP, qq, qd, dd, qn, qdot, dn);
      if(GC < bestGC)
        {
          bestGC = GC;
          bestK  = k;
        }
    }

  // Take the step and renormalise (same as Saxpy + normalize_q in the original loop).
  const float t   = steps[bestK];
  const float inv = rsqrtf(qn - 2.0f*t*qdot + t*t*dn);
  for(int c = 0; c < numComps; ++c)
    qp[c] = (qp[c] - t*dp[c])*inv;

  GCvals[p] = bestGC;
}

// ===========================================================================================
//  Version 3: fused kernels for numComps <= 32, many particles.
//
//  In this regime memory traffic, not arithmetic, is the limit, so nothing is stored per
//  (particle, frequency, component). Each thread handles one particle and loops over
//  frequencies. Each block loads K_f = [M_f; Sr_f; Si_f] into shared memory once per
//  frequency, and all its threads (particles) reuse it.
//
//  Template parameter P = numComps rounded up to 8, 16, 24 or 32. Arrays of size P live in
//  registers only because P is known at compile time; with a run-time size the compiler
//  would put them in slow "local" memory. Unused entries (c >= numComps) are zero, so the
//  padded maths gives exactly the unpadded result.
//
//  Padded K layout, per frequency a contiguous 3P x P column-major block:
//     Kpad[f*3P*P + k*3P + b*P + c] = block b (0 = M, 1 = Sr, 2 = Si), row c, column k
//  with zeros wherever c >= numComps or k >= numComps.
// ===========================================================================================

// V3_MAXSTEPS (16) is defined in kernels_vruj.h.
#define V3_BLOCK 128

// Step sizes for the line search. Constant memory is ideal here: every thread reads the
// same value at the same time, and the hardware broadcasts it.
__constant__ float c_steps[V3_MAXSTEPS];

int v3_paddedComps(int numComps)
{
  if(numComps <= 8)  return 8;
  if(numComps <= 16) return 16;
  if(numComps <= 24) return 24;
  if(numComps <= 32) return 32;
  return -1;
}

void v3_setSteps(const float *steps_host, int numSteps)
{
  cudaMemcpyToSymbol(c_steps, steps_host, sizeof(float)*numSteps);
}

__global__
void buildKpad(float * __restrict__ Kpad, const cuComplex * __restrict__ G,
               int numComps, int numFreqs, int P)
{
  // One thread per entry of Kpad (numFreqs * 3P * P entries). Runs once per model.
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx >= numFreqs*3*P*P){return;}

  const int f   = idx/(3*P*P);
  const int rem = idx - f*3*P*P;
  const int k   = rem/(3*P);       // column
  const int r   = rem - k*3*P;     // row within the 3P rows
  const int b   = r/P;             // 0 = M, 1 = Sr, 2 = Si
  const int c   = r - b*P;         // row within the block

  // G(row, col, f) = G[col*numFreqs*numComps + f*numComps + row]
  float val = 0.0f;
  if(c < numComps && k < numComps)
    {
      if(b == 0)
        {
          // M(c,k) = Re( sum_j conj(G(j,c)) G(j,k) )
          const cuComplex *Gc = G + c*numFreqs*numComps + f*numComps;
          const cuComplex *Gk = G + k*numFreqs*numComps + f*numComps;
          for(int j = 0; j < numComps; ++j)
            val += Gc[j].x*Gk[j].x + Gc[j].y*Gk[j].y;
        }
      else
        {
          const cuComplex g1 = G[k*numFreqs*numComps + f*numComps + c];  // G(c,k)
          const cuComplex g2 = G[c*numFreqs*numComps + f*numComps + k];  // G(k,c)
          val = (b == 1) ? 0.5f*(g1.x + g2.x) : 0.5f*(g1.y + g2.y);
        }
    }
  Kpad[idx] = val;
}

template<int P>
__device__ __forceinline__
void v3_stageK(float *Ks, const float *Kpad, int f)
{
  // All threads of the block copy K_f into shared memory together. Neighbouring threads
  // copy neighbouring floats, so the global reads are coalesced.
  const float *src = Kpad + (size_t)f*3*P*P;
  for(int i = threadIdx.x; i < 3*P*P; i += blockDim.x)
    Ks[i] = src[i];
}

template<int P>
__device__ __forceinline__
void v3_applyK(float (&y)[3*P], const float *Ks, const float (&x)[P])
{
  // y = K_f x, i.e. y[0:P) = M x, y[P:2P) = Sr x, y[2P:3P) = Si x.
  // Column k of K_f is contiguous, and every index here is a compile-time constant, so the
  // compiler can read 4 neighbouring rows with a single 128-bit shared-memory load. Every
  // thread in the warp reads the same address, so each load is a broadcast.
#pragma unroll
  for(int r = 0; r < 3*P; ++r) y[r] = 0.0f;
#pragma unroll
  for(int k = 0; k < P; ++k)
    {
      const float xk = x[k];
#pragma unroll
      for(int r = 0; r < 3*P; ++r)
        y[r] += Ks[k*3*P + r]*xk;
    }
}

template<int P>
__global__ void __launch_bounds__(V3_BLOCK)
v3_gradient(float * __restrict__ d, float * __restrict__ qq,
            const float * __restrict__ Kpad, const float * __restrict__ q,
            int numComps, int numFreqs, int numParticles)
{
  // Per particle: the projected gradient d (numComps floats), and for every frequency the
  // scalars a = q^T M q, sr = q^T Sr q, si = q^T Si q, stored for v3_lineSearch as
  //    qq[(b*numFreqs + f)*numParticles + p],  b = 0, 1, 2
  // so neighbouring threads (particles) write neighbouring addresses.
  // Replaces version 1's makeGq x2, norm2Gq, GstarGq, qTGq, calcGradient, delgF, projectQorth.
  __shared__ __align__(16) float Ks[3*P*P];

  const int p = threadIdx.x + blockDim.x*blockIdx.x;
  const bool active = (p < numParticles);
  // No early return: every thread must reach the __syncthreads() calls below, so threads
  // past the last particle still help load K_f; they just never write results.

  float qr[P], g[P];
#pragma unroll
  for(int c = 0; c < P; ++c)
    {
      qr[c] = (active && c < numComps) ? q[(size_t)p*numComps + c] : 0.0f;
      g[c]  = 0.0f;
    }

  for(int f = 0; f < numFreqs; ++f)
    {
      __syncthreads();            // everyone has finished with the previous K_f
      v3_stageK<P>(Ks, Kpad, f);
      __syncthreads();            // K_f is fully loaded before anyone reads it

      float y[3*P];
      v3_applyK<P>(y, Ks, qr);

      float a = 0.0f, sr = 0.0f, si = 0.0f;
#pragma unroll
      for(int c = 0; c < P; ++c)
        {
          a  += qr[c]*y[c];
          sr += qr[c]*y[P + c];
          si += qr[c]*y[2*P + c];
        }

      // grad += w_f [ 2 Mq / a  -  4 (sr Srq + si Siq) / (sr^2 + si^2) ]
      const float w  = trapWeight(f, numFreqs);
      const float cM = 2.0f*w/a;
      const float cS = 4.0f*w/(sr*sr + si*si);
#pragma unroll
      for(int c = 0; c < P; ++c)
        g[c] += cM*y[c] - cS*(sr*y[P + c] + si*y[2*P + c]);

      if(active)
        {
          qq[((size_t)0*numFreqs + f)*numParticles + p] = a;
          qq[((size_t)1*numFreqs + f)*numParticles + p] = sr;
          qq[((size_t)2*numFreqs + f)*numParticles + p] = si;
        }
    }

  if(!active){return;}

  // Project onto the tangent space of the sphere: d = g - q (q . g)
  float dot = 0.0f;
#pragma unroll
  for(int c = 0; c < P; ++c) dot += qr[c]*g[c];
#pragma unroll
  for(int c = 0; c < P; ++c)
    if(c < numComps) d[(size_t)p*numComps + c] = g[c] - qr[c]*dot;
}

// ---- Random unit vectors on the GPU (for starting points and recycling) ----
//
// cuRAND's Philox generator is "counter based": given (seed, subsequence) it produces its
// numbers directly, with no state carried between kernel launches and almost no set-up
// cost. Each (iteration, particle) pair gets its own subsequence, so every draw is
// independent and the run is reproducible for a given seed.
//
// Normalising a vector of independent standard normal numbers gives a point uniformly
// distributed on the unit sphere. (Normalising a vector of uniform numbers in [-1,1], as
// the host code does for the starting q's, over-weights directions towards the cube's
// corners.)
__device__ __forceinline__
unsigned long long v3_subsequence(int iter, int p, int numParticles)
{
  // iter = -1 for the starting vectors, 0, 1, 2, ... for recycling in each iteration
  return (unsigned long long)(iter + 1)*(unsigned long long)numParticles + (unsigned long long)p;
}

__global__
void v3_randomUnit(float * __restrict__ q, int numComps, int numParticles,
                   unsigned long long seed)
{
  // Starting q's: one random unit vector per particle.
  const int p = threadIdx.x + blockDim.x*blockIdx.x;
  if(p >= numParticles){return;}

  curandStatePhilox4_32_10_t st;
  curand_init(seed, v3_subsequence(-1, p, numParticles), 0, &st);

  float *qp = q + (size_t)p*numComps;
  float nrm2 = 0.0f;
  for(int c = 0; c < numComps; ++c)
    {
      const float x = curand_normal(&st);
      qp[c] = x;
      nrm2 += x*x;
    }
  const float inv = rsqrtf(nrm2);
  for(int c = 0; c < numComps; ++c) qp[c] *= inv;
}

__global__
void fillFloat(float * __restrict__ x, float value, int n)
{
  const int i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i < n) x[i] = value;
}

template<int P>
__global__ void __launch_bounds__(V3_BLOCK)
v3_lineSearch(float *q, float * __restrict__ GCvals, const float * __restrict__ d,
              const float * __restrict__ qq, const float * __restrict__ Kpad,
              float * __restrict__ bestGC, float * __restrict__ bestQ,
              int * __restrict__ recycleCount, float recycleTol,
              unsigned long long seed, int iter,
              int numSteps, int numComps, int numFreqs, int numParticles)
{
  // Per particle: GC at q_hat = (q - t d)/||q - t d|| for every step size t in c_steps,
  // accumulated over frequency in registers (one running sum per step size). Then keep the
  // smallest (first one wins ties, as in parseTable), update q in place, and store its GC.
  // Replaces version 1's numSteps x granger_vruj, Saxpy, normalize_q and parseTable.
  //
  // Then:
  //   bestGC[p], bestQ[p]: the best GC this particle has ever reached, and its q.
  //   Recycling: if this iteration improved GC by no more than recycleTol, the particle has
  //   stopped moving; its result is already in bestGC/bestQ, so q is replaced by a random
  //   unit vector. recycleTol < 0 turns recycling off. If recycleCount is not null, the
  //   number of recycled particles is added to it.
  //   GCvals[p] is the GC this iteration's line search found (before any recycling).
  //
  // q is updated in place (each thread only touches its own particle), so not __restrict__.
  __shared__ __align__(16) float Ks[3*P*P];

  const int p = threadIdx.x + blockDim.x*blockIdx.x;
  const bool active = (p < numParticles);

  float qr[P], dr[P];
  float qn = 0.0f, qdot = 0.0f, dn = 0.0f;
#pragma unroll
  for(int c = 0; c < P; ++c)
    {
      qr[c] = (active && c < numComps) ? q[(size_t)p*numComps + c] : 0.0f;
      dr[c] = (active && c < numComps) ? d[(size_t)p*numComps + c] : 0.0f;
      qn   += qr[c]*qr[c];
      qdot += qr[c]*dr[c];
      dn   += dr[c]*dr[c];
    }

  float GCacc[V3_MAXSTEPS];
#pragma unroll
  for(int k = 0; k < V3_MAXSTEPS; ++k) GCacc[k] = 0.0f;

  for(int f = 0; f < numFreqs; ++f)
    {
      __syncthreads();
      v3_stageK<P>(Ks, Kpad, f);
      __syncthreads();

      float z[3*P];
      v3_applyK<P>(z, Ks, dr);                 // z = [M d; Sr d; Si d]

      float qMd = 0.0f, qSr = 0.0f, qSi = 0.0f, dMd = 0.0f, dSr = 0.0f, dSi = 0.0f;
#pragma unroll
      for(int c = 0; c < P; ++c)
        {
          qMd += qr[c]*z[c];  qSr += qr[c]*z[P + c];  qSi += qr[c]*z[2*P + c];
          dMd += dr[c]*z[c];  dSr += dr[c]*z[P + c];  dSi += dr[c]*z[2*P + c];
        }

      // q^T M q, q^T Sr q, q^T Si q from v3_gradient (coalesced: neighbouring p, neighbouring address)
      const float a  = active ? qq[((size_t)0*numFreqs + f)*numParticles + p] : 1.0f;
      const float sr = active ? qq[((size_t)1*numFreqs + f)*numParticles + p] : 1.0f;
      const float si = active ? qq[((size_t)2*numFreqs + f)*numParticles + p] : 1.0f;
      const float w  = trapWeight(f, numFreqs);

#pragma unroll
      for(int k = 0; k < V3_MAXSTEPS; ++k)
        if(k < numSteps)
          {
            const float t   = c_steps[k];
            const float vv  = qn - 2.0f*t*qdot + t*t*dn;     // ||q - t d||^2
            const float vMv = a  - 2.0f*t*qMd  + t*t*dMd;    // v^T M v
            const float vSr = sr - 2.0f*t*qSr  + t*t*dSr;    // Re(v^T S v)
            const float vSi = si - 2.0f*t*qSi  + t*t*dSi;    // Im(v^T S v)
            GCacc[k] += w*logf(vMv*vv/(vSr*vSr + vSi*vSi));
          }
    }

  if(!active){return;}

  float best = GCacc[0];
  int bestK = 0;
#pragma unroll
  for(int k = 1; k < V3_MAXSTEPS; ++k)
    if(k < numSteps && GCacc[k] < best)
      {
        best  = GCacc[k];
        bestK = k;
      }

  const float t   = c_steps[bestK];
  const float inv = rsqrtf(qn - 2.0f*t*qdot + t*t*dn);
  float qnew[P];
#pragma unroll
  for(int c = 0; c < P; ++c) qnew[c] = (qr[c] - t*dr[c])*inv;

  GCvals[p] = best;

  // Record this particle's best result so far. bestGC starts at +infinity.
  if(best < bestGC[p])
    {
      bestGC[p] = best;
#pragma unroll
      for(int c = 0; c < P; ++c)
        if(c < numComps) bestQ[(size_t)p*numComps + c] = qnew[c];
    }

  // GCacc[0] is the GC at the current q (step size 0), so GCacc[0] - best is how much this
  // iteration improved things. With recycleTol = 0, a particle is recycled exactly when no
  // step size helped: it would then repeat the same computation forever.
  const bool recycle = (recycleTol >= 0.0f) && (GCacc[0] - best <= recycleTol);
  if(recycle)
    {
      curandStatePhilox4_32_10_t st;
      curand_init(seed, v3_subsequence(iter, p, numParticles), 0, &st);
      float nrm2 = 0.0f;
#pragma unroll
      for(int c = 0; c < P; ++c)
        {
          qnew[c] = (c < numComps) ? curand_normal(&st) : 0.0f;
          nrm2 += qnew[c]*qnew[c];
        }
      const float s = rsqrtf(nrm2);
#pragma unroll
      for(int c = 0; c < P; ++c) qnew[c] *= s;

      if(recycleCount) atomicAdd(recycleCount, 1);
    }

#pragma unroll
  for(int c = 0; c < P; ++c)
    if(c < numComps) q[(size_t)p*numComps + c] = qnew[c];
}

void v3_buildK(float *Kpad, const cuComplex *G, int numComps, int numFreqs)
{
  const int P = v3_paddedComps(numComps);
  const int total = numFreqs*3*P*P;
  buildKpad<<<(total + 255)/256, 256>>>(Kpad, G, numComps, numFreqs, P);
}

void v3_initParticles(float *q, float *bestGC, int numComps, int numParticles,
                      unsigned long long seed)
{
  const int grid = (numParticles + 255)/256;
  v3_randomUnit<<<grid, 256>>>(q, numComps, numParticles, seed);
  fillFloat<<<grid, 256>>>(bestGC, INFINITY, numParticles);
}

template<int P>
static void v3_iterateP(const v3_args &a)
{
  const int grid = (a.numParticles + V3_BLOCK - 1)/V3_BLOCK;
  v3_gradient<P><<<grid, V3_BLOCK>>>(a.d, a.qq, a.Kpad, a.q,
                                     a.numComps, a.numFreqs, a.numParticles);
  v3_lineSearch<P><<<grid, V3_BLOCK>>>(a.q, a.GCvals, a.d, a.qq, a.Kpad,
                                       a.bestGC, a.bestQ, a.recycleCount, a.recycleTol,
                                       a.seed, a.iter,
                                       a.numSteps, a.numComps, a.numFreqs, a.numParticles);
}

void v3_iterate(const v3_args &a)
{
  // Pick the compiled kernel size that fits numComps. Each case is a separate compiled
  // copy of the kernels with P fixed.
  switch(v3_paddedComps(a.numComps))
    {
    case 8:  v3_iterateP<8> (a); break;
    case 16: v3_iterateP<16>(a); break;
    case 24: v3_iterateP<24>(a); break;
    case 32: v3_iterateP<32>(a); break;
    default: printf("v3_iterate: numComps = %d is larger than 32\n", a.numComps);
    }
}
