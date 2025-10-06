#ifndef KERNEL_H
#define KERNEL_H

#include "utility.h"

__global__ void reformat(float2 *,float2 *,paramContainer,int);
__global__ void generateRotationMatrices(float *,float2 *,int, int);
// Transpose individual block matrices
// [A1 A2 A3]  [A1t A2t A3t]
// [B1 B2 B3]->[B1t B2t ...]
// [C1 C2 C3]  [... ... ...]
// Doesn't overwrite the input. Not memory efficient, but micro second fast.
__global__ void transposeBlockMatrices(float2 *, float2 *, int, int, int);
// Compute the (inverse of the) transfer function for multiple AR models.
// Takes the rotated models and forms the (inverse of the) transfer function. 
__global__ void compTransferFunc(float *,float2 *,int *,int,int,float,float,int,int,float);
// Scales the last column in each of an array of matrices.
// This is a utility for using gemm to "invert" things.
__global__ void scale_columns(float2 *, int, int, int);
// Squeezes an matrix to one size down.
__global__ void shrinkArrays(float2 *, float2 *, int,int,int);
// Multiply the eigenvalues together to get the determinant.
__global__ void prodEigs(float *, float *, int,int,int);
// Compute sum(log(det(S)/det(Sp))
__global__ void det2GC(float *,float *,float *,int, int);
#endif
