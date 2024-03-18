#ifndef KERNEL_H
#define KERNEL_H


__global__ void generateRotationMatrices(float *,float *,int, int);
// Transpose individual block matrices
__global__ void transposeBlockMatrices(float *, float *, int, int, int);
// Compute the (inverse of the) transfer function for multiple AR models.
__global__ void compTransferFunc(float *,float2 *,int *,int,int,float,float,int,int,float);
__global__ void scale_columns(float2 *, int, int, int);
__global__ void shrinkArrays(float2 *, float2 *, int,int,int);
__global__ void prodEigs(float *, float *, int,int,int);
__global__ void det2GC(float *,float *,float *,int, int);
#endif
