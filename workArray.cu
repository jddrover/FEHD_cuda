#include "workArray.h"
#include "utility.h"
#include <vector>
void allocateParams(workForGranger &workArray,int numComps,int particleBlockSize,paramContainer params,std::vector<int> lagList,std::vector<float> AR)
{
  workArray.lworkVal = particleBlockSize*params.numFreqs*2*(numComps-1)*(numComps-1);
  cudaMalloc((void**)&workArray.Qdev, sizeof(float)*particleBlockSize*numComps*numComps);
  cudaMalloc((void**)&workArray.rotatedModels, sizeof(float)*numComps*numComps*particleBlockSize*params.numLags);
  cudaMalloc((void**)&workArray.wArray, sizeof(float)*numComps*numComps*particleBlockSize*params.numLags);
  cudaMalloc((void**)&workArray.Tf,sizeof(float2)*numComps*numComps*particleBlockSize*params.numFreqs);
  cudaMalloc((void**)&workArray.Swhole,sizeof(float2)*numComps*numComps*params.numFreqs*particleBlockSize);
  cudaMalloc(&workArray.tmp,sizeof(float2)*particleBlockSize*params.numFreqs*numComps*numComps);
  cudaMalloc((void**)&workArray.Spartial,sizeof(float2)*(numComps-1)*(numComps-1)*params.numFreqs*particleBlockSize);
  cudaMalloc(&workArray.d_wholeSpec,sizeof(float2)*params.numFreqs*particleBlockSize*(numComps-1)*(numComps-1));
  cudaMalloc(&workArray.dev_W,sizeof(float)*particleBlockSize*params.numFreqs*(numComps-1));
  cudaMalloc(&workArray.d_info,sizeof(int)*particleBlockSize*params.numFreqs);  
  cudaMalloc(&workArray.d_work2,sizeof(float2)*workArray.lworkVal);
  cudaMalloc(&workArray.det_whole,sizeof(float)*particleBlockSize*params.numFreqs);
  cudaMalloc(&workArray.det_partial,sizeof(float)*particleBlockSize*params.numFreqs);
  cudaMalloc(&workArray.dev_GC,sizeof(float)*particleBlockSize);
  cudaMalloc(&workArray.lagList_DEVICE,sizeof(int)*lagList.size());
  cudaMemcpy(workArray.lagList_DEVICE,lagList.data(),sizeof(int)*lagList.size(),cudaMemcpyHostToDevice);
  cudaMalloc(&workArray.ARdev,sizeof(float)*AR.size());
  cudaMemcpy(workArray.ARdev,AR.data(),sizeof(float)*AR.size(),cudaMemcpyHostToDevice);
  
  return;
}

void freeWorkArray(workForGranger &workArray)
{
  cudaFree(workArray.Qdev);
  cudaFree(workArray.rotatedModels);
  cudaFree(workArray.wArray);
  cudaFree(workArray.Tf);
  cudaFree(workArray.Swhole);
  cudaFree(workArray.tmp);
  cudaFree(workArray.Spartial);
  cudaFree(workArray.d_wholeSpec);
  cudaFree(workArray.dev_W);
  cudaFree(workArray.d_info);
  cudaFree(workArray.d_work2);
  cudaFree(workArray.det_whole);
  cudaFree(workArray.det_partial);
  cudaFree(workArray.dev_GC);
  cudaFree(workArray.lagList_DEVICE);
  cudaFree(workArray.ARdev);
  return;
}
