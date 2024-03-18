#ifndef FEHD_H
#define FEHD_H
#include "dataContainers.h"
#include "utility.h"
// Computes the gradient for multiple rotations of an AR model.
void compGradient(ARmodel, std::vector<float> &, std::vector<float>,std::vector<float>,paramContainer,int,std::vector<int>);

void runFEHD(dataList, std::vector<float> &, paramContainer);

void runFEHDstep(std::vector<float> &, matrix &, dataList, paramContainer, int);

void singleQ(std::vector<float> &, std::vector<float>);

//void multMat(float *,float *,int,int);



#endif
