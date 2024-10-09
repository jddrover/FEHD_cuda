#ifndef FEHD_H
#define FEHD_H
#include "dataContainers.h"
#include "utility.h"
// Computes the gradient for multiple rotations of an AR model.
void compGradient(ARmodel, std::vector<float> &, std::vector<float>,std::vector<float>,paramContainer,int,std::vector<int>);
// The main call of the FEHD routine
//void runFEHD(dataList, std::vector<float> &, paramContainer);
// Executes the minimization of the upward Granger causality.
//void runFEHDstep(std::vector<float> &, matrix &, dataList, paramContainer, int);
// Computes a single rotation matrix from a list of angles. 
void singleQ(std::vector<float> &, std::vector<float>);

//void multMat(float *,float *,int,int);



#endif
