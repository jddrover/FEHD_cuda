#ifndef MKARGPU_H
#define MKARGPU_H
#include "utility.h"

void mkARGPU(dataList, std::vector<int>, ARmodel &, dataList &,paramContainer);
void orthonormalizeR(dataList, dataList &, matrix &);
void rotate_model(ARmodel &, matrix L);
#endif 
