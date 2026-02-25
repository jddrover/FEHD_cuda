#ifndef MKARGPU_H
#define MKARGPU_H
#include "utility.h"
#include <dataClass.h>

MVAR<float> mkARGPU(dataClass<float>, std::vector<int>,paramContainer);
void orthonormalizeR(dataList, dataList &, matrix &);
void rotate_model(ARmodel &, matrix L);
#endif 
