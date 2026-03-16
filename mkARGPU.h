#ifndef MKARGPU_H
#define MKARGPU_H
#include "utility.h"
#include "dataClass.h"

MVAR<float> mkARGPU(dataClass<float>,paramContainer);
//void orthonormalizeR(dataList, dataList &, matrix &);
MVAR<float> rotate_model(MVAR<float>, std::vector<float>);
#endif 
