#ifndef MKARGPU_H
#define MKARGPU_H

void mkARGPU(dataList, std::vector<int>, ARmodel &, dataList &);
void orthonormalizeR(dataList, dataList &, matrix &);
void rotate_model(ARmodel &, matrix L);
#endif 
