#ifndef TIMESERIESOPS_H
#define TIMESERIESOPS_H
#include "dataContainers.h"
#include <vector>
#include <string>
void loadFile(std::string,int,int,int,dataList &);
void removeEpoch(dataList &DS,int epochToRemove);
void removeMultipleEpochs(dataList &DS,std::vector<int> &epochsToRemove);
void convertDataListToRawArray(dataList DS,float *rawArray);
void convertRawArrayToDataList(float *rawArray,dataList &DL,int numComps,int epochPts,int numEpochs);
void PCA(dataList DS,dataList &PC, matrix &Tm);
void removeComponent(dataList &,int);
void removeMultipleComponents(dataList &,std::vector<int> compsToRemove);
float computeDeterminant(std::vector<float>, bool);
void generateNextLagList(std::vector<int> &);
#endif
