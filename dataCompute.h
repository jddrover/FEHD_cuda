#ifndef DATACOMPUTE_H
#define DATACOMPUTE_H

#include "dataClass.h"
#include <complex>
dataClass<std::complex<float>> FFT(dataClass<float>);
dataClass<std::complex<double>> FFT(dataClass<double>);
dataClass<float> dpss(int N,float WT,int numTapers);
dataClass<double> dpss(int N,double WT,int numTapers);
std::vector<std::vector<std::complex<float>>> computeSpectra(dataClass<float>,dataClass<float>);
MVAR<float> mkAR(dataClass<float>,std::vector<int>,float=1.0);
std::vector<float> PCA(dataClass<float>);
/*
Method ideas here
*/
#endif
