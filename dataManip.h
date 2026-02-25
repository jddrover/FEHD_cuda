#ifndef DATAMANIP_H
#define DATAMANIP_H

#include "dataClass.h"
#include <string>
#include <vector>


// Butterworth filters. Need some testing.
dataClass<float> butterFilter(dataClass<float>,std::string,float,int);
dataClass<double> butterFilter(dataClass<double>,std::string,double,int);
// Multiplies a matrix and the data.
dataClass<float> linearTrans(dataClass<float>,std::vector<float>);
dataClass<double> linearTrans(dataClass<double>,std::vector<double>);
// Removes the mean, epoch by epoch.
dataClass<float> removeMean(dataClass<float>);
dataClass<double> removeMean(dataClass<double>);
// Point by point multiplication
// Created as a utility to apply tapers
dataClass<float> applyTapers(dataClass<float>, dataClass<float>);
dataClass<double> applyTapers(dataClass<double>, dataClass<double>);

dataClass<float> removeFLines(dataClass<float>, float);
dataClass<double> removeFLines(dataClass<double>,double);

/*
List of additional ways to manipulate the data.
*/
#endif
