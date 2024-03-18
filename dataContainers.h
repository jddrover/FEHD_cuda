#ifndef DATACONTAINERS_H
#define DATACONTAINERS_H
#include <array>
#include <vector>
#include <list>
#include <string>

struct timePoint
{
  std::vector<float> dataVector;
  std::string timeString;
};

struct freqPoint
{
  std::vector<float> frequency;
  std::vector<float> power;
};

struct dataEpoch
{
  std::vector<timePoint> timePointArray;
  std::vector<freqPoint> powerSpectrum;
};

struct dataList
{
  int numEpochs;
  int numComps;
  int epochPts;
  
  std::vector<dataEpoch> epochArray;
};

struct matrix
{
  std::vector<float> elements;
};

struct ARmodel
{
  std::vector<matrix> lagMatrices;
};
  
#endif

  

  
