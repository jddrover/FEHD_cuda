#ifndef UTILITY_H
#define UTILITY_H
#include <string>
#include <vector>


struct paramContainer
{
  std::string filename;
  std::string lagListFilename;
  std::string outfolder;
  int outfolderFLAG;
  int filenameFLAG;
  int sampRate;
  int sampRateFLAG;
  int numChannels;
  int numChannelsFLAG;
  int epochPts;
  int epochPtsFLAG;
  int numEpochs;
  int numEpochsFLAG;
  int numPoints;
  int numPointsFLAG;
  int numPCs;
  int numPCsFLAG;
  int numLags;
  int numLagsFLAG;
  int lagListFLAG;
  int numParticles;
  int numParticlesFLAG;
  float freqLo;
  int freqLoFLAG;
  float freqHi;
  int freqHiFLAG;
  int numFreqs;
  int numFreqsFLAG;
  bool verbose;
  std::vector<int> lagList;
} ;

void setUpParameters(int argc,char** argv,paramContainer &params);
void writeOutput(float *transMat,paramContainer params);
void setFLAGStoZero(paramContainer &params);
void printParams(paramContainer params);
void printMatrixfloat(float *M, int lda, int numRows, int numCols);
void loadLagList(paramContainer &params);

#endif
