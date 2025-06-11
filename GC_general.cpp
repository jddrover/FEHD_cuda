#include <cblas.h>
#include <lapacke.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <complex>
#include <algorithm>
#include "utility.h"
#include <string>

float GCgen(std::vector<float> dataArray, std::vector<int> caused, paramContainer params)
{
  float GC;

  // First, need to organize so that the caused components are on top.
  int numCausedComps = caused.size();
  std::vector<float> dataArraySorted(dataArray);
  int counterCaused=0;
  int counterNot=0;
  for(int comp=0;comp<params.numChannels;comp++)
    {
      if(std::find(caused.begin(),caused.end(),comp) != caused.end())
	{
	  cblas_scopy(params.numPoints,dataArray.data()+comp,params.numChannels,dataArraySorted.data()+counterCaused,params.numChannels);
	  counterCaused++;
	}
      else
	{
	  cblas_scopy(params.numPoints,dataArray.data()+comp,params.numChannels,dataArraySorted.data()+counterNot+numCausedComps,
		      params.numChannels);
	  counterNot++;
	}
    }

  dataArray = dataArraySorted;
  
  std::vector<float> freq(params.numFreqs,0.0);
  for(int findx=0;findx<params.numFreqs;findx++)
    freq[findx] = (params.freqHi-params.freqLo)/(params.numFreqs-1)*float(findx)+params.freqLo;

  std::sort(params.lagList.begin(),params.lagList.end());
  int maxLag = params.lagList[params.numLags-1];
  int IPIV[params.numChannels*params.numLags];
  


  std::vector<float> RHS(params.numChannels*params.numEpochs*(params.epochPts-maxLag),0.0);
  std::vector<float> LHS(params.numChannels*params.numEpochs*params.numLags*(params.epochPts-maxLag),0.0);

  for(int epoch=0;epoch<params.numEpochs;epoch++)
    {
      std::copy(dataArray.begin()+(epoch*params.epochPts+maxLag)*params.numChannels,
		dataArray.begin()+(epoch+1)*params.epochPts*params.numChannels,
		RHS.begin()+epoch*(params.epochPts-maxLag)*params.numChannels);

      for(int lag=0;lag<params.numLags;lag++)
      {
        for(int tp=0;tp<params.epochPts-maxLag;tp++)
          {
            std::copy(dataArray.begin()+epoch*params.numChannels*params.epochPts+
		      (maxLag-params.lagList[lag])*params.numChannels+tp*params.numChannels,
		      dataArray.begin()+epoch*params.numChannels*params.epochPts+
		      (maxLag-params.lagList[lag])*params.numChannels+(tp+1)*params.numChannels,
		      LHS.begin()+epoch*params.numChannels*params.numLags*(params.epochPts-maxLag)+
		      lag*params.numChannels+tp*params.numChannels*params.numLags);
          }
      }
    }
  std::vector<float> LCOV(params.numLags*params.numChannels*params.numLags*params.numChannels,0.0);
  std::vector<float> RCOV(params.numChannels*params.numLags*params.numChannels,0.0);
  int info;
  cblas_ssyrk(CblasColMajor,CblasUpper,CblasNoTrans,params.numChannels*params.numLags,
	      params.numEpochs*(params.epochPts-maxLag),
	      1.0,LHS.data(),params.numLags*params.numChannels,
	      0.0,LCOV.data(),params.numLags*params.numChannels);

  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,params.numChannels*params.numLags,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),
	      1.0,LHS.data(),params.numLags*params.numChannels,
	      RHS.data(),params.numChannels,
	      0.0,RCOV.data(),params.numChannels*params.numLags);

  info = LAPACKE_ssysv(LAPACK_COL_MAJOR,'U',params.numChannels*params.numLags,params.numChannels,
		       LCOV.data(),params.numChannels*params.numLags,IPIV,
			     RCOV.data(),params.numChannels*params.numLags);
  // At this stage, RCOV contains the autoregressive coefficients, transposed.

  std::vector<float> A(params.numLags*params.numChannels*params.numChannels,0.0);

  // Compute the residuals

  cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),params.numChannels*params.numLags,
	      -1.0,RCOV.data(),params.numLags*params.numChannels,
	      LHS.data(),params.numLags*params.numChannels,
	      1.0,RHS.data(),params.numChannels);
	      
  // The covariance of the residuals
  std::vector<float> resCOV(params.numChannels*params.numChannels,0.0);
  cblas_ssyrk(CblasColMajor,CblasUpper,CblasNoTrans,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),
	      1.0,RHS.data(),params.numChannels,
	      0.0,resCOV.data(),params.numChannels);

  std::vector<float> P(params.numChannels*params.numChannels,0.0);
  std::vector<float> Pinv(params.numChannels*params.numChannels,0.0);

  for(int diagentry=0;diagentry<params.numChannels;diagentry++)
    P[diagentry*params.numChannels+diagentry]=1.0;

  // This is a little tricky - need an inverse of a symmetric matrix.
  // I am thinking just do it, and fix later.
  
  return (float)info;
}

void loadData(std::string filename,int numComps,int numEpochs,int epochPts,std::vector<float> &dataArray)
{
  std::ifstream dStream(filename.c_str(),std::ifstream::in);

  for(unsigned long int point=0;point<numEpochs*epochPts*numComps;point++)
    dStream >> dataArray[point];
  
  return;
}

void SUP(int argc,char** argv,paramContainer &params)
{
  params.transnameFLAG=0;
  params.filenameFLAG=0;
  params.lagListFLAG=0;
  params.sampRateFLAG=0;
  params.numChannelsFLAG=0;
  params.epochPtsFLAG=0;
  params.numLagsFLAG=0;
  params.freqLoFLAG=0;
  params.freqHiFLAG=0;
  params.numFreqs=0;
  params.numPCsFLAG=0;
  FILE *f;
  for(int i=1;i<argc;i+=2)
    {
      if(std::string(argv[i]) == "--datafile")
	{
	  //printf("Filename option specified \n");
	  params.filename = std::string(argv[i+1]);
	  // Try to open this
	  std::ifstream tester(params.filename.c_str(),std::ifstream::in);
	  if(!tester.is_open())
	    {
	      throw std::invalid_argument("Data file not found. Exiting.");
	      return;
	    }
	  tester.close();
	  params.filenameFLAG = 1; 
	}
      if(std::string(argv[i]) == "--FEHDtransform")
	{
	  params.transname = std::string(argv[i+1]);
	  // Try to open this
	  std::ifstream tester(params.transname.c_str(),std::ifstream::in);
	  if(!tester.is_open())
	    {
	      throw std::invalid_argument("Data file not found. Exiting.");
	      return;
	    }
	  params.transnameFLAG = 1;
	}
      if(std::string(argv[i]) == "--lagList")
	{
	  //printf("Reading lag list from file \n");
	  params.lagListFilename = std::string(argv[i+1]);
	  params.numLagsFLAG = 1;
	  params.lagListFLAG = 1;
	}
      if(std::string(argv[i]) == "--sampRate")
	{
	  //printf("Sampling rate specified \n");
	  params.sampRate = std::stoi(std::string(argv[i+1]));
	  params.sampRateFLAG = 1;
	}
      if(std::string(argv[i]) == "--numChannels")
	{
	  //printf("number of channels specified \n");	
	  params.numChannels = std::stoi(std::string(argv[i+1]));
	  params.numChannelsFLAG = 1;
	}
      if(std::string(argv[i]) == "--epochPts")
	{
	  //printf("number of points per epoch specified \n");
	  params.epochPts = std::stoi(std::string(argv[i+1]));
	  params.epochPtsFLAG = 1;
	}
      if(std::string(argv[i]) == "--numEpochs")
	{
	  //printf("number of epochs specified \n");
	  std::cout << "This quantity is computed automatically" << std::endl;
	  //params.numEpochs = std::stoi(std::string(argv[i+1]));
	  //params.numEpochsFLAG = 1;
	}

      if(std::string(argv[i]) == "--numLags")
	{
	  //printf("number of lags specified \n");
	  params.numLags = std::stoi(std::string(argv[i+1]));
	  params.numLagsFLAG = 1;
	}
      if(std::string(argv[i]) == "--freqLo")
	{
	  //printf("low frequency specified \n");
	  params.freqLo = std::stof(std::string(argv[i+1]));
	  params.freqLoFLAG = 1;
	}
      if(std::string(argv[i]) == "--freqHi")
	{
	  //printf("high frequency specified \n");
	  params.freqHi = std::stof(std::string(argv[i+1]));
	  params.freqHiFLAG = 1;
	}
      if(std::string(argv[i]) == "--numFreqs")
	{
	  //printf("number of frequencies specified \n");
	  params.numFreqs = std::stoi(std::string(argv[i+1]));
	  params.numFreqsFLAG = 1;
	}
    }
  if(params.filenameFLAG == 0)
    throw std::invalid_argument("No filename provided (--datafile). Exiting.");
  if(params.sampRateFLAG == 0)
    throw std::invalid_argument("Sampling rate was not provided (-sampRate). Exiting.");
  if(params.epochPtsFLAG == 0)
    throw std::invalid_argument("Number of points per epoch not provided. Exiting.");
  if(params.numPointsFLAG == 0 || params.numChannelsFLAG == 0)
    {
      std::vector<float> dataTest;

      //std::cout << "Determining channels and time points from file size" << std::endl;
      // I use the system command wc to do this.
      std::ifstream file(params.filename);
      if(!file.is_open()){
	std::cerr << "Error opening file" << std::endl;
	return;
      }

      std::string line;
      std::string word;
      float wordG;
      int numColumns=0;
      std::getline(file,line);
      
      std::stringstream ss(line);
      while(ss >> word)
	numColumns++;
      
      file.seekg(0,std::ios::beg);

      while(file >> wordG)
	{
	  dataTest.push_back(wordG);
	}
      file.close();
      //std::cout << numColumns << std::endl;
      //std::cout << dataTest.size() << std::endl;
      params.numChannels = numColumns;
      params.numPoints = dataTest.size()/numColumns;
      params.numEpochs = params.numPoints/params.epochPts;
      if(params.transnameFLAG)
	{
	  std::ifstream tfile(params.transname);
	  if(!tfile.is_open()){
	    std::cerr << "Error opening file" << std::endl;
	    return;
	  }
	  
	  std::getline(tfile,line);
	  int transcols=0;
	  std::vector<float> transvec;
	  std::stringstream s2(line);
	  while(s2 >> word)
	    transcols++;
	  
	  tfile.seekg(0,std::ios::beg);
      

	  while(tfile >> wordG)
	    {
	      transvec.push_back(wordG);
	    }
	  params.numPCs = transvec.size()/transcols; // Rows of transform
	  params.numPCsFLAG=1;

	}
      
    }

  if(params.numLagsFLAG == 0)
    throw std::invalid_argument("Neither the number of lags nor a laglist provided. Exiting.");
  if(params.freqLoFLAG == 0)
    throw std::invalid_argument("Lower frequency bound not provided. Exiting.");
  if(params.freqHiFLAG == 0)
    throw std::invalid_argument("Upper frequnecy bound not provided. Exiting.");
  if(params.numFreqsFLAG == 0)
     throw std::invalid_argument("Number of frequencies not provided. Exiting.");
  
  return;
}

int main(int argc, char** argv)
{

  paramContainer params;
  try
    {
      SUP(argc,argv,params);
    }
  
  catch (std::invalid_argument e)
    {
      std::cerr << e.what() << std::endl;
      return -1;
    }

  std::vector<float> dataArray(params.numChannels*params.numPoints,0.0);
  std::vector<float> dataTmp(params.numPCs*params.numPoints,0.0);
  loadData(params.filename,params.numChannels,params.numEpochs,params.epochPts,dataArray);
  std::vector<float> transArray(params.numPCs*params.numChannels,0.0);

  if(params.numPCsFLAG)
    {
     
      loadData(params.transname,params.numPCs,1,params.numChannels,transArray);
      
      cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,params.numPCs,params.numPoints,params.numChannels,
		  1.0,transArray.data(),params.numChannels,dataArray.data(),params.numChannels,
		  0.0,dataTmp.data(),params.numPCs);
      
      dataArray = dataTmp;
      params.numChannels = params.numPCs;
    }

  std::vector<int> lagList(params.numLags,0);
  for(int lag=0;lag<params.numLags;lag++)
    lagList[lag] = lag+1;
 
  params.lagList = lagList;
  std::vector<int> caused = {0,1,2,3,4,5,6,7,8}; // Note the zero indexing -  this is going to get some people. 
  //std::vector<float> X(params.numChannels*params.numChannels,0.0);
   
  float GCval;

  GCval = GCgen(dataArray,caused,params);
  std::cout << GCval << std::endl;
  //X = PGC(dataArray,params);
  /*for(int row=0;row<params.numChannels;row++)
    {
      for(int col=0;col<params.numChannels;col++)
	std::cout << X[col*params.numChannels+row] << " ";
      std::cout << "\n";
      }*/
  //std::cout << std::endl;
  return 0;
}

  
