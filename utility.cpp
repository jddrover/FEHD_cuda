#include <iostream>
#include <string>
#include <fstream>
//#include <experimental/filesystem>
#include <filesystem>
#include <time.h>
#include "utility.h"

//namespace fs = std::experimental::filesystem;
namespace fs = std::filesystem;
void setUpParameters(int argc,char** argv,paramContainer &params)
{

  setFLAGStoZero(params);

  // printf("number of args: %i \n",argc);

  params.verbose = false;
  params.Ptol = 1.0;
  for(int i=1;i<argc;i+=2)
    {
      if(std::string(argv[i]) == "-Ptol")
	params.Ptol = std::stof(std::string(argv[i+1]));
	
      if(std::string(argv[i]) == "-verbose")
	params.verbose = (bool)std::stoi(std::string(argv[i+1]));
	
      if(std::string(argv[i]) == "-filename")
	{
	  params.filename = std::string(argv[i+1]);

	  if(!fs::exists(params.filename))
	    {
	      throw std::invalid_argument("Data file not found. Exiting.");
	      return;
	    }
	  else
	    params.filenameFLAG = 1; 
	}
      if(std::string(argv[i]) == "-lagList")
	{
	  //printf("Reading lag list from file \n");
	  params.lagListFilename = std::string(argv[i+1]);
	  params.numLagsFLAG = 1;
	  params.lagListFLAG = 1;
	}
      if(std::string(argv[i]) == "-sampRate")
	{
	  //printf("Sampling rate specified \n");
	  params.sampRate = std::stoi(std::string(argv[i+1]));
	  params.sampRateFLAG = 1;
	}
      if(std::string(argv[i]) == "-numChannels")
	{
	  //printf("number of channels specified \n");	
	  params.numChannels = std::stoi(std::string(argv[i+1]));
	  params.numChannelsFLAG = 1;
	}
      if(std::string(argv[i]) == "-epochPts")
	{
	  //printf("number of points per epoch specified \n");
	  params.epochPts = std::stoi(std::string(argv[i+1]));
	  params.epochPtsFLAG = 1;
	}
      if(std::string(argv[i]) == "-numEpochs")
	{
	  //printf("number of epochs specified \n");
	  std::cout << "This quantity is computed automatically" << std::endl;
	  //params.numEpochs = std::stoi(std::string(argv[i+1]));
	  //params.numEpochsFLAG = 1;
	}
      if(std::string(argv[i]) == "-numPCs")
	{
	  //printf("number of PCs specified \n");
	  params.numPCs = std::stoi(std::string(argv[i+1]));
	  params.numPCsFLAG = 1;
	}
      if(std::string(argv[i]) == "-numLags")
	{
	  //printf("number of lags specified \n");
	  params.numLags = std::stoi(std::string(argv[i+1]));
	  params.numLagsFLAG = 1;
	}
      if(std::string(argv[i]) == "-numParticles")
	{
	  //printf("number of particles specified \n");
	  params.numParticles = std::stoi(std::string(argv[i+1]));
	  params.numParticlesFLAG = 1;
	}
      if(std::string(argv[i]) == "-freqLo")
	{
	  //printf("low frequency specified \n");
	  params.freqLo = std::stof(std::string(argv[i+1]));
	  params.freqLoFLAG = 1;
	}
      if(std::string(argv[i]) == "-freqHi")
	{
	  //printf("high frequency specified \n");
	  params.freqHi = std::stof(std::string(argv[i+1]));
	  params.freqHiFLAG = 1;
	}
      if(std::string(argv[i]) == "-numFreqs")
	{
	  //printf("number of frequencies specified \n");
	  params.numFreqs = std::stoi(std::string(argv[i+1]));
	  params.numFreqsFLAG = 1;
	}
      if(std::string(argv[i]) == "-outfolder")
	{
	  //std::cout << "directory for output chosen" << std::endl;
	  params.outfolder = std::string(argv[i+1]);
	  params.outfolderFLAG = 1;
	}
      if(std::string(argv[i]) == "-exitcount")
	{
	  params.STUCKCOUNT = std::stoi(std::string(argv[i+1]));
	  params.STUCKCOUNTFLAG = 1;
	}
    }

  FILE *f;


  if(params.STUCKCOUNTFLAG == 0)
    params.STUCKCOUNT = 5;
  
  if(params.filenameFLAG == 0)
    throw std::invalid_argument("No filename provided (-filename). Exiting");
    
  if(params.sampRateFLAG == 0)
    throw std::invalid_argument("Sampling rate was not provided (-sampRate). Exiting");
      
  // Determine the number of channels and time points automatically from the file

  if(params.numPointsFLAG == 0 || params.numChannelsFLAG == 0)
    {
      // This should be rewritten so that it does not depend on wc.
      char numPointsstr[100];

      std::string commandBase = "wc -l -w " + params.filename;
  
      f = popen(commandBase.c_str(),"r");
      fgets(numPointsstr,100,f);
      pclose(f);
      // Trim it up

      size_t skip = std::string(numPointsstr).find_first_not_of(" ");
      commandBase = std::string(numPointsstr).substr(skip);
      skip = commandBase.find(" ");
      params.numPoints = std::stoi(commandBase.substr(0,skip));
      commandBase = commandBase.substr(skip);
      skip = commandBase.find(" ");
      params.numChannels = std::stoi(commandBase.substr(1,skip-1))/params.numPoints;
      params.numPointsFLAG = 1;
      params.numChannelsFLAG = 1;
    }

  if(params.epochPtsFLAG == 0)
    throw std::invalid_argument("Points per epoch (trial) not provided (-epochPts). Exiting");

  // Determine the number of epochs from obtained information.

  params.numEpochs = (int)(params.numPoints/params.epochPts);

  // Check that this divides right.

  if(params.numEpochs*params.epochPts != params.numPoints)
    {
      std::cout << "The number of points do not divide into an integral number of epochs" << std::endl;
      std::cout << "Trimming the data off the end to fit." << std::endl;

      params.numPoints = params.numEpochs*params.epochPts;
    }
  
  if(params.numPCsFLAG == 0)
    throw std::invalid_argument("Number of principal components not specified (-numPCs). Exiting.");

  if(params.numLagsFLAG == 0)
    throw std::invalid_argument("lag list has not been described (-numLags or -lagList). Exiting.");

  if(params.freqLoFLAG == 0)
    throw std::invalid_argument("Lower frequency bound not provided (-freqLo). Exiting.");
    
  if(params.freqHiFLAG == 0)
    throw std::invalid_argument("Upper frequency bound not provided (-freqHi). Exiting.");
    
  if(params.numFreqsFLAG == 0)
    throw std::invalid_argument("Number of frequencies to evaluate not provided (-numFreqs). Exiting.");
    
  if(params.numParticlesFLAG == 0)
    throw std::invalid_argument("Number of particles not provided (-numParticles). Exiting.");
    
  if(params.outfolderFLAG == 0)
    throw std::invalid_argument("Folder for output not specified (-outfolder). Exiting.");
      
      
}


void writeOutput(float *transMat,paramContainer params)
{
  // First, write the meta-data to file.
  
  // See if the output folder exists already.
  
  std::cout << params.outfolder << std::endl;
  
  //fs::current_path(fs::temp_directory_path());
  
  fs::path fullpath = fs::current_path();
  
  fullpath /= params.outfolder;

  fs::path metapath = fullpath;
  fs::path datapath = fullpath;



  // Make the filenames here

  time_t rawtime;
  time(&rawtime);
  struct tm * timeinfo;
  timeinfo = localtime(&rawtime);

  srand((unsigned)time(NULL));

  int rn;
  rn = rand()%1000; // A little something to prevent repeats
  //printf("%i \n",rn);
  
  std::string dateString = std::to_string(timeinfo->tm_mon)+
    std::to_string(timeinfo->tm_mday)+std::to_string(timeinfo->tm_hour)+
    std::to_string(timeinfo->tm_min)+std::to_string(timeinfo->tm_sec)+
    std::to_string(rn);

  std::string metafile = "metadata_" + dateString;
  std::string datafile = "data_" + dateString;
  
  metapath /= metafile;
  datapath /= datafile;
  
  fs::create_directory(fullpath);
  
  
  std::ofstream metaStream (metapath);
  
  metaStream << "Data file = " << params.filename << std::endl;
  metaStream << "Sampling rate = " << params.sampRate << std::endl;
  metaStream << "Number of channels = " << params.numChannels << std::endl;
  metaStream << "Points per epoch = " << params.epochPts << std::endl;
  metaStream << "Number of epochs = " << params.numEpochs << std::endl;
  metaStream << "Number of points = " << params.numPoints << std::endl;
  metaStream << "Number of PC = " << params.numPCs << std::endl;
  metaStream << "Number of lags = " << params.numLags << std::endl;
  metaStream << "Number of particles = " << params.numParticles << std::endl;
  metaStream << "low frequency = " << params.freqLo << std::endl;
  metaStream << "high frequency = " << params.freqHi << std::endl;
  metaStream << "number of frequencies = " << params.numFreqs << std::endl;
  metaStream << "Exit count = " << params.STUCKCOUNT << std::endl;
  metaStream << "AR variance tolerance = " << params.Ptol << std::endl;
  metaStream.close();

  // Now write the data.
  
  std::ofstream dataStream (datapath);

  // What are the dimensions of this?
  // The number of components x the number of channels.
  // Stored column-major

  for(int row=0;row<params.numPCs;row++)
    {
      for(int col=0;col<params.numChannels;col++)
	{
	  dataStream << transMat[col*params.numPCs+row] << " ";
	}
      dataStream << std::endl;
    }
  
  dataStream.close();
	
}

void setFLAGStoZero(paramContainer &params)
{
    
  params.filenameFLAG = 0;
  params.outfolderFLAG = 0;
  params.sampRateFLAG = 0;
  params.numChannelsFLAG = 0;
  params.epochPtsFLAG = 0;
  params.numEpochsFLAG = 0;
  params.numPointsFLAG = 0;
  params.numPCsFLAG = 0;
  params.numLagsFLAG = 0;
  params.numParticlesFLAG = 0;
  params.freqLoFLAG = 0;
  params.freqHiFLAG = 0;
  params.numFreqsFLAG = 0;
  params.STUCKCOUNTFLAG = 0;
}
  
void printParams(paramContainer params)
{
   if(params.filenameFLAG)
    printf("filename = %s \n",params.filename.c_str());
  if(params.sampRateFLAG)
    printf("sampling rate = %i \n",params.sampRate);
  if(params.numChannelsFLAG)
    printf("number of channels = %i \n",params.numChannels);
  if(params.epochPtsFLAG)
    printf("points per epoch = %i \n",params.epochPts);
  if(params.numEpochsFLAG)
    printf("number of epochs = %i \n",params.numEpochs);
  if(params.numPointsFLAG)
    printf("number of time points = %i \n",params.numPoints);
  if(params.numPCsFLAG)
    printf("number of principal components = %i \n",params.numPCs);
  if(params.numLagsFLAG)
    printf("number of lags in AR model = %i \n",params.numLags);
  if(params.numParticlesFLAG)
    printf("number of particles for minimization = %i \n",params.numParticles);
  if(params.freqLoFLAG)
    printf("low end of frequency = %f \n",params.freqLo);
  if(params.freqHiFLAG)
    printf("high end of frequnecy = %f \n",params.freqHi);
  if(params.numFreqsFLAG)
    printf("number of frequencies = %i \n",params.numFreqs);
  
}

void printMatrixfloat(float *M, int lda, int numRows, int numCols)
{
  for(int row=0;row<numRows;row++)
    {
      for(int col=0;col<numCols;col++)
	{
	  printf("%f ",M[col*lda+row]);
	}
      printf("\n");
    }
  return;
}

void loadLagList(paramContainer &params)
{
  std::ifstream dStream(params.lagListFilename.c_str(),std::ifstream::in);
  if(!dStream.is_open())
    {
      //std::cerr << "Laglist file not found\n";
      throw std::runtime_error("Laglist file not found. Exiting");
    }
  else
    {
      int tmpVal;
      while(dStream.good())
	{
	  dStream >> tmpVal;
	  params.lagList.push_back(tmpVal);
	}                                        
    }
  dStream.close();

  return;
}

