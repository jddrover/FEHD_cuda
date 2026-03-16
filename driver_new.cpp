#include "utility.h"
#include "FEHD.h"
#include <vector>
#include <iostream>
#include "dataClass.h"
#include "dataManip.h"
#include "dataCompute.h"

int main(int argc, char** argv)
{
  // The paramContainer object contains all of the analysis parameters
  // such as sampling rate and the number of particles for the optimizer.
  paramContainer params;
  // The setUpParameters call fills the params object. If it is not
  // completely filled an exception is thrown and execution is terminated.
  try
    {
      setUpParameters(argc,argv,params);
    }
  // If there was a problem above (ie missing argument), the exception is
  // noted here, and execution is terminated with an error message. 
  catch (std::invalid_argument e)
    {
      std::cerr << e.what() << std::endl;
      return -1;
    }

  std::vector<float> dataArray;
  loadFile(params.filename, dataArray);

  dataClass<float> dataSet(params.epochPts,params.numChannels,dataArray,params.sampRate);
  dataClass<float> dataRM = removeMean(dataSet);
  
  // If the laglist was given in a file, load the file.
  if(params.lagListFLAG == 1)
    {
      try
	{
	  loadLagList(params);
	}
      catch(std::runtime_error e)
	{
	  std::cerr << e.what() << std::endl;
	  return -1;
	}
      params.numLags = params.lagList.size();
    }

  else // In the event it wasn't it just assigns a sequence up to the numLags parameter.
    {
      for(int lag=0;lag<params.numLags;lag++)
	params.lagList.push_back(lag+1);
    }

  std::vector<float> PCTrans = PCA(dataRM);
  dataClass<float> PC = linearTrans(dataRM,PCTrans);
  int PCTransDim = PC.getNumComps();

  if(PC.getNumComps() < params.numPCs)
    {
      std::cout << "Too many Principal Components requested." << std::endl;
      exit(1);
    }

  std::vector<int> compsToKeep;
  for(int comp=0;comp<params.numPCs;comp++)
    compsToKeep.push_back(comp);
  PC.keepComponents(compsToKeep);
  
  std::vector<float> LmatTrimmed(params.numPCs*params.numChannels,0);
  // Need to trim PCtrans to numPCs rows;
  for(int row=0;row<params.numPCs;row++)
    for(int col=0;col<params.numChannels;col++)
      LmatTrimmed[col*params.numPCs+row] = PCTrans[col*PCTransDim+row];

  // Run FEHD on the principal components.
  runFEHD(PC, LmatTrimmed, params);
  // Write the transformation matrix to file.
  // This could use an update.
  writeOutput(LmatTrimmed.data(), params);
  
  return 0;
  
}

