#include "utility.h"
#include "timeSeriesOPs.h"
#include "dataContainers.h"
#include "mkARGPU.h"
#include "FEHD.h"
#include <vector>
#include <iostream>


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
  
  // Declare a dataArray object.
  // This is mine. It isn't a great fit here, but it does work for other purposes.
  dataList dataArray;
  // Load the data from file to memory.
  loadFile(params.filename, params.numChannels, params.numEpochs, params.epochPts, dataArray);

  // If the laglist was given in a file, take care of that here.
  if(params.lagListFLAG == 1)
    {
      loadLagList(params);
      params.numLags = params.lagList.size();
    }
  else // In the event it wasn't it just assigns a sequence up to the numLags parameter.
    {
      for(int lag=0;lag<params.numLags;lag++)
	params.lagList.push_back(lag+1);
    }
  
  // Compute the principal components and discard unused components.
  // We also need the transformation matrix that generates the principal components. 
  // Each of these should have params.numPCs rows.

  dataList PC;
  matrix Lmat;

  PCA(dataArray,PC,Lmat);
  
  std::vector<int> compsToRemove; // Empty, fill below

  for(int comp=params.numPCs;comp<params.numChannels;comp++)
    compsToRemove.push_back(comp);

  std::vector<float> LmatTrimmed(params.numPCs*params.numChannels,0);
  
  // Need to trim Lmat to numPCs rows;
  for(int row=0;row<params.numPCs;row++)
    for(int col=0;col<params.numChannels;col++)
      LmatTrimmed[col*params.numPCs+row] = Lmat.elements[col*params.numChannels+row];
	  
  removeMultipleComponents(PC,compsToRemove);

  // Run FEHD on the principal components.
  // This is where the algorithm begins on the principal components determined above. 
  runFEHD(PC, LmatTrimmed, params);
  // Write the transformation matrix to file. 
  writeOutput(LmatTrimmed.data(), params);
    
  return 0;
  
}

