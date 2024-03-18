#include "utility.h"
#include "timeSeriesOPs.h"
#include "dataContainers.h"
#include "mkARGPU.h"
#include "FEHD.h"
#include <vector>
#include <iostream>


int main(int argc, char** argv)
{

  // This will hold all of the parameter values.
  paramContainer params;
  // Determine parameters
  // This function will read the given command line arguments
  // and uses widgets to get the remaining information.
  // The widgets aren't going to work for anyone but me.
  setUpParameters(argc,argv,params);

  if(params.verbose)
    {
      printf("Verbosity selected \n");
    }
  // Load the data from the file into "dataArray"

  
  dataList dataArray;
  // Rewrite this definition to just take the two arguments.
  loadFile(params.filename, params.numChannels, params.numEpochs, params.epochPts, dataArray);

  if(params.lagListFLAG == 1)
    {
      loadLagList(params);
      params.numLags = params.lagList.size();
    }
  else
    {
      for(int lag=0;lag<params.numLags;lag++)
	params.lagList.push_back(lag+1);
    }
  
  // Compute the principal components and discard unused components.
  // We also need the transformation matrix that generates the principal components. 
  // Each of these should have params.numPCs rows.
  // Future option is to skip pca and put the data as-is into the FEHD algorithm.
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
  
  runFEHD(PC, LmatTrimmed, params);

  writeOutput(LmatTrimmed.data(), params);
    
  return 0;
  
}

