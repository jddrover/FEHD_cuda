#include "timeSeriesOPs.h"
#include "dataContainers.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <string>

void loadFile(std::string filename,int numComps,int numEpochs,int epochPts,dataList &dataSet)
{
  std::ifstream dStream(filename.c_str(),std::ifstream::in);

  dataEpoch epochtmp;
  timePoint timePointtmp;

  float tempVal;
  
  if(dStream.is_open())
    {
      for(int epoch=0;epoch<numEpochs;epoch++)
	{
	  if(!epochtmp.timePointArray.empty())
	    {
	      std::cerr << "epoch is not empty" << std::endl;
	    }
	  for(int TP=0;TP<epochPts;TP++)
	    {
	      // Check that the timePoint array structure is clear
	      if(!timePointtmp.dataVector.empty())
		{
		  std::cerr << "time point is not empty" << std::endl;
		}
	      for(int comp=0;comp<numComps;comp++)
		{
		  dStream >> tempVal;
		  timePointtmp.dataVector.push_back(tempVal);
		}
	      epochtmp.timePointArray.push_back(timePointtmp);
	      timePointtmp.dataVector.clear();
	    }
	  dataSet.epochArray.push_back(epochtmp);
	  epochtmp.timePointArray.clear();
	}
      dStream.close();
      dataSet.numEpochs=numEpochs;
    }
  else
    throw std::runtime_error("Datafile not found");
  
  
}

void removeEpoch(dataList &DS,int epochToRemove)
{
  DS.epochArray.erase(DS.epochArray.begin()+epochToRemove);
}

void removeMultipleEpochs(dataList &DS,std::vector<int> &epochsToRemove)
{
  // Sort the vector so that the strategy below works. 
  std::sort(epochsToRemove.begin(),epochsToRemove.end());

  for(int i=0;i<epochsToRemove.size();i++)
    {
      removeEpoch(DS,epochsToRemove[i]-i);
    }
}

void convertDataListToRawArray(dataList DS,float *rawArray)
{
  // This can be better, with some structural changes.
  int numEpochs = DS.epochArray.size();
  int epochPts = DS.epochArray[0].timePointArray.size();
  int numComps = DS.epochArray[0].timePointArray[0].dataVector.size();
  
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      for(int TP=0;TP<epochPts;TP++)
	{
	  for(int comp=0;comp<numComps;comp++)
	    {
	      rawArray[epoch*epochPts*numComps+TP*numComps+comp] = DS.epochArray[epoch].timePointArray[TP].dataVector[comp];
	    }
	}
    }
}

void convertRawArrayToDataList(float *rawArray,dataList &DL,int numComps,int epochPts,
			       int numEpochs)
{
  dataEpoch epochtmp;
  timePoint timePointtmp;
  float tempVal;

  DL.epochArray.clear();
  
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      for(int TP=0;TP<epochPts;TP++)
	{
	  // Check that the timePoint array structure is clear
	  if(!timePointtmp.dataVector.empty())
	    {
	      std::cerr << "time point is not empty" << std::endl;
	    }
	  for(int comp=0;comp<numComps;comp++)
	    {
	      tempVal = rawArray[epoch*epochPts*numComps+TP*numComps+comp];
	      timePointtmp.dataVector.push_back(tempVal);
	    }
	  epochtmp.timePointArray.push_back(timePointtmp);
	  timePointtmp.dataVector.clear();
	}
      DL.epochArray.push_back(epochtmp);
      epochtmp.timePointArray.clear();
    }  
}


// Compute the principal components
void PCA(dataList DS,dataList &PC, matrix &transMat)
{
  
  // Convert DS to a raw array to be used in blas lapack.

  int numEpochs = DS.epochArray.size();
  int epochPts = DS.epochArray[0].timePointArray.size();
  int numComps = DS.epochArray[0].timePointArray[0].dataVector.size();
  int numPoints = numEpochs*epochPts;

  
  
  std::vector<float> dataArray(numPoints*numComps,0.0);
    
  convertDataListToRawArray(DS,dataArray.data());
  
  std::vector<float> DAT(dataArray);
  // I need to flip this. Graceful.
  for(int row=0;row<numPoints;row++)
    for(int col=0;col<numComps;col++)
      dataArray[col*numPoints+row] = DAT[row*numComps+col];
  
  float superb[numComps-1];

  std::vector<float> sMat(numComps,0.0);
  std::vector<float> uMat(numComps*numPoints,0.0);
  std::vector<float> vMatT(numComps*numComps,0.0);

  int info;

  info = LAPACKE_sgesvd(LAPACK_COL_MAJOR,'S','A',numPoints,numComps,
			dataArray.data(),numPoints,
			sMat.data(),
			uMat.data(),numPoints,
			vMatT.data(),numComps,superb);

  if(info != 0)
    {
      std::cout << "SVD in PCA did not exit successfully" << std::endl;
      exit(1);
    }
  
  //std::cout << info << std::endl;
  


  // Do a check here - look for very small svs
  float tol = 1e-2;
  float scaleVal;
  for(int row=0;row<numComps;row++)
    {
      if(sMat[row]<tol)
	{
	  std::cout << "In PCA, there is a very small singular value." << sMat[row] << std::endl;
	  std::cout << "If things go south, look here" << std::endl;
	}
      scaleVal = 1.0f/sMat[row];
      cblas_sscal(numComps,scaleVal,vMatT.data()+row,numComps);
    }
  
  std::vector<float> uMatT(uMat);
  for(int row=0;row<numComps;row++)
    for(int col=0;col<numPoints;col++)
      uMatT[col*numComps+row] = uMat[row*numPoints+col];
  
  convertRawArrayToDataList(uMatT.data(),PC,numComps,epochPts,numEpochs);
  
  transMat.elements.insert(transMat.elements.begin(),vMatT.begin(),vMatT.end());

}


void removeComponent(dataList &DL,int compToRemove)
{
  int numEpochs = DL.epochArray.size();
  int epochPts = DL.epochArray[0].timePointArray.size();
  int numComps = DL.epochArray[0].timePointArray[0].dataVector.size();

  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      for(int TP=0;TP<epochPts;TP++)
	{
	  DL.epochArray[epoch].timePointArray[TP].dataVector.erase
	    (DL.epochArray[epoch].timePointArray[TP].dataVector.begin()+
	     compToRemove);
	}
    }
  DL.numComps = DL.numComps - 1;
  
}

void removeMultipleComponents(dataList &DL,std::vector<int> compsToRemove)
{
  // Sort the components so the below strategy works.
  std::sort(compsToRemove.begin(),compsToRemove.end());

  for(int comp=0;comp<compsToRemove.size();comp++)
    {
      removeComponent(DL,compsToRemove[comp]-comp);
    }
}

void removeMean(dataList &DL)
{
  // Get the size of things.
  int epochPts = DL.epochArray[0].timePointArray.size();
  int numComps = DL.epochArray[0].timePointArray[0].dataVector.size();
  int numEpochs = DL.epochArray.size();

  std::vector<float> sums(numComps*numEpochs,0);
  const float alpha = 1.0/((float) epochPts);
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      for(int timePoint=0;timePoint<epochPts;timePoint++)
	{
	  cblas_saxpy(numComps,alpha,DL.epochArray[epoch].timePointArray[timePoint].dataVector.data(),1,
		      sums.data()+epoch*numComps,1);
	}
      for(int timePoint=0;timePoint<epochPts;timePoint++)
	{
	  cblas_saxpy(numComps,-1.0,sums.data()+epoch*numComps,1,
		      DL.epochArray[epoch].timePointArray[timePoint].dataVector.data(),1);
	}
    }
  //printf("first entry %f \n",sums[0]);
  
  return;
}
