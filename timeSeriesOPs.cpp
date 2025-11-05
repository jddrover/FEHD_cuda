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
  std::cout << "numEpochs = " << numEpochs << "\n" <<
    "epochPts = " << epochPts << "\n" <<
    "numComps = " << numComps << "\n" <<
    "numPoints = " << numPoints << std::endl;
    
  float *dataArray = new float[numEpochs*epochPts*numComps];

  const float alpha=1.0f;
  const float beta=0.0f;

  float *covMat = new float[numComps*numComps];
  
  // Convert the Data to a raw array for the blas and lapack routines.
  
  convertDataListToRawArray(DS,dataArray);

  // Compute the covariance matrix // This is a rather large calculation.
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,numComps,numComps,
	      numPoints,alpha,dataArray,numComps,dataArray,numComps,beta,
	      covMat,numComps);

  // Compute the SVD
  float superb[numComps-1];

  float *sMat = new float[numComps];
  float *uMat = new float[numComps*numComps];
  float *vMatT = new float[numComps*numComps];
  int info;
  std::cout << "going in" << std::endl;
  for(int row=0;row<numComps;row++)
    {
      for(int col=0;col<numComps;col++)
	{
	  std::cout << covMat[col*numComps+row] << " ";
	}
      std::cout << "\n";
    }
  
  info = LAPACKE_sgesvd(LAPACK_COL_MAJOR,'A','A',numComps,numComps,covMat,numComps,sMat,uMat,numComps,
			vMatT,numComps,superb);
  if(info != 0)
    {
      std::cout << "info = " << info << std::endl;
      std::cout << "SVD problem" << std::endl;
      exit(0);
    }
  /*
  for(int comp=0;comp<numComps;comp++)
    {
      std::cout << sMat[comp] << std::endl;
    }
  for(int row=0;row<numComps;row++)
    {
      for(int col=0;col<numComps;col++)
	{
	  std::cout << uMat[col*numComps+row] << " ";
	}
      std::cout << std::endl;
      }*/
  // Now apply U' to the data

  // Set aside some space for the principal components.
  float *PCrawArray = new float[numComps*numPoints];

  // Scale the transformation matrix so that the transformed components are orthonormal.
  float scaleVal;
  
  for(int comp=0;comp<numComps;comp++)
    {
      scaleVal = 1.0f/(sqrt(sMat[comp]));
      // Changed 7/17/23
      //scaleVal = sqrt(float(numPoints))/(sqrt(sMat[comp]));
      cblas_sscal(numComps,scaleVal,uMat+(comp*numComps), 1);
    }


  cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,numComps,numPoints,numComps,
	      alpha, uMat, numComps, dataArray, numComps,beta,PCrawArray,numComps);

  //Convert the PC raw array to the PC data structure.

  //cblas_sscal(numPoints*numComps,sqrt(numPoints), PCrawArray, 1);
  
  convertRawArrayToDataList(PCrawArray,PC,numComps,epochPts,numEpochs);

  transMat.elements.insert(transMat.elements.begin(),uMat,uMat+(numComps*numComps));
  // clean up

  // I want to transpose this before output, I think it's better to have it in transformation form.
  float tempval;
  for(int col=0;col<numComps;col++)
    {
      for(int row=0;row<col;row++)
	{
	  tempval = transMat.elements[col*numComps+row];
	  transMat.elements[col*numComps+row] = transMat.elements[row*numComps+col];
	  transMat.elements[row*numComps+col] = tempval;
	}
    }



	  
  delete [] dataArray;
  delete [] covMat;
  delete [] sMat;
  delete [] uMat;
  delete [] vMatT;
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
