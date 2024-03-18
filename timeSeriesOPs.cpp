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
    }
  dStream.close();
  dataSet.numEpochs=numEpochs;
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
  
  LAPACKE_sgesvd(LAPACK_COL_MAJOR,'A','A',numComps,numComps,covMat,numComps,sMat,uMat,numComps,
		 vMatT,numComps,superb);

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
/*
void mkAR(dataList dataArray, std::vector<int> lagList, ARmodel &A, dataList &R)
{
  int numEpochs = dataArray.epochArray.size();
  int epochPts = dataArray.epochArray[0].timePointArray.size();
  int numComps = dataArray.epochArray[0].timePointArray[0].dataVector.size();
  int numPoints = numEpochs*epochPts;
  
  matrix LHS;
  matrix RHS;

  // We want maxLag, not numLags
  
  int maxLag = *std::max_element(lagList.begin(),lagList.end());
  int tpoint;
  // Sort the lag list in reverse order, so everything matches up without changing a lot.

  std::sort(lagList.begin(),lagList.end(), std::greater<int>());
  
  for(int epoch=0;epoch<numEpochs;epoch++)
      {
      for(int tp=maxLag;tp<epochPts;tp++)
	{
	  RHS.elements.insert(RHS.elements.end(),
			      dataArray.epochArray[epoch].timePointArray[tp].dataVector.begin(),
			      dataArray.epochArray[epoch].timePointArray[tp].dataVector.end());
	}

      for(int tp=0;tp<epochPts-maxLag;tp++)
	{
	  for(int lagIndx=0;lagIndx<lagList.size();lagIndx++)
	    {
	      tpoint = tp + maxLag-lagList[lagIndx];
	      LHS.elements.insert(LHS.elements.end(),
				  dataArray.epochArray[epoch].timePointArray[tpoint].dataVector.begin(),
				  dataArray.epochArray[epoch].timePointArray[tpoint].dataVector.end());
	    }
	}

      }

  // Make some copies (probably some of them are unnecessary.
  std::vector<float> LHSvec(LHS.elements);
  std::vector<float> RHSvec(RHS.elements);
  std::vector<float> LHSvecBAK(LHS.elements);
  std::vector<float> RHSvecBAK(RHS.elements);

  int numLags = lagList.size();
  
  int mval = numLags*numComps;
  int nval = numLags*numComps;
  int kval = (epochPts-maxLag)*numEpochs;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  float *LHScov = new float[numComps*numLags*numComps*numLags];
  float *RHScov = new float[numComps*numLags*numComps];

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
	      mval, nval, kval, alpha, LHSvec.data(), mval, LHSvec.data(), mval, beta, LHScov, mval);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
	      mval, numComps, kval, alpha, LHSvec.data(), mval, RHSvec.data(), numComps, beta, RHScov, mval);

  // Solve LHScov*A=RHScov for A

  int info;
  int IPIV[numComps*numLags];
  
  info=LAPACKE_ssysv(LAPACK_COL_MAJOR, 'U', numComps*numLags, numComps, LHScov, numComps*numLags,IPIV,
		     RHScov,numComps*numLags);

  matrix lagTmp;

  for(int lag=numLags-1;lag>=0;lag--)
    {
      for(int col=0;col<numComps;col++)
	for(int row=0;row<numComps;row++)
	  {
	    // Transpose and copy.
	    lagTmp.elements.push_back(RHScov[row*numLags*numComps+col+lag*numComps]);
	  }

      A.lagMatrices.push_back(lagTmp);
      lagTmp.elements.clear();
    }

  // Compute the residuals.
  float *modelvals = new float[numComps*(epochPts-numLags)*numEpochs];
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
	      numComps,kval,mval,alpha,RHScov,mval,LHSvecBAK.data(),mval,beta,modelvals,numComps);
  const float alphaneg = -1.0f;
  int mval2 = RHS.elements.size();
  cblas_saxpy(mval2, alphaneg, modelvals, 1, RHSvecBAK.data(), 1);
  convertRawArrayToDataList(RHSvecBAK.data(), R, numComps, epochPts-maxLag, numEpochs);
  
  delete [] modelvals;
  delete [] RHScov;
  delete [] LHScov;
  
  
  
  return;
}
*/





float computeDeterminant(std::vector<float> A, bool lessOne)
{
  int Mval = std::sqrt(A.size());
  int lda = Mval;
  if(lessOne)
    {
      Mval = Mval-1;
    }
  int info;
  info = LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', Mval, A.data(), lda);
  //printf("info = %i \n", info);
  float determinant = 1.0f;

  for(int row=0;row<Mval;row++)
    {
      determinant *= A[row*lda+row];
      //printf("det = %e \n",determinant);
    }

  determinant = determinant*determinant;
  return determinant;
}

void generateNextLagList(std::vector<int> &lagList)
{

  bool unique = false;
  int lastVal;
  while(!unique)
    {
      lastVal = lagList.back();
      lagList.pop_back();
      lagList.push_back(lastVal+1);
      
      std::vector<int> tmp(lagList);
      std::sort(tmp.begin(),tmp.end());
      std::vector<int>::iterator it;
      it = std::unique(tmp.begin(),tmp.end());
      tmp.resize(std::distance(tmp.begin(),it));
      //printf("lagList size = %li, tmp size = %li \n",lagList.size(),tmp.size());
      if(tmp.size()==lagList.size())
	unique = true;
      
    }
  
  return;
}
