#include <cblas.h>
#include <lapacke.h>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <complex>
//#include <cmath>
#include <algorithm>
#include "utility.h"
#include <numbers>
#include <omp.h>
#include <string>

std::vector<float> PGC(std::vector<float> dataArray, paramContainer params)
{
  // Size of large arrays 
  int numEpochs = params.numEpochs;
  int numComps = params.numChannels;
  std::vector<float> Xout(numComps*numComps,0.0);
  std::vector<float> freq(params.numFreqs);
  for(int findx=0;findx<params.numFreqs;findx++)
    freq[findx] = (params.freqHi-params.freqLo)/(params.numFreqs-1)*float(findx)+params.freqLo;


  
  // Check that this is an integer
  int epochPts = params.epochPts;
  std::vector<int> lagList(params.lagList);
  std::sort(lagList.begin(),lagList.end());
  int numLags = lagList.size();
  int maxLag = lagList[numLags-1];
  std::vector<float> RHS(numComps*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> LHS(numComps*numLags*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> GCatFreq(params.numFreqs,0.0);
  for(int epoch=0;epoch<numEpochs;epoch++)
    {// huh?
      std::copy(dataArray.begin()+(epoch*epochPts+maxLag)*numComps,dataArray.begin()+(epoch+1)*epochPts*numComps,
		RHS.begin()+epoch*(epochPts-maxLag)*numComps);

      for(int lag=0;lag<numLags;lag++)
      {
        for(int tp=0;tp<epochPts-maxLag;tp++)
          {
            std::copy(dataArray.begin()+epoch*numComps*epochPts+(maxLag-lagList[lag])*numComps+tp*numComps,
      		dataArray.begin()+epoch*numComps*epochPts+(maxLag-lagList[lag])*numComps+(tp+1)*numComps,
      		LHS.begin()+epoch*numComps*numLags*(epochPts-maxLag)+lag*numComps+tp*numComps*numLags);
          }
      }
    }
  float dt = 1.0/((float)params.sampRate);
  //std::cout << "step size" << dt << std::endl;
  float argmtBASE = -2.0*M_PI*dt;
  std::complex<float> argmt;
  std::vector<float> LS(2*numLags*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> RS(2*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> RS2(2*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> LCOV(2*numLags*2*numLags,0.0);
  std::vector<float> RCOV(2*numLags*2,0.0);
  std::vector<float> A(4*numLags,0.0);
  std::vector<float> resCOV(4,0.0);
  std::vector<float> Pmat(4,0.0);
  std::vector<float> Pinv(4,0.0);
  std::vector<float> resCOVinv(4,0.0);
  std::vector<std::complex<float>> RI(4,std::complex<float>(0.0,0.0));
  std::vector<std::complex<float>> Tf(4*params.numFreqs,std::complex<float>(0.0,0.0));
  std::complex<float> alphaC(1.0,0.0);
  std::complex<float> betazero(0.0,0.0);
  std::vector<std::complex<float>> Swhole(4*params.numFreqs,std::complex<float>(0.0,0.0));
  std::vector<std::complex<float>> Stmp(4*params.numFreqs,std::complex<float>(0.0,0.0));
  std::complex<float> Spartial;
  float tmp;
  int info;
  int IPIV[2*numLags];
  const float alpha=1.0;
  const float oneoverN = 1.0/((float)(numEpochs*(epochPts-maxLag)));
  float wholeSpec;
  float totalGC;
  std::vector<std::complex<float>> RCOVcomplex(4*numLags,std::complex<float>(0.0,0.0));
  for(int pair1=0;pair1<numComps;pair1++)
    for(int pair2=0;pair2<numComps;pair2++)
      {
	if(pair1==pair2) 
	  continue;
	// Use threads to do these things at the same time.
#pragma omp parallel sections
	{// I might have just seen it rihgt away. Hopefully, Jon
	  #pragma omp section
	  {
	    cblas_scopy(numLags*numEpochs*(epochPts-maxLag),LHS.data()+pair1,numComps,LS.data(),2);
	  }
	  #pragma omp section
	  {
	    cblas_scopy(numLags*numEpochs*(epochPts-maxLag),LHS.data()+pair2,numComps,LS.data()+1,2);
	  }
	  #pragma omp section
	  {
	    cblas_scopy(numEpochs*(epochPts-maxLag),RHS.data()+pair1,numComps,RS.data(),2);
	  }
	  #pragma omp section
	  {
	    cblas_scopy(numEpochs*(epochPts-maxLag),RHS.data()+pair2,numComps,RS.data()+1,2);
	  }
	}

	// Check section
	// I want to make sure that data has been transferred as I expect
	//
	//std::cout << "Marker 1" << std::endl;
	//std::cout << RS[0] << " " << RS[1] << std:: endl;

	
#pragma omp parallel sections
	{
	  #pragma omp section
	  {
	    cblas_ssyrk(CblasColMajor,CblasUpper,CblasNoTrans,2*numLags,numEpochs*(epochPts-maxLag),
		    alpha,LS.data(),2*numLags,0.0,LCOV.data(),2*numLags);
	  }
	  #pragma omp section
	  {
	    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2*numLags,2,numEpochs*(epochPts-maxLag),
			alpha,LS.data(),2*numLags,RS.data(),2,0.0,RCOV.data(),2*numLags);
	  }
	}


	
	info = LAPACKE_ssysv(LAPACK_COL_MAJOR,'U',2*numLags,2,LCOV.data(),2*numLags,IPIV,
			     RCOV.data(),2*numLags);


	/*for(int row=0;row<4;row++)
	  {
	    for(int col=0;col<2;col++)
	      {
		std::cout << RCOV[col*4+row] << " ";
	      }
	    std::cout << std::endl;
	    }*/
		  
	
	// Compute the residuals
	// RHS-A*LHS
	// LONG
	cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,2,numEpochs*(epochPts-maxLag),2*numLags,
		    -1.0,RCOV.data(),2*numLags,LS.data(),2*numLags,1.0,RS.data(),2);


	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2,2,numEpochs*(epochPts-maxLag),
		    oneoverN,RS.data(),2,RS.data(),2,0.0,resCOV.data(),2);

	//std::cout << "rcov" << std::endl;
	//std::cout << resCOV[0] << " " << resCOV[2] << std::endl;
	//std::cout << resCOV[1] << " " << resCOV[3] << std::endl;
	//std::cout << std::endl;
	Pmat[0]=1.0;
	Pmat[1]=-resCOV[2]/resCOV[0];
	Pmat[2]=0.0;
	Pmat[3]=1.0;
	
	Pinv[0]=1.0;
	Pinv[1]=-Pmat[1];
	Pinv[2]=0.0;
	Pinv[3]=Pmat[3];

	//std::cout << "P matrices" << std::endl;
	//std::cout << Pmat[0] << " " << Pmat[2] << std::endl;
	//std::cout << Pmat[1] << " " << Pmat[3] << std::endl;
	//std::cout << "P inverse" << std::endl;
	//std::cout << Pinv[0] << " " << Pinv[2] << std::endl;
	//std::cout << Pinv[1] << " " << Pinv[3] << std::endl;
	  
	// still use RCOV
	//cblas_ssymm(CblasColMajor,CblasRight,CblasUpper,2,numLags*2,
	//	    1.0,RCOV.data(),2*numLags,Pmat.data(),2,
	//	    0.0,A.data(),2);
	//std::cout << A[0] << " " << A[2] << std::endl;
	//std::cout << A[1] << " " << A[3] << std::endl;
	//std::cout << std::endl;
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2,2*numLags,2,
		    1.0,Pmat.data(),2,RCOV.data(),2*numLags,0.0,A.data(),2);
	//std::cout << "A's 1" << std::endl;
	//std::cout << A[0] << " " << A[2] << " " << A[4] << " " << A[6] << std::endl;
	//std::cout << A[1] << " " << A[3] << " " << A[5] << " " << A[7] << std::endl;
	for(int lag=0;lag<numLags;lag++)
	  {
	    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2,2,
			1.0,A.data()+lag*4,2,Pinv.data(),2,0.0,RCOV.data()+lag*4,2);
	  }


	// Check these A's
	


	
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2,2,
		    1.0,Pmat.data(),2,resCOV.data(),2,
		    0.0,resCOVinv.data(),2);
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2,2,2,
		    1.0,resCOVinv.data(),2,Pmat.data(),2,
		    0.0,resCOV.data(),2);

	//std::cout << "rcov2" << std::endl;
	//std::cout << resCOV[0] << " " << resCOV[2] << std::endl;
	//std::cout << resCOV[1] << " " << resCOV[3] << std::endl;
	//std::cout << std::endl;
	tmp = resCOV[0]*resCOV[3]-resCOV[1]*resCOV[2];
	//std::cout << tmp << std::endl;
	resCOVinv[0] = (resCOV[3]/tmp)*(oneoverN);
	resCOVinv[1] = 0.0;
	resCOVinv[2] = 0.0;
	resCOVinv[3] = resCOV[0]/tmp*oneoverN;
	//std::cout << "rcov2" << std::endl;
	//std::cout << resCOVinv[0] << " " << resCOVinv[2] << std::endl;
	//std::cout << resCOVinv[1] << " " << resCOVinv[3] << std::endl;
	//std::cout << std::endl;
	RI[0] = std::complex<float>(resCOVinv[0],0.0);
	RI[1] = std::complex<float>(resCOVinv[1],0.0);
	RI[2] = std::complex<float>(resCOVinv[2],0.0);
	RI[3] = std::complex<float>(resCOVinv[3],0.0);

	
	for(int indx=0;indx<RCOV.size();indx++)
	  RCOVcomplex[indx] = std::complex<float>(RCOV[indx],0.0);
	
	std::fill(Tf.begin(),Tf.end(),std::complex<float>(0.0,0.0));

	for(int findx=0;findx<params.numFreqs;findx++)
	  {
	      
	    // Compute the "transfer function" (inverse) 
	    Tf[findx*4]=std::complex<float>(1.0,0.0);
	    Tf[findx*4+3]=std::complex<float>(1.0,0.0);
	    //std::cout << "A's" << std::endl;
	    //std::cout << RCOV[0] << " " << RCOV[2] << " " << RCOV[4] << " " << RCOV[6] << std::endl;
	    //std::cout << RCOV[1] << " " << RCOV[3] << " " << RCOV[5] << " " << RCOV[7] << std::endl;
	    for(int lag=0;lag<params.numLags;lag++)
	      {
		argmt = -std::exp(std::complex<float>(0.0,argmtBASE*(float)(lagList[lag])*freq[findx]));
		//std::cout << argmt << std::endl;
		cblas_caxpy(4,&argmt,RCOVcomplex.data()+lag*4,1,Tf.data()+findx*4,1);
		//std::cout << "Transfer " << findx << std::endl;
		//std::cout << Tf[findx*4] << " " << Tf[findx*4+2] << std::endl;
		//std::cout << Tf[findx*4+1] << " " << Tf[findx*4+3] << std::endl;
	      }
	    
	    // S^-1=TF^-* E^-1 TF^-1
	    // (part 1 of 2)
	    cblas_cgemm(CblasColMajor,CblasConjTrans,CblasNoTrans,2,2,2,
			&alphaC,Tf.data()+findx*4,2,RI.data(),2,
			&betazero,Stmp.data()+findx*4,2);
	    // (part 2 of 2)
	    cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2,2,
			&alphaC,Stmp.data()+findx*4,2,Tf.data()+4*findx,2,
			&betazero,Swhole.data()+4*findx,2);
	    
	    // The "whole" spectrum submatrix (which is a scalar) (inverse) 
	    wholeSpec = (Swhole[findx*4]-Swhole[findx*4+2]*Swhole[findx*4+1]/Swhole[findx*4+3]).real();
	    
	    // The "partial" spectrum (also a scalar) (inverse)
	    Spartial = Tf[findx*4]-Tf[findx*4+2]*Tf[findx*4+1]/Tf[findx*4+3];
	    Spartial = std::conj(Spartial)*RI[0]*Spartial;
	    
	    // Record the GC at this frequency.
	    GCatFreq[findx] = std::log(Spartial.real()/wholeSpec);
	    //std::cout << GCatFreq[findx] << std::endl;
	  }
	//std::cout << "freq" << std::endl;
	// Integration using trapezoids. 
	totalGC = 0.0;
	for(int findx=0;findx<params.numFreqs;findx++)
	  {
	    if(findx == 0)
	      totalGC = totalGC + 0.5*GCatFreq[findx];
	    else if(findx == params.numFreqs-1)
	      totalGC = totalGC + 0.5*GCatFreq[findx];
	    else
	      totalGC = totalGC + GCatFreq[findx];
	  }
	
	Xout[pair1+pair2*numComps] = totalGC*(freq[1]-freq[0]);
		
      }
  return Xout;
}

void loadData(std::string filename,int numComps,int numEpochs,int epochPts,std::vector<float> &dataArray)
{
  std::ifstream dStream(filename.c_str(),std::ifstream::in);

  for(unsigned long int point=0;point<numEpochs*epochPts*numComps;point++)
    dStream >> dataArray[point];
  
  return;
}


int main()
{
  int numComps = 16;
  int numEpochs = 39;
  int epochPts = 600;
  int numLags = 30;
  int N = numEpochs*epochPts;
  std::random_device gen;
  std::default_random_engine generator(gen());
  std::normal_distribution<float> normDist(0.0,1.0);
  std::vector<float> dataArray(numComps*N,0.0);
 // std::vector<float> dataArray(numComps*N,0.0);
  std::string filename ("IN301M_v1_HD.dat");
  loadData(filename,16,39,600,dataArray);
  //std::vector<float> da2(3*N,0.0);
  //for(int row=0;row<3;row++)
  //  {
  //    for(int col=0;col<N;col++)
  //	da2[col*2+row]=dataArray[col*16+row];
  //  }

  //  std::cout << da2[10*2+1] << " " << dataArray[10*16+1] << std::endl;

  //dataArray.resize(3*N);

  
  //dataArray = da2;
  //  dataArray[indx] = normDist(generator);
    
  // Create a dummy data array
  // Create parameter structure.
  paramContainer params;
  params.epochPts=epochPts;
  params.numLags =numLags;
  std::vector<int> lagList(numLags,0);
  for(int lag=0;lag<numLags;lag++)
    lagList[lag] = lag+1;
  params.numChannels = numComps;
  params.lagList = lagList;
  params.numEpochs = numEpochs;
  params.numFreqs = 13;
  params.freqLo = 8.0;
  params.freqHi = 12.0;
  params.sampRate = 200;
  std::vector<float> X(numComps*numComps,0.0);



  
  X = PGC(dataArray,params);
  for(int row=0;row<numComps;row++)
    {
      for(int col=0;col<numComps;col++)
	std::cout << X[col*numComps+row] << " ";
      std::cout << std::endl;
    }
  return 0;
}

  
 
