#include <cblas.h>
#include <lapacke.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <complex>
#include <algorithm>
#include "utility.h"
#include <string>

struct GCgrids
{
  std::vector<float> Xgrid;
  std::vector<std::vector<float>> intGrid;
};

void transform_matrices(std::vector<float> &ARm,std::vector<float> &covMat)
{

  int M = covMat.size(); // M is going to be 4.
  int L = ARm.size()/M;
  M = sqrt(M);


  // Construct Pmat
  std::vector<float> Pmat(4,0.0);
  std::vector<float> Pinv(4,0.0);
  Pmat[0]=1.0;
  Pmat[1]=-covMat[2]/covMat[0];
  Pmat[2]=0.0;
  Pmat[3]=1.0;
	// The inverse
  Pinv[0]=1.0;
  Pinv[1]=-Pmat[1];
  Pinv[2]=0.0;
  Pinv[3]=Pmat[3];
  
  // Transform the AR coefficients

  std::vector<float> Atmp(ARm);
  std::vector<float> COVtmp(covMat);
  
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,M,M*L,M,
	      1.0,Pmat.data(),M,ARm.data(),M*L,0.0,Atmp.data(),M);
	
  for(int lag=0;lag<L;lag++)
    {
      cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,M,M,
		  1.0,Atmp.data()+lag*M*M,M,Pinv.data(),M,0.0,ARm.data()+lag*M*M,M);
    }

  // Determine the new covariance matrix
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,M,M,
	      1.0,Pmat.data(),M,covMat.data(),M,
	      0.0,COVtmp.data(),M);
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,M,M,M,
	      1.0,COVtmp.data(),M,Pmat.data(),M,
	      0.0,covMat.data(),M);

  return;
}


GCgrids PGC(std::vector<float> dataArray, paramContainer params)
{
  // Size of large arrays 
  int numEpochs = params.numEpochs;
  int numComps = params.numChannels;
  int epochPts = params.epochPts;

  // Sort and copy the lag list
  std::vector<int> lagList(params.lagList);
  std::sort(lagList.begin(),lagList.end());
  int numLags = lagList.size();
  int maxLag = lagList[numLags-1];
  float dt = 1.0/((float)params.sampRate);

  // The output (either a heatmap or the integrands - gridout carries both).
  std::vector<float> Xout(numComps*numComps,0.0);
  std::vector<std::vector<float>> integrand(numComps*numComps);
  GCgrids gridout;

  // Make the frequency grid.
  std::vector<float> freq(params.numFreqs);
  for(int findx=0;findx<params.numFreqs;findx++)
    freq[findx] = (params.freqHi-params.freqLo)/(params.numFreqs-1)*float(findx)+params.freqLo;
  
  // Declarations of large arrays to hold the left and right Yule Walker coefficients.  
  std::vector<float> RHS(numComps*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> LHS(numComps*numLags*numEpochs*(epochPts-maxLag),0.0);
  // Hold the 2-d subsets.
  std::vector<float> LS(2*numLags*numEpochs*(epochPts-maxLag),0.0);
  std::vector<float> RS(2*numEpochs*(epochPts-maxLag),0.0);
  // Hold the covariances.
  std::vector<float> LCOV(2*numLags*2*numLags,0.0);
  std::vector<float> RCOV(2*numLags*2,0.0);
  
  // Storage for the Granger Causalities at each frequency.
  std::vector<float> GCatFreq(params.numFreqs,0.0);

  // Set up lagged version.
#pragma omp parallel for default(shared) 
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
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
  
  //std::cout << "step size" << dt << std::endl;
  float argmtBASE = -2.0*M_PI*dt;
  std::complex<float> argmt;



  std::vector<float> resCOV(4,0.0);


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
  std::vector<float> RCOVflip(4*numLags,0.0);
  std::vector<float> resCOVflip(4,0.0);
  std::vector<std::complex<float>> ARcoeff(4*numLags,std::complex<float>(0.0,0.0));
  for(int pair1=0;pair1<numComps;pair1++)
    for(int pair2=0;pair2<numComps;pair2++)
      {
	if(pair1==pair2)
	  {
	    integrand[pair1+pair2*numComps]=std::vector<float>(params.numFreqs,0.0);	    
	    continue;
	  }

	// Organize the data
	// Takes the two components in play and copies them to an array.
	// It does this for both the left and right side matrices.
	
#pragma omp parallel sections
	{
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


	// Multiplies to create the left and right LS matrices
	// The left will be symmetric, so the appropriate multiply routine is used.
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
	// The symmetry definitely saves time below (the solver is faster),
	// but it probably isn't helpful above since we have to wait for
	// the gemm to move on. Doesn't hurt either.
	
	// Solve the linear systems AX=B, where A is symmetric.
	info = LAPACKE_ssysv(LAPACK_COL_MAJOR,'U',2*numLags,2,LCOV.data(),2*numLags,IPIV,
			     RCOV.data(),2*numLags);

	
	// Compute the residuals
	// RHS-A*LHS
	// LONG
	cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,2,numEpochs*(epochPts-maxLag),2*numLags,
		    -1.0,RCOV.data(),2*numLags,LS.data(),2*numLags,1.0,RS.data(),2);

	// And compute the covariance matrix.
	// Could halve the effort here, this is a rank update.
	// 
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2,2,numEpochs*(epochPts-maxLag),
		    oneoverN,RS.data(),2,RS.data(),2,0.0,resCOV.data(),2);

	// The information up to this point can be used twice, by flipping it upside down.

	/*
	  RCOVflip = RCOV;
	  cblas_sswap(numLags*numComps,RCOVflip.data(),1,RCOVflip.data()+numLags*2,1);
	  resCOVflip = resCOV;
	  tmp = resCOVflip[0];
	  resCOVflip[0] = resCOVflip[3];
	  resCOVflip[3] = tmp;
	*/
	
	// PAP^-1 and PCP^T. Called by reference, so both arguments change.
	transform_matrices(RCOV,resCOV);

	// Explicitly compute the inverse of the covariance matrix.
	tmp = resCOV[0]*resCOV[3]-resCOV[1]*resCOV[2];
	/*
	resCOVinv[0] = (resCOV[3]/tmp)*(oneoverN);
	resCOVinv[1] = 0.0;
	resCOVinv[2] = 0.0;
	resCOVinv[3] = resCOV[0]/tmp*oneoverN;
	*/	
	// Make a complex version.
	RI[0] = std::complex<float>(resCOV[3]/tmp*oneoverN,0.0);
	RI[1] = std::complex<float>(0.0,0.0);
	RI[2] = std::complex<float>(0.0,0.0);
	RI[3] = std::complex<float>(resCOV[0]/tmp*oneoverN,0.0);


	// ARcoeffs
	for(int indx=0;indx<RCOV.size();indx++)
	  ARcoeff[indx] = std::complex<float>(RCOV[indx],0.0);

	
	std::fill(Tf.begin(),Tf.end(),std::complex<float>(0.0,0.0));

#pragma omp parallel default(shared) private(argmt,wholeSpec,Spartial)
	{
#pragma omp for
	  for(int findx=0;findx<params.numFreqs;findx++)
	  {	      
	    // Compute the "transfer function" (inverse) 
	    Tf[findx*4]=std::complex<float>(1.0,0.0);
	    Tf[findx*4+3]=std::complex<float>(1.0,0.0);
	    
	    for(int lag=0;lag<params.numLags;lag++)
	      {
		argmt = -std::exp(std::complex<float>(0.0,argmtBASE*(float)(lagList[lag])*freq[findx]));
		cblas_caxpy(4,&argmt,ARcoeff.data()+lag*4,1,Tf.data()+findx*4,1);
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
	integrand[pair1+pair2*numComps] = GCatFreq;
	Xout[pair1+pair2*numComps] = totalGC*(freq[1]-freq[0]);
	
      }
  gridout.Xgrid = Xout;
  gridout.intGrid = integrand;
  return gridout;
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
  params.outputFLAG=0;
  FILE *f;
  for(int i=1;i<argc;i+=2)
    {
      if(std::string(argv[i]) == "--output")
	{
	  params.outputType = std::string(argv[i+1]);

	  if(params.outputType!="heatmap" && params.outputType!="integrands")
	    {
	      throw std::invalid_argument("output type not recognized. Exiting");
	      return;
	    }
	  params.outputFLAG=1;
	}
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
  if(params.outputFLAG == 0)
    throw std::invalid_argument("No output option chosen. Exiting");
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
 
  std::vector<float> X(params.numChannels*params.numChannels,0.0);
   
  GCgrids op;
  op = PGC(dataArray,params);


  if(params.outputType=="heatmap")
    {
      for(int row=0;row<params.numChannels;row++)
	{
	  for(int col=0;col<params.numChannels;col++)
	    std::cout << op.Xgrid[col*params.numChannels+row] << " ";
	  std::cout << "\n";
	}
      std::cout << std::endl;
    }
  if(params.outputType=="integrands")
    {
      for(int row=0;row<params.numChannels;row++)
      	{
      	  for(int col=0;col<params.numChannels;col++)
	    {
	      for(int fp=0;fp<params.numFreqs;fp++)
		std::cout << op.intGrid[col*params.numChannels+row][fp] << " ";
	      std::cout << "\n";
	    }
	}
      std::cout << std::endl;
      }
  return 0;
}

  
 
