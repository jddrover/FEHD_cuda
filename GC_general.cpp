#include <cblas.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_float_real(z) z.real()
#define lapack_complex_float_imag(z) z.imag()
#include <lapacke.h>
#include <algorithm>
#include "utility.h"
#include <string>

float GCgen(std::vector<float> dataArray, std::vector<int> caused, paramContainer params)
{
  float GC;

  // First, need to organize so that the caused components are on top.
  
  int numCausedComps = caused.size();
  int numCausalComps = params.numChannels-numCausedComps;

  std::vector<float> dataArraySorted(dataArray);
  int counterCaused=0;
  int counterNot=0;

  for(int comp=0;comp<params.numChannels;comp++)
    {
      if(std::find(caused.begin(),caused.end(),comp) != caused.end())
	{
	  cblas_scopy(params.numPoints,dataArray.data()+comp,params.numChannels,
		      dataArraySorted.data()+counterCaused,params.numChannels);
	  counterCaused++;
	}
      else
	{
	  cblas_scopy(params.numPoints,dataArray.data()+comp,params.numChannels,
		      dataArraySorted.data()+counterNot+numCausedComps,
		      params.numChannels);
	  counterNot++;
	}
    }

  dataArray = dataArraySorted;

  // Construct the frequency array
  std::vector<float> freq(params.numFreqs,0.0);
  for(int findx=0;findx<params.numFreqs;findx++)
    freq[findx] = (params.freqHi-params.freqLo)/(params.numFreqs-1)*float(findx)+params.freqLo;

  // Sort the lag list. This is for when one specifies the lags. This has not been checked.
  std::sort(params.lagList.begin(),params.lagList.end());
  int maxLag = params.lagList[params.numLags-1];
  int IPIV[params.numChannels*params.numLags];
  
  // Construct the AR model
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
	  std::copy(dataArray.begin()+epoch*params.numChannels*params.epochPts+
		    (maxLag-params.lagList[lag])*params.numChannels+tp*params.numChannels,
		    dataArray.begin()+epoch*params.numChannels*params.epochPts+
		    (maxLag-params.lagList[lag])*params.numChannels+(tp+1)*params.numChannels,
		    LHS.begin()+epoch*params.numChannels*params.numLags*(params.epochPts-maxLag)+
		    lag*params.numChannels+tp*params.numChannels*params.numLags);   
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

 
  // Compute the residuals

  cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),params.numChannels*params.numLags,
	      -1.0,RCOV.data(),params.numLags*params.numChannels,
	      LHS.data(),params.numLags*params.numChannels,
	      1.0,RHS.data(),params.numChannels);
	      
  // Compute the covariance of the residuals
  std::vector<float> resCOV(params.numChannels*params.numChannels,0.0);
  cblas_ssyrk(CblasColMajor,CblasLower,CblasNoTrans,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),
	      1.0,RHS.data(),params.numChannels,
	      0.0,resCOV.data(),params.numChannels);

  // I want to keep a copy of this
  std::vector<float> RCC(resCOV);
  
  // P matrix is used to decorrelate the residuals. 
  std::vector<float> P(params.numChannels*params.numChannels,0.0);
  std::vector<float> Pinv(params.numChannels*params.numChannels,0.0);


  //     [ I_caused       0    ]
  // P = [                     ]
  //     [    C       I_causal ]
  for(int diagentry=0;diagentry<params.numChannels;diagentry++)
    P[diagentry*params.numChannels+diagentry]=1.0;

  // The following sequence of commands forms the lower left portion of the matrix T:
  // C=-Rcov[x1,x2]^T x Rcov[x1,x1]^-1
  // The first two compute the inverse,
  // The third does the matmul.
  info = LAPACKE_ssytrf(LAPACK_COL_MAJOR,'L',numCausedComps,resCOV.data(),params.numChannels,IPIV);
  info = LAPACKE_ssytri(LAPACK_COL_MAJOR,'L',numCausedComps,resCOV.data(),params.numChannels,IPIV);
  cblas_ssymm(CblasColMajor,CblasRight,CblasLower,params.numChannels-numCausedComps, numCausedComps,
	      -1.0,resCOV.data(),params.numChannels,
	      resCOV.data()+numCausedComps,params.numChannels,
	      0.0,P.data()+numCausedComps,params.numChannels);
  // Using triangular mm since P is unit triangular
  cblas_strmm(CblasColMajor,CblasRight,CblasLower,CblasTrans,CblasUnit,params.numChannels*params.numLags,params.numChannels,
	      1.0,P.data(),params.numChannels,RCOV.data(),params.numChannels*params.numLags);

  // Before we invert P, we want to Pcov(R)P^T

  cblas_strmm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit,
	      params.numChannels,params.numChannels,
	      1.0,P.data(),params.numChannels,
	      resCOV.data(),params.numChannels);
  cblas_strmm(CblasColMajor,CblasRight,CblasLower,CblasTrans,CblasUnit,
	      params.numChannels,params.numChannels,
	      1.0,P.data(),params.numChannels,
	      resCOV.data(),params.numChannels); // resCOV is fully filled.
  Pinv = P;
  // now I want the inverse of P. P is lower unit triangular matrix.
  info = LAPACKE_strtri(LAPACK_COL_MAJOR,'L','U',params.numChannels,Pinv.data(),params.numChannels);
  // P is now the inverse.
  // Finish the similarity transformation on the AR coefficents
  for(int lag=0;lag<params.numLags;lag++)
    {
      cblas_strmm(CblasColMajor,CblasLeft,CblasLower,CblasTrans,CblasUnit,
		  params.numChannels,params.numChannels,
		  1.0,Pinv.data(),params.numChannels,
		  RCOV.data()+lag*params.numChannels,params.numChannels*params.numLags);
    }

  // Declare the transfer function coeffients, and a complex array to hold AR coefficients
  // (cblas complex routines shit the bed if you give them a float array)
  std::vector<std::complex<float>> Tf(params.numChannels*params.numChannels,
				      std::complex<float>(0.0,0.0));
  std::vector<std::complex<float>> A(params.numLags*params.numChannels*params.numChannels,std::complex<float>(0.0,0.0));

  for(int col=0;col<params.numLags*params.numChannels;col++)
    for(int row=0;row<params.numChannels;row++)
      A[col*params.numChannels+row]=std::complex<float>(RCOV[row*params.numLags*params.numChannels+col],0.0);





  for(int col=1;col<params.numChannels;col++)
    for(int row=0;row<col;row++)
      {
	//std::cout << resCOVcopy[row*params.numChannels+col] << std::endl;
	RCC[col*params.numChannels+row] = RCC[row*params.numChannels+col];
      }

  cblas_strmm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit,
	      params.numChannels,params.numChannels,
	      1.0,P.data(),params.numChannels,
	      RCC.data(),params.numChannels);
  cblas_strmm(CblasColMajor,CblasRight,CblasLower,CblasTrans,CblasUnit,
	      params.numChannels,params.numChannels,
	      1.0,P.data(),params.numChannels,
	      RCC.data(),params.numChannels);

       
  // Now, make a complex copy
  std::vector<std::complex<float>> resCC(params.numChannels*params.numChannels);
  for(int indx=0;indx<params.numChannels*params.numChannels;indx++)
    resCC[indx] = std::complex<float>(RCC[indx],0.0);
  
  float dt = 1.0/((float)params.sampRate);
  float argmtBASE = -2.0*M_PI*dt;
  std::complex<float> argmt;
  std::vector<std::complex<float>> stmp1(params.numChannels*params.numChannels);
  std::vector<std::complex<float>> stmp2(stmp1);
  const std::complex<float> alphaComplex(1.0,0.0);
  const std::complex<float> betaComplex(0.0,0.0);
  std::vector<std::complex<float>> whole(numCausedComps*numCausedComps);
  std::vector<std::complex<float>> partial(whole);
  std::vector<float> wholeEigs(numCausedComps);
  std::vector<float> partialEigs(numCausedComps);
  float wProd,pProd;
  float GCtotal = 0.0;
  for(int f_indx=0;f_indx<params.numFreqs;f_indx++)
    {
      std::fill(Tf.begin(),Tf.end(),std::complex<float>(0.0,0.0));
      for(int comp=0;comp<params.numChannels;comp++)
	Tf[comp*params.numChannels+comp]=std::complex<float>(1.0,0.0);

     
      for(int lag=0;lag<params.numLags;lag++)
	{
	  argmt = -std::exp(std::complex<float>(0.0,argmtBASE*(float)(params.lagList[lag])*freq[f_indx]));
	  cblas_caxpy(params.numChannels*params.numChannels,&argmt,
		      A.data()+lag*params.numChannels*params.numChannels,1,
		      Tf.data(),1);
	}      
      // Invert Tf (this truly is the path of least resistance)
      info = LAPACKE_cgetrf(LAPACK_COL_MAJOR,params.numChannels,params.numChannels,
			    Tf.data(),params.numChannels,IPIV);
      info = LAPACKE_cgetri(LAPACK_COL_MAJOR,params.numChannels,Tf.data(),params.numChannels,IPIV);

      // Compute the whole spectrum
      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		  params.numChannels,params.numChannels,params.numChannels,
		  &alphaComplex,Tf.data(),params.numChannels,
		  resCC.data(),params.numChannels,
		  &betaComplex,stmp1.data(),params.numChannels);
      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasConjTrans,
		  params.numChannels,params.numChannels,params.numChannels,
		  &alphaComplex,stmp1.data(),params.numChannels,
 		  Tf.data(),params.numChannels,
		  &betaComplex,stmp2.data(),params.numChannels);

      info = LAPACKE_cheev(LAPACK_COL_MAJOR,'N','U',numCausedComps,
			   stmp2.data(),params.numChannels,wholeEigs.data());
      
      // Compute the partial spectrum
      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		  numCausedComps,numCausedComps,numCausedComps,
		  &alphaComplex,Tf.data(),params.numChannels,
		  resCC.data(),params.numChannels,
		  &betaComplex,stmp1.data(),params.numChannels);
      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasConjTrans,
		  numCausedComps,numCausedComps,numCausedComps,
		  &alphaComplex,stmp1.data(),params.numChannels,
		  Tf.data(),params.numChannels,
		  &betaComplex,stmp2.data(),params.numChannels);

      info = LAPACKE_cheev(LAPACK_COL_MAJOR,'N','U',numCausedComps,
			   stmp2.data(),params.numChannels,partialEigs.data());

      wProd = 1.0;
      pProd = 1.0;
      
      for(int eig=0;eig<numCausedComps;eig++)
	{
	  wProd = wProd*wholeEigs[eig];
	  pProd = pProd*partialEigs[eig];
	}
      // Trapezoid rule - I don't see the point in anything higher order.
      if(f_indx==0)
	GCtotal = GCtotal + 0.5*std::log(wProd/pProd);
      else if(f_indx==params.numFreqs-1)
	GCtotal = GCtotal + 0.5*std::log(wProd/pProd);
      else
	GCtotal = GCtotal + std::log(wProd/pProd);
    }
  GCtotal = GCtotal*(freq[1]-freq[0]);
	      
  return GCtotal;
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
  std::vector<int> caused = {0};
   
  float GCval;

  GCval = GCgen(dataArray,caused,params);
  std::cout << GCval << std::endl;

  caused = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  
  GCval = GCgen(dataArray,caused,params);

  std::cout << GCval << std::endl;

  return 0;
}

  
