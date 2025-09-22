#include "utility.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_float_real(z) z.real()
#define lapack_complex_float_imag(z) z.imag()
#include <cblas.h>
#include <lapacke.h>


void mkAR(std::vector<float> dataArray, std::vector<float> &A, std::vector<float> &R, paramContainer params)
{

  // Laglist sort and determination of largest lag value
  std::sort(params.lagList.begin(),params.lagList.end());
  int maxLag = params.lagList.back();

  // Hold the lagged copies of the data on the L and R sides of the (over-determined)
  // linear system to be solved.
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
  
  // We take LHS*LHS^t and LHS*RHS^t. These form something indistinguishable from
  // the Yule-Walker equations. 
  std::vector<float> LCOV(params.numLags*params.numChannels*params.numLags*params.numChannels,0.0);
  std::vector<float> RCOV(params.numChannels*params.numLags*params.numChannels,0.0);

  // LAPACK utility parameters
  int info;
  int IPIV[params.numChannels*params.numLags];
  
  cblas_ssyrk(CblasColMajor,CblasUpper,CblasNoTrans,params.numChannels*params.numLags,
	      params.numEpochs*(params.epochPts-maxLag),
	      1.0,LHS.data(),params.numLags*params.numChannels,
	      0.0,LCOV.data(),params.numLags*params.numChannels);
  
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,params.numChannels*params.numLags,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),
	      1.0,LHS.data(),params.numLags*params.numChannels,
	      RHS.data(),params.numChannels,
	      0.0,RCOV.data(),params.numChannels*params.numLags);

  // This is the automated routine - it determines the appropriate work array itself.
  // This probably slows it down.
  info = LAPACKE_ssysv(LAPACK_COL_MAJOR,'U',params.numChannels*params.numLags,params.numChannels,
		       LCOV.data(),params.numChannels*params.numLags,IPIV,
		       RCOV.data(),params.numChannels*params.numLags);

  // Compute the residuals. 
  cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,
	      params.numChannels,params.numEpochs*(params.epochPts-maxLag),params.numChannels*params.numLags,
	      -1.0,RCOV.data(),params.numLags*params.numChannels,
	      LHS.data(),params.numLags*params.numChannels,
	      1.0,RHS.data(),params.numChannels);
  
  // Put things into the called arrays.
  for(int col=0;col<params.numChannels*params.numLags;col++)
    for(int row=0;row<params.numChannels;row++)
      A[col*params.numChannels+row] = RCOV[row*params.numChannels*params.numLags+col];
  
  
  R = RHS;
  
  return;
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
  params.numCausedFLAG=0;
  params.numCausalFLAG=0;
  params.outputFLAG=1;
  params.outputType = "integral";
  
  FILE *f;
  for(int i=1;i<argc;i+=2)
    {

      if(std::string(argv[i]) == "--output")
	{
	  if(std::string(argv[i+1]) == "byfrequency")
	    params.outputType = "byfrequency";
	}
	    
      if(std::string(argv[i]) == "--causedList")
	{
	  std::string listofvals = std::string(argv[i+1]);
	  char c = ',';
	  params.numCaused = std::count(listofvals.begin(),listofvals.end(),c)+1;
	  
	  for(int it=0;it<params.numCaused-1;it++)
	    {
	      std::size_t pos = listofvals.find(",");
	      params.causedComps.push_back(std::stoi(listofvals.substr(0,pos)));
	      listofvals = listofvals.substr(pos+1);
	    }
	  params.causedComps.push_back(std::stoi(listofvals));
	  
	  params.numCausedFLAG = 1;
	}
      if(std::string(argv[i]) == "--causalList")
	{
	  std::string listofvals = std::string(argv[i+1]);
	  char c = ',';
	  
	  params.numCausal = std::count(listofvals.begin(),listofvals.end(),c)+1;
	  
	  for(int it=0;it<params.numCausal-1;it++)
	    {
	      std::size_t pos = listofvals.find(",");
	      params.causalComps.push_back(std::stoi(listofvals.substr(0,pos)));
	      listofvals = listofvals.substr(pos+1);
	    }
	  params.causalComps.push_back(std::stoi(listofvals));
	  params.numCausalFLAG=1;
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
  
  if(params.numCausedFLAG == 0)
    throw std::invalid_argument("No list of caused components (--causedList)");
  if(params.numCausalFLAG == 0)
    throw std::invalid_argument("No list of causal components (--causedList)");
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
  loadData(params.filename,params.numChannels,params.numEpochs,params.epochPts,dataArray);
  int numUsedComps = params.numCaused + params.numCausal;
  // Create data sets for these
  std::vector<float> sortedData(numUsedComps*params.numPoints,0.0);
  
  // Fill these

  for(int compIndx=0;compIndx<params.numCaused;compIndx++)
    {
      cblas_scopy(params.numPoints,dataArray.data()+params.causedComps[compIndx],params.numChannels,
		  sortedData.data()+compIndx,numUsedComps);
    }

  for(int compIndx=0;compIndx<params.numCausal;compIndx++)
    {
      cblas_scopy(params.numPoints,dataArray.data()+params.causalComps[compIndx],params.numChannels,
		  sortedData.data()+compIndx+params.numCaused,numUsedComps);
    }
  // Fill the lag list if it has not been filled already and determine the maximum lag.
  if(params.lagListFLAG==0)
    for(int lag=0;lag<params.numLags;lag++)
      params.lagList.push_back(lag+1);
  
  std::sort(params.lagList.begin(),params.lagList.end());
  int maxLag = params.lagList.back();

  // Generate the AR models.
  // Structures to hold the AR coefficents (A), and the residuals (R)
  int Rtotalpoint = params.numEpochs*(params.epochPts-maxLag);
  std::vector<float> A(numUsedComps*numUsedComps*params.numLags,0.0);
  std::vector<float> R(numUsedComps*Rtotalpoint,0.0);
  
  params.numChannels = params.numCaused + params.numCausal;
  mkAR(sortedData,A,R,params);
  
  std::vector<float> COV(numUsedComps*numUsedComps,0.0);

  
  // Compute the covariance of the residuals

  cblas_ssyrk(CblasColMajor,CblasLower,CblasNoTrans,numUsedComps,Rtotalpoint,
	      1.0,R.data(),numUsedComps,
	      0.0,COV.data(),numUsedComps);
  /*
  for(int row=0;row<numUsedComps;row++)
    {
      for(int col=0;col<numUsedComps;col++)
	{
	  std::cout << COV[col*numUsedComps+row] << " ";
	}
      std::cout << std::endl;
    }
  std::cout << std::endl;
  */
  
    // P matrix is used to decorrelate the residuals. 
  std::vector<float> P(numUsedComps*numUsedComps,0.0);
  std::vector<float> Pinv(numUsedComps*numUsedComps,0.0);


  //     [ I_caused       0    ]
  // P = [                     ]
  //     [    C       I_causal ]
  for(int diagentry=0;diagentry<numUsedComps;diagentry++)
    P[diagentry*numUsedComps+diagentry]=1.0;

  // The following sequence of commands forms the lower left portion of the matrix T:
  // C=-Rcov[x1,x2]^T x Rcov[x1,x1]^-1
  // The first two compute the inverse,
  // The third does the matmul.
  int info;
  int IPIV[params.numCaused];
  std::vector<float> COVbak(COV);
  info = LAPACKE_ssytrf(LAPACK_COL_MAJOR,'L',params.numCaused,COV.data(),numUsedComps,IPIV);
  info = LAPACKE_ssytri(LAPACK_COL_MAJOR,'L',params.numCaused,COV.data(),numUsedComps,IPIV);
  /*
  for(int row=0;row<numUsedComps;row++)
    {
      for(int col=0;col<numUsedComps;col++)
	{
	  std::cout << COV[col*numUsedComps+row] << " ";
	}
      std::cout << std::endl;
    }
  std::cout << std::endl;
  */  
  // This step is not straightforward, it depends heaviliy on only using populated blocks.
  cblas_ssymm(CblasColMajor,CblasRight,CblasLower,params.numCausal, params.numCaused,
	      -1.0,COV.data(),numUsedComps,
	      COV.data()+params.numCaused,numUsedComps,
	      0.0,P.data()+params.numCaused,numUsedComps);

  // Using triangular mm since P is unit triangular
  cblas_strmm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit,
	      numUsedComps,numUsedComps*params.numLags,
	      1.0,P.data(),numUsedComps,
	      A.data(),numUsedComps);


    // Before we invert P, we want to Pcov(R)P^T

  // Fill out COV
  for(int col=0;col<numUsedComps;col++)
    for(int row=0;row<col;row++)
      COVbak[col*numUsedComps+row]=COVbak[row*numUsedComps+col];

      // Before we invert P, we want to Pcov(R)P^T

  cblas_strmm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit,
	      numUsedComps,numUsedComps,
	      1.0,P.data(),numUsedComps,
	      COVbak.data(),numUsedComps);
  
  cblas_strmm(CblasColMajor,CblasRight,CblasLower,CblasTrans,CblasUnit,
	      numUsedComps,numUsedComps,
	      1.0,P.data(),numUsedComps,
	      COVbak.data(),numUsedComps);

  Pinv = P;
  // P is a unit-lower-diagonal matrix, the inverse need not be computed, it can just be stated.
  for(int col=0;col<params.numCaused;col++)
    {
      for(int row=params.numCaused;row<numUsedComps;row++)
	{
	  Pinv[col*numUsedComps+row]=
	    -Pinv[col*numUsedComps+row];
	}
    }

  for(int lag=0;lag<params.numLags;lag++)
    {
      cblas_strmm(CblasColMajor,CblasRight,CblasLower,CblasNoTrans,CblasUnit,
		  numUsedComps,numUsedComps,
		  1.0,Pinv.data(),numUsedComps,
		  A.data()+lag*numUsedComps*numUsedComps,
		  numUsedComps);
    }

  
  // Create a copy of the AR coefficients in a complex vector, blas won't mix.
  std::vector<std::complex<float>> Acmplx(params.numLags*numUsedComps*numUsedComps,std::complex<float>(0.0,0.0));
  for(int col=0;col<params.numLags*numUsedComps;col++)
    for(int row=0;row<numUsedComps;row++)
      Acmplx[col*numUsedComps+row]=std::complex<float>(A[col*numUsedComps+row],0.0);

  // We are also going to need to a complex-compatible copy of the residual covariance matrix
  std::vector<std::complex<float>> COVcmplx(numUsedComps*numUsedComps,std::complex<float>(0.0,0.0));
  for(int col=0;col<numUsedComps;col++)
    for(int row=0;row<numUsedComps;row++)
      COVcmplx[col*numUsedComps+row]=std::complex<float>(COVbak[col*numUsedComps+row],0.0);

  // Something to hold the "transfer" matrix.
  std::vector<std::complex<float>> Tf(numUsedComps*numUsedComps,std::complex<float>(0.0,0.0));

  float dt = 1.0/((float)params.sampRate);
  float argmtBASE = -2.0*M_PI*dt;
  std::complex<float> argmt;
  // Create something to hold the frequencies.
  std::vector<float> freq(params.numFreqs,0.0);
  for(int findx=0;findx<params.numFreqs;findx++)
    freq[findx] = (params.freqHi-params.freqLo)/(params.numFreqs-1)*float(findx)+params.freqLo;

  std::complex<float> alphaComplex(1.0,0.0);
  std::complex<float> betaComplex(0.0,0.0);

  // Some work matrices for the loop
  std::vector<std::complex<float>> tmp1(numUsedComps*numUsedComps,std::complex<float>(0.0,0.0));
  std::vector<std::complex<float>> tmp2(numUsedComps*numUsedComps,std::complex<float>(0.0,0.0));

  std::vector<float> wholeEigs(params.numCaused,0.0);
  std::vector<float> partialEigs(params.numCaused,0.0);

  float DETwhole;
  float DETpartial;

  std::vector<float> GC(params.numFreqs,0.0);
  
  for(int f_indx=0;f_indx<params.numFreqs;f_indx++)
    {
      // Reset Tf to the identity.
      std::fill(Tf.begin(),Tf.end(),std::complex<float>(0.0,0.0));
      for(int comp=0;comp<numUsedComps;comp++)
	Tf[comp*numUsedComps+comp]=std::complex<float>(1.0,0.0);

      for(int lag=0;lag<params.numLags;lag++)
	{
	  argmt = -std::exp(std::complex<float>(0.0,argmtBASE*(float)(params.lagList[lag])*freq[f_indx]));
	  cblas_caxpy(numUsedComps*numUsedComps,&argmt,
		      Acmplx.data()+lag*numUsedComps*numUsedComps,1,
		      Tf.data(),1);
	}

      // I am going to explicitly invert this.
      info = LAPACKE_cgetrf(LAPACK_COL_MAJOR,numUsedComps,numUsedComps,
			    Tf.data(),numUsedComps,IPIV);
      info = LAPACKE_cgetri(LAPACK_COL_MAJOR,numUsedComps,Tf.data(),numUsedComps,IPIV);

      
      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		  numUsedComps,numUsedComps,numUsedComps,
		  &alphaComplex,Tf.data(),numUsedComps,
		  COVcmplx.data(),numUsedComps,
		  &betaComplex,tmp1.data(),numUsedComps);
      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasConjTrans,
		  numUsedComps,numUsedComps,numUsedComps,
		  &alphaComplex,tmp1.data(),numUsedComps,
 		  Tf.data(),numUsedComps,
		  &betaComplex,tmp2.data(),numUsedComps);

      info = LAPACKE_cheev(LAPACK_COL_MAJOR,'N','U',params.numCaused,
			   tmp2.data(),numUsedComps,wholeEigs.data());

      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		  params.numCaused,params.numCaused,params.numCaused,
		  &alphaComplex,Tf.data(),numUsedComps,
		  COVcmplx.data(),numUsedComps,
		  &betaComplex,tmp1.data(),numUsedComps);
      cblas_cgemm(CblasColMajor,CblasNoTrans,CblasConjTrans,
		  params.numCaused,params.numCaused,params.numCaused,
		  &alphaComplex,tmp1.data(),numUsedComps,
		  Tf.data(),numUsedComps,
		  &betaComplex,tmp2.data(),numUsedComps);

      info = LAPACKE_cheev(LAPACK_COL_MAJOR,'N','U',params.numCaused,
			   tmp2.data(),numUsedComps,partialEigs.data());

      DETwhole = std::accumulate(wholeEigs.begin(),wholeEigs.end(),1.0,std::multiplies<float>());
      DETpartial = std::accumulate(partialEigs.begin(),partialEigs.end(),1.0,std::multiplies<float>());

      GC[f_indx] = std::log(DETwhole/DETpartial);
    }

  

  if(params.outputType == "byfrequency")
    {
      for(int f_indx=0;f_indx<params.numFreqs;f_indx++)
	std::cout << freq[f_indx] << " " << GC[f_indx] << std::endl;
    }

  // Trapezoidal rule to do the integration
  if(params.outputType == "integral")
    {
      GC.front() = 0.5*GC.front();
      GC.back() = 0.5*GC.back();
  
      float intCausality = std::accumulate(GC.begin(),GC.end(),0.0)*(freq[1]-freq[0]);
  
      std::cout << intCausality << std::endl;
    }
  
  return 0;
}
