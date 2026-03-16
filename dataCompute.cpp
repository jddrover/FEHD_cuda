#include "dataClass.h"
#include "dataCompute.h"
#include "dataManip.h"
#include <complex>
#include <fftw3.h>
#include <cblas.h>
#include <lapacke.h>
#include <algorithm>
#include <numeric>

dataClass<std::complex<float>> FFT(dataClass<float> datain)
{
  // The length of the array
  int numComps = datain.getNumComps();
  int epochPts = datain.getEpochPoints();
  int numEpochs = datain.getNumEpochs();
  int numFFT = numComps*numEpochs;
  int outEpochPts = epochPts/2+1;
  int istride = 1;
  int ostride = 1;
  int idist = epochPts;
  int odist = outEpochPts;
  
  int outsize = outEpochPts*numComps*numEpochs;
  // Need to make this into a double array - fftw3
  
  double *in;
  in = (double*) fftw_malloc(sizeof(double)*datain.dataArray().size());
  fftw_complex *out;
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*outsize);
  
  fftw_plan plan = fftw_plan_many_dft_r2c(1,&epochPts,numFFT,
					  in,NULL,istride,idist,
					  out,NULL,ostride,odist,
					  FFTW_MEASURE);
  // Something learned here - FFTW_MEASURE changes in, so that data should be initialized
  // AFTER the plan is made.
  // Set in here
  std::vector<float> data = datain.dataArray();
  for(int epoch = 0;epoch<numEpochs;epoch++)
    for(int comp=0;comp<numComps;comp++)
      for(int tp=0;tp<epochPts;tp++)
	in[epoch*epochPts+comp*datain.getTotalPoints()+tp] = double(data[epoch*numComps*epochPts+comp+tp*numComps]);

  
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  fftw_free(in);
  in = nullptr;

  std::vector<std::complex<float>> toGo(outsize);
  for(int epoch=0;epoch<numEpochs;epoch++)
    for(int comp=0;comp<numComps;comp++)
      for(int tp=0;tp<outEpochPts;tp++)
	toGo[epoch*outEpochPts*numComps+comp+tp*numComps] =
	  std::complex<float>(float(out[epoch*outEpochPts+comp*numEpochs*outEpochPts+tp][0]),
			      float(out[epoch*outEpochPts+comp*numEpochs*outEpochPts+tp][1]));
  
  fftw_free(out);
  out = nullptr;
  
  dataClass<std::complex<float>> toReturn(outEpochPts,numComps,toGo);

  return toReturn;
}

dataClass<std::complex<double>> FFT(dataClass<double> datain)
{
  // The length of the array
  int numComps = datain.getNumComps();
  int epochPts = datain.getEpochPoints();
  int numEpochs = datain.getNumEpochs();
  int numFFT = numComps*numEpochs;
  int outEpochPts = epochPts/2+1;
  int istride = 1;
  int ostride = 1;
  int idist = epochPts;
  int odist = outEpochPts;
  
  int outsize = outEpochPts*numComps*numEpochs;
  // Need to make this into a double array - fftw3
  
  double *in;
  in = (double*) fftw_malloc(sizeof(double)*datain.dataArray().size());
  fftw_complex *out;
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*outsize);
  
  fftw_plan plan = fftw_plan_many_dft_r2c(1,&epochPts,numFFT,
					  in,NULL,istride,idist,
					  out,NULL,ostride,odist,
					  FFTW_MEASURE);
  // Something learned here - FFTW_MEASURE changes in, so that data should be initialized
  // AFTER the plan is made.
  // Set in here
  std::vector<double> data = datain.dataArray();
  for(int epoch = 0;epoch<numEpochs;epoch++)
    for(int comp=0;comp<numComps;comp++)
      for(int tp=0;tp<epochPts;tp++)
	in[epoch*epochPts+comp*datain.getTotalPoints()+tp] = double(data[epoch*numComps*epochPts+comp+tp*numComps]);

  
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  fftw_free(in);
  in = nullptr;

  std::vector<std::complex<double>> toGo(outsize);
  for(int epoch=0;epoch<numEpochs;epoch++)
    for(int comp=0;comp<numComps;comp++)
      for(int tp=0;tp<outEpochPts;tp++)
	toGo[epoch*outEpochPts*numComps+comp+tp*numComps] =
	  std::complex<double>(double(out[epoch*outEpochPts+comp*numEpochs*outEpochPts+tp][0]),
			      double(out[epoch*outEpochPts+comp*numEpochs*outEpochPts+tp][1]));
  
  fftw_free(out);
  out = nullptr;
  
  dataClass<std::complex<double>> toReturn(outEpochPts,numComps,toGo);

  return toReturn;
}

// This should be rewritten so that just N, and WT are given.
dataClass<float> dpss(int N,float WT,int numTapers)
{
  // Construct the prolate matrix
  float Wfactor = float(WT)/float(N);
  std::vector<float> A(N*N,2.0*Wfactor);
  
  float val;
  for(int n=0;n<N;n++)
    for(int m=n+1;m<N;m++)
      {
	val = 2.0*M_PI*Wfactor*float(n-m);
	A[m*N+n] = std::sin(val)/(M_PI*float(n-m));
      }
  
  std::vector<float> evs(N,0.0); // Store the eigenvalues
  int info;
  std::vector<int> ifail(N);
  int eigsFound;
  std::vector<float> Z(N*numTapers);
  // Want to use the expert routine here. SSYEVX - only a certain number if eigenvalues are found.
  //info = LAPACKE_ssyev(LAPACK_COL_MAJOR,'V','U',N,A.data(),N,evs.data());
  info = LAPACKE_ssyevx(LAPACK_COL_MAJOR,'V','I','U',N,A.data(),N,0.5,1.1,N-numTapers+1,N,
			0.0,&eigsFound,evs.data(),Z.data(),N,ifail.data());
  if(info != 0)
    {
      std::cout << "ssyevx gave non zero info" << std::endl;
      exit(1);
    }
  std::vector<float> Aout(N*numTapers);
  for(int row=0;row<numTapers;row++)
    for(int col=0;col<N;col++)
      Aout[col*numTapers+row] = Z[(numTapers-row-1)*N+col];
  
  dataClass<float> toReturn(N,numTapers,Aout);
  
  return toReturn;
}

dataClass<double> dpss(int N,double WT,int numTapers)
{
  // Construct the prolate matrix
  double Wfactor = double(WT)/double(N);
  std::vector<double> A(N*N,2.0*Wfactor);
  
  double val;
  for(int n=0;n<N;n++)
    for(int m=n+1;m<N;m++)
      {
	val = 2.0*M_PI*Wfactor*double(n-m);
	A[m*N+n] = std::sin(val)/(M_PI*double(n-m));
      }
  
  std::vector<double> evs(N,0.0); // Store the eigenvalues
  int info;
  std::vector<int> ifail(N);
  int eigsFound;
  std::vector<double> Z(N*numTapers);
  // Want to use the expert routine here. SSYEVX - only a certain number if eigenvalues are found.
  //info = LAPACKE_ssyev(LAPACK_COL_MAJOR,'V','U',N,A.data(),N,evs.data());
  info = LAPACKE_dsyevx(LAPACK_COL_MAJOR,'V','I','U',N,A.data(),N,0.5,1.1,N-numTapers+1,N,
			0.0,&eigsFound,evs.data(),Z.data(),N,ifail.data());
  if(info != 0)
    {
      std::cout << "ssyevx gave non zero info" << std::endl;
      exit(1);
    }
  std::vector<double> Aout(N*numTapers);
  for(int row=0;row<numTapers;row++)
    for(int col=0;col<N;col++)
      Aout[col*numTapers+row] = Z[(numTapers-row-1)*N+col];
  
  dataClass<double> toReturn(N,numTapers,Aout);
  
  return toReturn;
}
std::vector<std::vector<std::complex<float>>> computeSpectra(dataClass<float> datain,dataClass<float> tapers)
{
  int numEpochs = datain.getNumEpochs();
  int numTapers = tapers.getNumComps();
  int numComps = datain.getNumComps();
  
  dataClass<float> taperedData = applyTapers(tapers,datain);
  dataClass<std::complex<float>> FFTs = FFT(taperedData);

  int freqPts = FFTs.getEpochPoints();

  // Compute xx* of the FFTs  
  std::vector<std::vector<std::vector<std::vector<std::complex<float>>>>> Sall(numEpochs);
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      std::vector<std::complex<float>> dataVec = FFTs.isoEpoch(epoch).dataArray();
      std::vector<std::vector<std::complex<float>>> S(freqPts);
      std::vector<std::vector<std::vector<std::complex<float>>>> T(numTapers);
      for(int taper=0;taper<numTapers;taper++)
	{	  
	  for(int fp=0;fp<freqPts;fp++)
	    { 
	      std::vector<std::complex<float>> tmp(numComps*numComps,std::complex<float>(0.0,0.0));
	      cblas_cherk(CblasColMajor,CblasUpper,CblasNoTrans,numComps,1,
			  1.0,dataVec.data()+taper*numComps+fp*numComps*numTapers,numComps*numTapers,0.0,tmp.data(),numComps);
	      // Fill the lower triangle.
	      for(int col=0;col<numComps-1;col++)
		for(int row=col+1;row<numComps;row++)
		  tmp[col*numComps+row] = std::conj(tmp[row*numComps+col]);
	      S[fp] = tmp;
	      
	    }
	  T[taper] = S;
	}
      Sall[epoch] = T;
    }
  
  std::vector<std::vector<std::complex<float>>> S(freqPts);
  
  for(int fp=0;fp<freqPts;fp++)
    {
      std::vector<std::complex<float>> esum(numComps*numComps,0.0);
      for(int epoch=0;epoch<numEpochs;epoch++)
	{
	  std::vector<std::complex<float>> Ssum(numComps*numComps,0.0);
	  for(int taper=0;taper<numTapers;taper++)
	    {
	      float scaleVal = 1.0/float(numTapers);
	      cblas_caxpy(numComps*numComps,&scaleVal,Sall[epoch][taper][fp].data(),1,Ssum.data(),1);
	    }
	  float escale = 1.0/float(numEpochs);
	  cblas_caxpy(numComps*numComps,&escale,Ssum.data(),1,esum.data(),1);
	}
      S[fp] = esum;
    }
  
  return S;

}

MVAR<float> mkAR(dataClass<float> datain,std::vector<int> lagList,float P)
{
  int numEpochs = datain.getNumEpochs();
  int epochPts = datain.getEpochPoints();
  int numComps = datain.getNumComps();

  std::sort(lagList.begin(),lagList.end());

  int maxLag = lagList.back();
  int numLags = lagList.size();
  int epochAdj = epochPts-maxLag;
  
  std::vector<float> RHS(epochAdj*numEpochs*numComps,0.0);
  std::vector<float> LHS(epochAdj*numEpochs*numComps*numLags,0.0);
  
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      std::vector<float> dataArray = datain.isoEpoch(epoch).dataArray();
      for(int tp=maxLag;tp<epochPts;tp++)
	{
	  int tpUse = tp - maxLag;
	  for(int comp=0;comp<numComps;comp++)
	    RHS[epoch*epochAdj+tpUse+comp*numEpochs*epochAdj]=
	      dataArray[tp*numComps+comp];
	}
      for(int lagIndx=0;lagIndx<numLags;lagIndx++)
	for(int tp=0;tp<epochAdj;tp++)
	  {
	    int tpUse = tp+maxLag-lagList[lagIndx];
	    for(int comp=0;comp<numComps;comp++)
	      LHS[epoch*epochAdj+tp+lagIndx*numComps*numEpochs*epochAdj+
		  comp*numEpochs*epochAdj]=
		dataArray[tpUse*numComps+comp];
	  }
    }

  // Compute the SVD.
  int info;
  int m = numEpochs*epochAdj;
  int n = numLags*numComps;
  float superb[n-1];
  std::vector<float> S(n);
  std::vector<float> U(m*n);
  std::vector<float> VT(n*n);

  // Need to make a backup of LHS, gesvd detroys it.
  std::vector<float> LHSBAK(LHS);
  info = LAPACKE_sgesvd(LAPACK_COL_MAJOR,'S','S',epochAdj*numEpochs,numComps*numLags,
			LHSBAK.data(),epochAdj*numEpochs,
			S.data(),
			U.data(),epochAdj*numEpochs,
			VT.data(),numComps*numLags,superb);

  // Look for small singular values.

  float Ssum = 0;
  int breakVal;
  float Stotal = std::accumulate(S.begin(),S.end(),0);
  for(int indx=0;indx<n;indx++)
    {
      Ssum += S[n-indx-1]/Stotal;
      if(Ssum > 1-P)
	{
	  for(int idx=0;idx<indx;idx++)
	    S[n-idx-1] = 0.0;
	  break;
	}
    }

  for(int indx=0;indx<n;indx++)
    {
      if(S[indx]>0.0)
	{
	  float scaleFactor = 1.0/S[indx];
	  cblas_sscal(m,scaleFactor,U.data()+indx*m,1);
	}
      else
	cblas_sscal(m,0.0,U.data()+indx*m,1);
    }
  std::vector<float> SUB(n*numComps);
  cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,numComps,m,
	      1.0,U.data(),m,RHS.data(),m,
	      0.0,SUB.data(),n);

  std::vector<float> AT(numComps*n);
  cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,numComps,n,
	      1.0,VT.data(),n,SUB.data(),n,
	      0.0,AT.data(),n);

  // Compute the residuals
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,numComps,n,
	      -1.0,LHS.data(),m,AT.data(),n,
	      1.0,RHS.data(),m);
  
  std::vector<float> R(RHS);
  std::vector<float> A(AT);

  for(int col=0;col<n;col++)
    for(int row=0;row<numComps;row++)
      A[col*numComps+row] = AT[row*n+col];
  for(int col=0;col<m;col++)
    for(int row=0;row<numComps;row++)
      R[col*numComps+row] = RHS[row*m+col];


  MVAR<float> toReturn(A,R,numComps,lagList);
  return toReturn;
}

// Should also return the transform. 
std::vector<float> PCA(dataClass<float> datain)
{
  int numComps = datain.getNumComps();
  int N = datain.getTotalPoints();
  std::vector<float> dataArray = datain.dataArray();

  // Create a transposed copy
  std::vector<float> dT(dataArray);

  for(int col=0;col<numComps;col++)
    for(int row=0;row<N;row++)
      dataArray[col*N+row] = dT[row*numComps+col];

  float superb[numComps-1];
  std::vector<float> S(numComps);
  std::vector<float> U(numComps*N);
  std::vector<float> VT(numComps*numComps);
  
  int info = LAPACKE_sgesvd(LAPACK_COL_MAJOR,'S','S',N,numComps,
			    dataArray.data(),N,
			    S.data(),
			    U.data(),N,
			    VT.data(),numComps,superb);

  float Stotal = std::accumulate(S.begin(),S.end(),0);
  float P = 0.99995;// This is a choice. I want to catch any approximation of zero.
  int nonZeroSingVals;
  float Ssum = 0.0;
  for(int indx=0;indx<numComps;indx++)
    {
      Ssum += S[numComps-indx-1]/Stotal;

      if(Ssum > 1.0-P)
	{
	  nonZeroSingVals = numComps-indx;
	  break;
	}
    }

  std::vector<float> Tmat(nonZeroSingVals*numComps);
  
  for(int indx=0;indx<nonZeroSingVals;indx++)
    {
      float scaleFactor = 1.0/S[indx];
      cblas_sscal(numComps,scaleFactor,VT.data()+indx,numComps);
      cblas_scopy(numComps,VT.data()+indx,numComps,Tmat.data()+indx,nonZeroSingVals);
    }

  return Tmat;
}
