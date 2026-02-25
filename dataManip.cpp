#include "dataClass.h"
#include "dataManip.h"
#include "dataCompute.h"
#include <vector>
#include <cmath>
#include <cblas.h>
#include <string>
#include <complex>

  
dataClass<float> butterFilter(dataClass<float> datain,std::string filtType,float fCutoff,int filtOrder)
{
  dataClass<float> OP(datain.getEpochPoints());
  int sampRate = datain.getSampRate();
  OP.setSampRate(sampRate);
  float phiCPrime = std::tan(M_PI*fCutoff/float(sampRate));
  float phiCPrimeSq = std::pow(phiCPrime,2);
  
  int k = filtOrder/2-1;

  std::vector<float> a(3*(k+1),0.0);
  std::vector<float> b(3*(k+1),0.0);

  // Set the coefficients

  float num,den,tmp;
  
  for(int i=1;i<=k+1;i++)
    {
      num = M_PI*(2.0*float(i)-1.0);
      den = 2.0*float(filtOrder);
      if(filtType == "LP")
	{
	  tmp = 2.0*std::cos(num/den)*phiCPrime;

	  a[(i-1)*3+0] = phiCPrimeSq;
	  a[(i-1)*3+1] = 2.0*phiCPrimeSq;
	  a[(i-1)*3+2] = phiCPrimeSq;

	  b[(i-1)*3+0] = 1.0+tmp+phiCPrimeSq;
	  b[(i-1)*3+1] = 2.0*(phiCPrimeSq-1.0);
	  b[(i-1)*3+2] = 1.0-tmp+phiCPrimeSq;
	}
      if(filtType == "HP")
	{
	  tmp = 2.0*std::cos(num/den)/phiCPrime;
            
	  a[(i-1)*3+0] = 1.0/phiCPrimeSq;
	  a[(i-1)*3+1] = -2.0/phiCPrimeSq;
	  a[(i-1)*3+2] = 1.0/phiCPrimeSq;
 
          b[(i-1)*3+0] = 1.0+tmp+1.0/phiCPrimeSq;
	  b[(i-1)*3+1] = 2.0*(1.0-1.0/phiCPrimeSq);
	  b[(i-1)*3+2] = 1.0-tmp+1.0/phiCPrimeSq;
	}
    }

  // Left to Right
  std::vector<float> Tactive;
  std::vector<float> Ttmp;

  int numComps = datain.getNumComps();
  for(int epoch=0;epoch<datain.getNumEpochs();epoch++)
    {
      Tactive = datain.isoEpoch(epoch).dataArray();
      

      for(int filt=1;filt<=k+1;filt++)
	{
	  Ttmp = Tactive;
	  for(int tp=2;tp<datain.getEpochPoints();tp++)
	    {
	      cblas_saxpy(numComps,a[(filt-1)*3+0],Ttmp.data()+tp*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,a[(filt-1)*3+1],Ttmp.data()+(tp-1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,a[(filt-1)*3+2],Ttmp.data()+(tp-2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,b[(filt-1)*3+1],Tactive.data()+(tp-1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,b[(filt-1)*3+2],Tactive.data()+(tp-2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_sscal(numComps,1.0/b[(filt-1)*3+0],Tactive.data()+tp*numComps,1);
	    }
	}
      // Right to Left

      for(int filt=1;filt<=k+1;filt++)
	{
	  Ttmp = Tactive;
	  for(int tp=datain.getEpochPoints()-3;tp>=0;tp--)
	    {
	      cblas_saxpy(numComps,a[(filt-1)*3+0],Ttmp.data()+tp*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,a[(filt-1)*3+1],Ttmp.data()+(tp+1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,a[(filt-1)*3+2],Ttmp.data()+(tp+2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,b[(filt-1)*3+1],Tactive.data()+(tp+1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_saxpy(numComps,b[(filt-1)*3+2],Tactive.data()+(tp+2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_sscal(numComps,1.0/b[(filt-1)*3+0],Tactive.data()+tp*numComps,1);
	    }
	}
      // Insert at the end
      
      OP.addEpoch(Tactive,OP.getNumEpochs());
    }
  return OP;
}
  
dataClass<double> butterFilter(dataClass<double> datain,std::string filtType,double fCutoff,int filtOrder)
{
  dataClass<double> OP(datain.getEpochPoints());
  float sampRate = datain.getSampRate();
  OP.setSampRate(sampRate);
  double phiCPrime = std::tan(M_PI*fCutoff/double(sampRate));
  double phiCPrimeSq = std::pow(phiCPrime,2);
  
  int k = filtOrder/2-1;

  std::vector<double> a(3*(k+1),0.0);
  std::vector<double> b(3*(k+1),0.0);

  // Set the coefficients

  double num,den,tmp;
  
  for(int i=1;i<=k+1;i++)
    {
      num = M_PI*(2.0*double(i)-1.0);
      den = 2.0*double(filtOrder);
      if(filtType == "LP")
	{
	  tmp = 2.0*std::cos(num/den)*phiCPrime;

	  a[(i-1)*3+0] = phiCPrimeSq;
	  a[(i-1)*3+1] = 2.0*phiCPrimeSq;
	  a[(i-1)*3+2] = phiCPrimeSq;

	  b[(i-1)*3+0] = 1.0+tmp+phiCPrimeSq;
	  b[(i-1)*3+1] = 2.0*(phiCPrimeSq-1.0);
	  b[(i-1)*3+2] = 1.0-tmp+phiCPrimeSq;
	}
      if(filtType == "HP")
	{
	  tmp = 2.0*std::cos(num/den)/phiCPrime;
            
	  a[(i-1)*3+0] = 1.0/phiCPrimeSq;
	  a[(i-1)*3+1] = -2.0/phiCPrimeSq;
	  a[(i-1)*3+2] = 1.0/phiCPrimeSq;
 
          b[(i-1)*3+0] = 1.0+tmp+1.0/phiCPrimeSq;
	  b[(i-1)*3+1] = 2.0*(1.0-1.0/phiCPrimeSq);
	  b[(i-1)*3+2] = 1.0-tmp+1.0/phiCPrimeSq;
	}
    }

  // Left to Right
  std::vector<double> Tactive;
  std::vector<double> Ttmp;

  int numComps = datain.getNumComps();
  for(int epoch=0;epoch<datain.getNumEpochs();epoch++)
    {
      Tactive = datain.isoEpoch(epoch).dataArray();
      

      for(int filt=1;filt<=k+1;filt++)
	{
	  Ttmp = Tactive;
	  for(int tp=2;tp<datain.getEpochPoints();tp++)
	    {
	      cblas_daxpy(numComps,a[(filt-1)*3+0],Ttmp.data()+tp*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,a[(filt-1)*3+1],Ttmp.data()+(tp-1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,a[(filt-1)*3+2],Ttmp.data()+(tp-2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,b[(filt-1)*3+1],Tactive.data()+(tp-1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,b[(filt-1)*3+2],Tactive.data()+(tp-2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_dscal(numComps,1.0/b[(filt-1)*3+0],Tactive.data()+tp*numComps,1);
	    }
	}
      // Right to Left

      for(int filt=1;filt<=k+1;filt++)
	{
	  Ttmp = Tactive;
	  for(int tp=datain.getEpochPoints()-3;tp>=0;tp--)
	    {
	      cblas_daxpy(numComps,a[(filt-1)*3+0],Ttmp.data()+tp*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,a[(filt-1)*3+1],Ttmp.data()+(tp+1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,a[(filt-1)*3+2],Ttmp.data()+(tp+2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,b[(filt-1)*3+1],Tactive.data()+(tp+1)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_daxpy(numComps,b[(filt-1)*3+2],Tactive.data()+(tp+2)*numComps,1,Tactive.data()+tp*numComps,1);
	      cblas_dscal(numComps,1.0/b[(filt-1)*3+0],Tactive.data()+tp*numComps,1);
	    }
	}
      // Insert at the end
      
      OP.addEpoch(Tactive,OP.getNumEpochs());
    }
  return OP;
}
  

dataClass<float> linearTrans(dataClass<float> datain,std::vector<float> L)
{
  // Transform is M x C, data is C x N
  int C = datain.getNumComps();
  int N = datain.getTotalPoints();
  int M;
  if(std::floor(L.size()/C) == std::ceil(L.size()/C))
    M = int(L.size()/C);
  else
    {
      std::cout << "The transformation matrix is incompatible" << std::endl;
      exit(1);
    }
  // The output size will be M*N
  std::vector<float> outSet(M*N,0.0);
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,N,C,
	      1.0,L.data(),M,datain.dataArray().data(),C,
	      0.0,outSet.data(),M);

  dataClass<float> toReturn(datain.getEpochPoints(),M,outSet,datain.getSampRate());
  return toReturn;
}


dataClass<double> linearTrans(dataClass<double> datain,std::vector<double> L)
{
  // Transform is M x C, data is C x N
  int C = datain.getNumComps();
  int N = datain.getTotalPoints();
  int M;
  if(std::floor(L.size()/C) == std::ceil(L.size()/C))
    M = int(L.size()/C);
  else
    {
      std::cout << "The transformation matrix is incompatible" << std::endl;
      exit(1);
    }
  // The output size will be M*N
  std::vector<double> outSet(M*N,0.0);
  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,N,C,
	      1.0,L.data(),M,datain.dataArray().data(),C,
	      0.0,outSet.data(),M);

  dataClass<double> toReturn(datain.getEpochPoints(),M,outSet,datain.getSampRate());
  return toReturn;
}

dataClass<float> removeMean(dataClass<float> datain)
{
  
  int numComps = datain.getNumComps();
  int epochPts = datain.getEpochPoints();
  int numEpochs = datain.getNumEpochs();

  dataClass<float> toReturn(epochPts);
  toReturn.setSampRate(datain.getSampRate());
  std::vector<float> sumVec(numComps,0.0);

  std::vector<float> dd(numComps*epochPts,0.0);
  std::vector<float> dout(numComps*epochPts,0.0);
  std::vector<float> ones(epochPts,1.0);
  
  float scaleFactor = 1.0/float(epochPts);
  
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      dd = datain.isoEpoch(epoch).dataArray();
      for(int tp=0;tp<epochPts;tp++)
	cblas_saxpy(numComps,scaleFactor,dd.data()+tp*numComps,1,sumVec.data(),1);

      for(int comp=0;comp<numComps;comp++)
	cblas_saxpy(epochPts,-sumVec[comp],ones.data(),1,dd.data()+comp,numComps);
      
      
      toReturn.addEpoch(dd,epoch);
    }
  
  return toReturn;
}

dataClass<double> removeMean(dataClass<double> datain)
{
  
  int numComps = datain.getNumComps();
  int epochPts = datain.getEpochPoints();
  int numEpochs = datain.getNumEpochs();

  dataClass<double> toReturn(epochPts);
  toReturn.setSampRate(datain.getSampRate());
  std::vector<double> sumVec(numComps,0.0);

  std::vector<double> dd(numComps*epochPts,0.0);
  std::vector<double> dout(numComps*epochPts,0.0);
  std::vector<double> ones(epochPts,1.0);
  
  double scaleFactor = 1.0/double(epochPts);
  
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      dd = datain.isoEpoch(epoch).dataArray();
      for(int tp=0;tp<epochPts;tp++)
	cblas_daxpy(numComps,scaleFactor,dd.data()+tp*numComps,1,sumVec.data(),1);

      for(int comp=0;comp<numComps;comp++)
	cblas_daxpy(epochPts,-sumVec[comp],ones.data(),1,dd.data()+comp,numComps);

      toReturn.addEpoch(dd,epoch);
    }
  return toReturn;
}

dataClass<float> applyTapers(dataClass<float> tapers,dataClass<float> datain)
{
  if(tapers.getNumEpochs() != 1)
    {
      std::cout << "tapers arguments should be a single epoch" << std::endl;
      exit(1);
    }
  if(tapers.getEpochPoints() != datain.getEpochPoints())
    {
      std::cout << "The number of points per epoch is different in the two arguments" << std::endl;
      exit(1);
    }
  int numComps = datain.getNumComps();
  int numTapers = tapers.getNumComps();
  int epochPts = tapers.getEpochPoints();
  int N = datain.getTotalPoints();
  
  std::vector<float> scaleVec = tapers.dataArray();
  std::vector<float> dataVec = datain.dataArray();
  std::vector<float> dataOut(numTapers*numComps*N);
      
  for(int epoch=0;epoch<datain.getNumEpochs();epoch++)
    for(int tp=0;tp<epochPts;tp++)
      for(int tap=0;tap<numTapers;tap++)
	for(int comp=0;comp<numComps;comp++)
	  dataOut[epoch*epochPts*(numComps*numTapers)+tp*(numComps*numTapers)+tap*numComps+comp] =
	    scaleVec[tp*numTapers+tap]*dataVec[epoch*epochPts*numComps+tp*numComps+comp];
       
  dataClass<float> toReturn(epochPts,numComps*numTapers,dataOut,datain.getSampRate());
  
  return toReturn;
}

dataClass<double> applyTapers(dataClass<double> tapers,dataClass<double> datain)
{
  if(tapers.getNumEpochs() != 1)
    {
      std::cout << "tapers arguments should be a single epoch" << std::endl;
      exit(1);
    }
  if(tapers.getEpochPoints() != datain.getEpochPoints())
    {
      std::cout << "The number of points per epoch is different in the two arguments" << std::endl;
      exit(1);
    }
  int numComps = datain.getNumComps();
  int numTapers = tapers.getNumComps();
  int epochPts = tapers.getEpochPoints();
  int N = datain.getTotalPoints();
  
  std::vector<double> scaleVec = tapers.dataArray();
  std::vector<double> dataVec = datain.dataArray();
  std::vector<double> dataOut(numTapers*numComps*N);
      
  for(int epoch=0;epoch<datain.getNumEpochs();epoch++)
    for(int tp=0;tp<epochPts;tp++)
      for(int tap=0;tap<numTapers;tap++)
	for(int comp=0;comp<numComps;comp++)
	  dataOut[epoch*epochPts*(numComps*numTapers)+tp*(numComps*numTapers)+tap*numComps+comp] =
	    scaleVec[tp*numTapers+tap]*dataVec[epoch*epochPts*numComps+tp*numComps+comp];
       
  dataClass<double> toReturn(epochPts,numComps*numTapers,dataOut,datain.getSampRate());
  
  return toReturn;
}

dataClass<float> removeFLines(dataClass<float> datain,float f0)
{
  float dt = 1.0/float(datain.getSampRate());
  float Nyquist = float(datain.getSampRate())/2.0;
  int epochPts = datain.getEpochPoints();
  int numComps = datain.getNumComps();
  int numEpochs = datain.getNumEpochs();
  float T = float(epochPts)/float(datain.getSampRate());
  
  int numTapers = int(2.0*T-1.0); // I am assuming BW of 1Hz+1Hz
  
  dataClass<float> tapers = dpss(epochPts,T,numTapers);

  dataClass<std::complex<float>> taperFFT = FFT(tapers);
  std::vector<std::complex<float>> Wvec(numTapers);
  std::complex<float> Wnrm = std::complex<float>(0.0,0.0);

  for(int taper=0;taper<numTapers;taper++)
    {
      Wvec[taper] = taperFFT.dataArray()[taper];
      Wnrm += std::conj(Wvec[taper])*Wvec[taper];
    }

  dataClass<float> taperedData = applyTapers(tapers,datain);
  dataClass<std::complex<float>> dataFFT = FFT(taperedData);

  int freqindx = std::round((float(epochPts)*f0)/float(datain.getSampRate()));
  
  std::complex<float> coeffVector;
  std::vector<std::complex<float>> expVec(epochPts);
  // A unit sinusoid with z(0)=1
  for(int tp=0;tp<epochPts;tp++)
    expVec[tp] = std::exp(std::complex<float>(0.0,2.0*M_PI*f0*tp*dt));

  dataClass<float> dataOut(epochPts);
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      std::vector<float> epochTimeData = datain.isoEpoch(epoch).dataArray();
      for(int comp=0;comp<numComps;comp++)
	{
	  dataClass<std::complex<float>> tmpData = dataFFT.isoEpoch(epoch);
	  std::vector<int> compsToKeep(numTapers);
	  for(int taper=0;taper<numTapers;taper++)
	    {
	      compsToKeep[taper] = comp+taper*numComps;
	    }
	  tmpData.keepComponents(compsToKeep);
	  std::vector<std::complex<float>> tmpVector = tmpData.dataArray();
	  std::complex<float> Xnrm = std::complex<float>(0.0,0.0);
	  for(int taper=0;taper<numTapers;taper++)
	    Xnrm += std::complex<float>(2.0,0.0)*std::conj(Wvec[taper])*tmpVector[freqindx*numTapers+taper];

	  coeffVector = Xnrm/Wnrm;
	  //std::cout << coeffVector.real() << " + i" << coeffVector.imag() << std::endl;
	  for(int tp=0;tp<epochPts;tp++)
	    epochTimeData[comp+tp*numComps] -= (coeffVector*expVec[tp]).real();
	}
      dataOut.addEpoch(epochTimeData,epoch); // This should add each epoch at the end.      
    }
  
  return dataOut;
}
	  
dataClass<double> removeFLines(dataClass<double> datain,double f0)
{
  double dt = 1.0/double(datain.getSampRate());
  double Nyquist = double(datain.getSampRate())/2.0;
  int epochPts = datain.getEpochPoints();
  int numComps = datain.getNumComps();
  int numEpochs = datain.getNumEpochs();
  double T = double(epochPts)/double(datain.getSampRate());
  
  int numTapers = int(2.0*T-1.0); // I am assuming BW of 1Hz+1Hz
  
  dataClass<double> tapers = dpss(epochPts,T,numTapers);

  dataClass<std::complex<double>> taperFFT = FFT(tapers);
  std::vector<std::complex<double>> Wvec(numTapers);
  std::complex<double> Wnrm = std::complex<double>(0.0,0.0);

  for(int taper=0;taper<numTapers;taper++)
    {
      Wvec[taper] = taperFFT.dataArray()[taper];
      Wnrm += std::conj(Wvec[taper])*Wvec[taper];
    }

  dataClass<double> taperedData = applyTapers(tapers,datain);
  dataClass<std::complex<double>> dataFFT = FFT(taperedData);

  int freqindx = std::round((double(epochPts)*f0)/double(datain.getSampRate()));
  
  std::complex<double> coeffVector;
  std::vector<std::complex<double>> expVec(epochPts);
  // A unit sinusoid with z(0)=1
  for(int tp=0;tp<epochPts;tp++)
    expVec[tp] = std::exp(std::complex<double>(0.0,2.0*M_PI*f0*tp*dt));

  dataClass<double> dataOut(epochPts);
  for(int epoch=0;epoch<numEpochs;epoch++)
    {
      std::vector<double> epochTimeData = datain.isoEpoch(epoch).dataArray();
      for(int comp=0;comp<numComps;comp++)
	{
	  dataClass<std::complex<double>> tmpData = dataFFT.isoEpoch(epoch);
	  std::vector<int> compsToKeep(numTapers);
	  for(int taper=0;taper<numTapers;taper++)
	    {
	      compsToKeep[taper] = comp+taper*numComps;
	    }
	  tmpData.keepComponents(compsToKeep);
	  std::vector<std::complex<double>> tmpVector = tmpData.dataArray();
	  std::complex<double> Xnrm = std::complex<double>(0.0,0.0);
	  for(int taper=0;taper<numTapers;taper++)
	    Xnrm += std::complex<double>(2.0,0.0)*std::conj(Wvec[taper])*tmpVector[freqindx*numTapers+taper];

	  coeffVector = Xnrm/Wnrm;
	  //std::cout << coeffVector.real() << " + i" << coeffVector.imag() << std::endl;
	  for(int tp=0;tp<epochPts;tp++)
	    epochTimeData[comp+tp*numComps] -= (coeffVector*expVec[tp]).real();
	}
      dataOut.addEpoch(epochTimeData,epoch); // This should add each epoch at the end.      
    }
  
  return dataOut;
}	  

