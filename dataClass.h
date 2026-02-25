#ifndef DATACLASS_H
#define DATACLASS_H

#include <vector>
#include <iostream>
#include <cmath>
#include <string>


template<typename T>
struct MVAR {
  std::vector<T> A;
  std::vector<T> R;

  float numComps;
  std::vector<int> lagList;
  
  MVAR(std::vector<T> Ain,std::vector<T> Rin,int numCompsin,std::vector<int> lagListin)
  {
    A = Ain;
    R = Rin;
    numComps = numCompsin;
    lagList = lagListin;
  }

  std::vector<T> getLag(int lag)
  {
    std::vector<T> lagMatrix(numComps*numComps);
    std::copy(A.begin()+lag*numComps*numComps,A.begin()+(lag+1)*numComps*numComps,lagMatrix.begin());
    return lagMatrix;
  }
};


template<typename T>
struct spectra {
  std::vector<std::vector<T>> S;
  std::vector<float> f;

  spectra(std::vector<std::vector<T>> Sin)
  {
    S = Sin;
  }
};

template<class T>
class dataClass {
public:
  dataClass(int,int = 0,std::vector<T> = {},int = 1);
  int getNumEpochs();
  int getNumComps();
  int getEpochPoints();
  int getTotalPoints();
  dataClass<T> isoEpoch(int);
  std::vector<T> dataArray();
  void removeEpoch(int);
  void addEpoch(std::vector<T>,int = 0);
  void addEpoch(dataClass<T>,int = 0);
  void removeComponent(int);
  void keepComponents(std::vector<int>);
  void addComponent(std::vector<T>);
  void setSampRate(int);
  int getSampRate();
  
private:
  int N;
  int numEpochs;
  int epochPts;
  int numComps;
  int sampRate;
  std::vector<T> dataVector;
  std::vector<std::string> timeLabels;
  std::vector<std::string> compLabels;
};
// End of header


// Definitions from here

template<class T>
dataClass<T>::dataClass(int ePts,int comps,std::vector<T> datain,int samp)
{
  dataVector = datain;
  numComps = comps;
  epochPts = ePts;
  sampRate = samp;
  if(datain.size() == 0)
    {
      N = 0;
      numEpochs = 0;
    }
  else
    {
      //std::cout << float(datain.size())/float(numComps) << std::endl;
      if(std::floor(float(datain.size())/float(numComps)) == std::ceil(float(datain.size())/float(numComps)))
	N = int(datain.size()/numComps);
      else
	{
	  std::cout << "data does not divide into components cleanly." << std::endl;
	  exit(1);
	}
      if(std::floor(float(N)/float(epochPts)) == std::ceil(float(N)/float(epochPts)))
	numEpochs = int(N/epochPts);
      else
	{
	  std::cout << "data does not divide into epochs cleanly." << std::endl;
	  exit(1);
	}
    }
}

template<class T>
int dataClass<T>::getNumEpochs()
{
  return numEpochs;
}

template<class T>
int dataClass<T>::getNumComps()
{
  return numComps;
}

template<class T>
int dataClass<T>::getEpochPoints()
{
  return epochPts;
}

template<class T>
int dataClass<T>::getTotalPoints()
{
  return N;
}

template<class T>
dataClass<T> dataClass<T>::isoEpoch(int epochNum)
{
  std::vector<T> outPut(epochPts*numComps);

  if(epochNum>=numEpochs)
    {
      std::cout << "Epoch out of bounds" << std::endl;
      exit(1);
    }
  else
    {
      std::copy(dataVector.begin()+epochNum*epochPts*numComps,
		dataVector.begin()+(epochNum+1)*epochPts*numComps,
		outPut.begin());
    }

  dataClass<T> toReturn(epochPts,numComps,outPut,sampRate);
  
  return toReturn;
}

template<class T>
void dataClass<T>::removeEpoch(int epochNum)
{
  if(epochNum>=getNumEpochs())
    {
      std::cout << "Epoch out of bounds" << std::endl;
      exit(1);
    }
  else
    {
      dataVector.erase(dataVector.begin()+epochNum*epochPts*numComps,
		       dataVector.begin()+(epochNum+1)*epochPts*numComps);
      numEpochs = numEpochs - 1;
      N = N-epochPts;
    }
  return;
}

// Removes a single component.
// Zero indexing applies.
template<class T>
void dataClass<T>::removeComponent(int compNum)
{
  if(compNum >= getNumComps())
    {
      std::cout << "Component out of bounds" << std::endl;
      exit(1);
    }
  else
    {
      for(int tp=N-1;tp>=0;tp--)// On really big sets this will not be fast.
	dataVector.erase(dataVector.begin()+compNum+tp*numComps);
      numComps = numComps - 1;
    }
  return;
}

template<class T>
void dataClass<T>::keepComponents(std::vector<int> compList)
{
  int numToKeep = compList.size();
  std::vector<T> rArray(numToKeep*N);
  if(numToKeep > numComps)
    {
      std::cout << "Component list is too long." << std::endl;
      exit(1);
    }
  for(int comp : compList)
    {
      if(comp>=numComps)
	{
	  std::cout << "A component index is too large." << std::endl;
	  exit(1);
	}
    }
  
  for(int tp=0;tp<N;tp++)
    for(int comp=0;comp<numToKeep;comp++)
      {
	rArray[tp*numToKeep+comp] = dataVector[tp*numComps+compList[comp]];
      }

  dataVector.resize(numToKeep*N);
  std::copy(rArray.begin(),rArray.end(),dataVector.begin());
  numComps = numToKeep;

  return;
}

template<class T>
void dataClass<T>::addComponent(std::vector<T> datain)
{
  // When the dataClass is empty 
  if(dataVector.size() == 0)
    {
      if(std::floor(float(datain.size())/float(epochPts)) != std::ceil(float(datain.size())/float(epochPts)))
	{
	  std::cout << "New component does not divide epochs" << std::endl;
	  exit(1);
	}
      dataVector = datain;
      numComps = 1;
      N = datain.size();
      numEpochs = int(datain.size()/epochPts);
    }
  else
    {
      if(datain.size() != N)
	{
	  std::cout << "Component to be added is the wrong length" << std::endl;
	  exit(1);
	}
      // numComps is old here, on purpose for now.
      
      typename std::vector<T>::iterator it = dataVector.begin()+numComps;

      for(int tp=0;tp<N;tp++)
	{
	  dataVector.insert(it,datain[tp]);
	  it += numComps+1;
	}	  

      numComps = numComps+1;
    }

  return;
}

template<class T>
void dataClass<T>::addEpoch(dataClass<T> datain,int loc)
{
  std::vector<T> dataVec = datain.isoEpoch(0).dataArray();
  if(datain.getEpochPoints() != epochPts)
    {
      std::cout << "Added epoch needs to have the same number of epoch points" << std::endl;
      exit(1);
    }
  
  if(dataVector.size() == 0)
    {
      if(std::floor(float(dataVec.size())/float(epochPts)) == std::ceil(float(dataVec.size())/float(epochPts)))
	numComps = int(dataVec.size()/epochPts);
      else
	{
	  std::cout << "The epoch to be added does not have the correct size" << std::endl;
	  exit(1);
	}
      
      dataVector = dataVec;
      N += epochPts;
      numEpochs++;
    }
  
  else if(dataVec.size() != epochPts*numComps)
    {
      std::cout << "The epoch to be added does not have the correct size" << std::endl;
      exit(1);
    }
  else
    {
      if(loc > numEpochs)
	{
	  std::cout << "Specified location is out of bounds" << std::endl;
	  exit(1);
	}
      if(loc == numEpochs) // At the end
	{
	  dataVector.insert(dataVector.end(),dataVec.begin(),dataVec.end());
	  N += epochPts;
	  numEpochs++;
	}
      else
	{
	  dataVector.insert(dataVector.begin()+numComps*epochPts*loc,dataVec.begin(),dataVec.end());
	  N += epochPts;
	  numEpochs++;
	}
    }
  
  return;
}

template<class T>
void dataClass<T>::addEpoch(std::vector<T> datain,int loc)
{  
  if(dataVector.size() == 0)
    {
      if(std::floor(float(datain.size())/float(epochPts)) == std::ceil(float(datain.size())/float(epochPts)))
	numComps = int(datain.size()/epochPts);
      else
	{
	  std::cout << "The epoch to be added does not have the correct size" << std::endl;
	  exit(1);
	}
      
      dataVector = datain;
      N += epochPts;
      numEpochs++;
    }
  
  else if(datain.size() != epochPts*numComps)
    {
      std::cout << "The epoch to be added does not have the correct size" << std::endl;
      exit(1);
    }
  else
    {
      if(loc > numEpochs)
	{
	  std::cout << "Specified location is out of bounds" << std::endl;
	  exit(1);
	}
      if(loc == numEpochs) // At the end
	{
	  dataVector.insert(dataVector.end(),datain.begin(),datain.end());
	  N += epochPts;
	  numEpochs++;
	}
      else
	{
	  dataVector.insert(dataVector.begin()+numComps*epochPts*loc,datain.begin(),datain.end());
	  N += epochPts;
	  numEpochs++;
	}
    }
  
  return;
}

template<class T>
std::vector<T> dataClass<T>::dataArray()
{
  return dataVector;
}

template<class T>
void dataClass<T>::setSampRate(int samp)
{
  sampRate = samp;
}

template<class T>
int dataClass<T>::getSampRate()
{
  return sampRate;
}

  


#endif


