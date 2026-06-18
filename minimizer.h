#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <cblas.h>
struct planet
{
  std::vector<float> center;
  float mass;
};

class minimizer
{
public:
  minimizer(int, int);
  void printPlanets();
  void assignPlanets(std::vector<float>,std::vector<std::vector<float>>);
  std::vector<float> evaluateFunction(std::vector<float>);
  std::vector<float> distance(std::vector<float>);
  std::vector<std::vector<float>> differenceVector(std::vector<float>);
  std::vector<float> force(std::vector<float> coords,float=1.0,float=1.0);
  std::vector<float> advanceInTime(std::vector<float>,float=0.1,float=1.0,float=1.0);
private:
  std::list<planet> planets;
};
// -----------------------
// Constructor
// -----------------------
minimizer::minimizer(int numPlanets,int numVars)
{
  for(int indx=0;indx<numPlanets;indx++)
    {
      planet tmpPlanet;
      tmpPlanet.mass = 1.0;
      std::vector<float> tmpCenter(numVars,0.0);
      tmpPlanet.center = tmpCenter;
      planets.push_back(tmpPlanet);
    }
}

// Print planets
void minimizer::printPlanets()
{
  for(std::list<planet>::iterator it=planets.begin();it!=planets.end();++it)
    {
      planet tmpPlanet = (*it);
      std::cout << "Planet mass: " << tmpPlanet.mass << std::endl;
      for(std::vector<float>::iterator it2=tmpPlanet.center.begin();it2!=tmpPlanet.center.end();++it2)
	{
	  std::cout << (*it2) << std::endl;
	}

    }
}


// ----------------------
// Assign planets
// ----------------------
void minimizer::assignPlanets(std::vector<float> masses,std::vector<std::vector<float>> centers)
{
  int numPlanets = planets.size();
  if(numPlanets == 0)
    exit(1);
  
  // check arrays for correct size.
  int numVars = centers[0].size();
  //std::cout << "numvars = " << numVars << std::endl;
  int indx = 0;

  for(std::list<planet>::iterator it=planets.begin();it != planets.end();++it)
    {
      for(int cindx=0;cindx<numVars;cindx++)
	(*it).center[cindx] = centers[indx][cindx];
      (*it).mass = masses[indx];
      indx++;      
    }  
  return;
}
// -------------------------------------
// Compute the distance vector. This should be combined with the distance function.
// -------------------------------------
std::vector<std::vector<float>> minimizer::differenceVector(std::vector<float> coords)
{
  if(planets.size() == 0)
    {
      std::cout << "There are no planets" << std::endl;
      exit(1);
    }
  if(coords.size() != (*planets.begin()).center.size())
    {
      std::cout << "Sizes do not match" << std::endl;
      exit(1);
    }
  
  int numPlanets = planets.size();
  std::vector<std::vector<float>> diffVec;

  for(std::list<planet>::iterator it=planets.begin();it != planets.end();++it)
    {
      std::vector<float> tmpCenter = (*it).center;
      cblas_saxpy(coords.size(),-1.0,coords.data(),1,tmpCenter.data(),1);      
      diffVec.push_back(tmpCenter);
    }
	  
  return diffVec;
}
//---------------------------------------------------------------
//---------------------------------------------------------------
// Computes the distance from each planet
std::vector<float> minimizer::distance(std::vector<float> coords)
{
  if(planets.size() == 0)
    {
      std::cout << "There are no planets" << std::endl;
      exit(1);
    }
  if(coords.size() != (*planets.begin()).center.size())
    {
      std::cout << "Sizes do not match" << std::endl;
      exit(1);
    }
  int numPlanets = planets.size();
  
  std::vector<float> dist(numPlanets,0.0);
  int pindx = 0;
  for(std::list<planet>::iterator it=planets.begin();it != planets.end();++it)
    {
      planet tmpPlanet = (*it);            
      cblas_saxpy(coords.size(),-1.0,coords.data(),1,tmpPlanet.center.data(),1);
      dist[pindx] = cblas_snrm2(coords.size(),tmpPlanet.center.data(),1);
      pindx++;
    }
  
  return dist;
}
// ---------------
//
//
//
//
// Compute the force. This needs to be a small bump - x exp(-x)
std::vector<float> minimizer::force(std::vector<float> coords,float g,float lambda)
{
  // F = G * planet mass * 1/distance^2+

  if(planets.size() == 0)
    {
      std::cout << "There are no planets" << std::endl;
      exit(1);
    }
  if(coords.size() != (*planets.begin()).center.size())
    {
      std::cout << "Sizes do not match" << std::endl;
      exit(1);
    }

  int numPlanets = planets.size();
  std::vector<float> dist(numPlanets);
  //std::vector<std::vector<float>> diff(numPlanets);

  std::vector<std::vector<float>> diffVec = differenceVector(coords);
  for(int pindx=0;pindx<numPlanets;pindx++)
    {
      dist[pindx] = cblas_snrm2(coords.size(),diffVec[pindx].data(),1);
      if(dist[pindx] < 0.001) // Its a zero vector, don't normalize it, make it really zero.
	std::fill(diffVec[pindx].begin(),diffVec[pindx].end(),0.0);
      else
	cblas_sscal(coords.size(),1.0/dist[pindx],diffVec[pindx].data(),1);
    }

  std::vector<std::vector<float>> forceVec;
  std::list<planet>::iterator it = planets.begin();
  for(int indx=0;indx<numPlanets;indx++)
    {
      std::vector<float> forceTmp(coords.size(),0.0);
      forceVec.push_back(forceTmp);
      // At some point, normalize this so when g=1, the objective function has
      // a maximum value of 1.      
      float force = g*((*it).mass)*lambda*dist[indx]*std::exp(-lambda*dist[indx]);
      cblas_saxpy(coords.size(),force,diffVec[indx].data(),1,forceVec[indx].data(),1);
      it++;
    }
  // Add up all of the force vectors.
  std::vector<float> force_output(coords.size(),0.0);
  for(int indx=0;indx<numPlanets;indx++)
    cblas_saxpy(coords.size(),1.0,forceVec[indx].data(),1,force_output.data(),1);
      
  return force_output;
}
// ---------------------------
// Advance in time
// ---------------------------
std::vector<float> minimizer::advanceInTime(std::vector<float> coords,float dt,float g,float lambda)
{
  std::vector<float> forceVec = force(coords,g,lambda);
  cblas_saxpy(coords.size(),dt,forceVec.data(),1,coords.data(),1);
  return coords;
}
