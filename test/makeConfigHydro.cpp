/**
 * \file makeConfigHydro.cpp
 * \brief a simple config builder, to be used for testing
 * g++ -O3 -o makeConfigHydro makeConfigHydro.cpp
 *
 * \date January 2010
 * \author P. Kestener
 */
#include "GetPot.h"
#include "cstdlib"
#include "iostream"

using namespace std;

int main(int argc, char **argv)
{
  /* parse command line arguments */
  GetPot cl(argc, argv);

  const int nx  = cl.follow(100,   "--nx"); 
  const int ny  = cl.follow(100,   "--ny");
  const int nz  = cl.follow(1,     "--nz");
  const int noutput = cl.follow(100, "--noutput");
  const int nstepmax = cl.follow(1000, "--nstepmax");

  /* dump parameter file (INI format) */
  cout << "[run]" << endl;
  cout << "tend=1.2" << endl;
  cout << "noutput=" << noutput << endl;
  cout << "nstepmax="<< nstepmax << endl;
  cout << endl;
  cout << endl;
  cout << "[mesh]" << endl;
  cout << "nx=" << nx << endl;
  cout << "ny=" << ny << endl;
  cout << "nz=" << nz << endl;
  cout << "# BoundaryConditionType :" << endl;
  cout << "# BC_UNDEFINED=0" << endl;
  cout << "# BC_DIRICHLET=1" << endl;
  cout << "# BC_NEUMANN=2 " << endl;
  cout << "# BC_PERIODIC=3" << endl;
  cout << "boundary_xmin=2" << endl;
  cout << "boundary_xmax=2" << endl;
  cout << "boundary_ymin=2" << endl;
  cout << "boundary_ymax=2" << endl;
  cout << "boundary_zmin=2" << endl;
  cout << "boundary_zmax=2" << endl;
  cout << endl;
  cout << endl;
  cout << "[hydro]" << endl;
  cout << "problem=jet" << endl;
  cout << "courant_factor=0.8" << endl;
  cout << "niter_riemann=10" << endl;
  cout << "traceVersion=0" << endl;
  cout << "iorder=1" << endl;
  cout << "slope_type=2" << endl;
  cout << "scheme=muscl" << endl;
  cout << "# valid Riemann config number are integer between 0 and 18" << endl;
  cout << "riemann_config_number=0" << endl;
  cout << "XLAMBDA=0.25" << endl;
  cout << "YLAMBDA=0.25" << endl;
  cout << "cfl=0.475" << endl;
  cout << endl;
  cout << endl;
  cout << "[jet]" << endl;
  cout << "enableJet=0" << endl;
  cout << "ijet=10" << endl;
  cout << "djet=1." << endl;
  cout << "ujet=300." << endl;
  cout << "pjet=1." << endl;
  cout << endl;
  cout << endl;
  cout << "[output]" << endl;
  cout << "latexAnimation=no" << endl;
  cout << "outputXsm=yes" << endl;
  cout << "outputVtk=no"  << endl;
  cout << "outputhdf5=no" << endl;
  cout << "outputPrefix=riemann" << endl;
  cout << "colorPng=no" << endl;
  
  return EXIT_SUCCESS;
}
