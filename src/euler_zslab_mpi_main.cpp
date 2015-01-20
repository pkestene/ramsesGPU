/**
 * \file euler_zslab_mpi_main.cpp
 * \brief
 * Solve 3D Euler equation on a cartesian grid using the Godunov
 * method (with Riemann solvers) using MPI+CUDA and z-slab update method..
 *
 * \date September 27, 2012
 * \author P. Kestener
 *
 * $Id: euler_zslab_mpi_main.cpp 3234 2014-02-03 16:40:34Z pkestene $ 
 */

#include <cstdlib>
#include <iostream>

// HYDRO solver
#include "HydroRunGodunovZslabMpi.h"

// MHD solver
#include "MHDRunGodunovZslabMpi.h"

#include <GlobalMpiSession.h>

#include "utils/monitoring/date.h"

//#include "svn_version.h"
//#include "build_date.h"

/** Parse configuration parameter file */
#include "GetPot.h"
#include <ConfigMap.h>

void print_help(int argc, char* argv[]);

using hydroSimu::HydroRunGodunovZslabMpi;
using hydroSimu::MHDRunGodunovZslabMpi;
using hydroSimu::GlobalMpiSession;
using hydroSimu::MpiComm;

/************************************
 ************************************/
int main (int argc, char * argv[]) {

  /* Initialize MPI session */
  GlobalMpiSession mpiSession(&argc,&argv);

  /* 
   * read parameters from parameter file or command line arguments
   */
  
  /* parse command line arguments */
  GetPot cl(argc, argv);
  
  /* search for multiple options with the same meaning HELP */
  if( cl.search(3, "--help", "-h", "--sos") ) {
    print_help(argc,argv);
    exit(0);
  }

  /* search for configuration parameter fileName on the command line */
  if( !cl.search(1, "--param") ) {
    std::cerr << "[Error] You did not provide a .ini parameter file !\n";
    exit(-1);
  }
  
  /* parse parameter fileName */
  const std::string default_input_file = std::string(argv[0])+ ".ini";
  const std::string input_file = cl.follow(default_input_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(input_file); 

  /* MHD enabled ? */
  bool mhdEnabled = configMap.getBool("MHD","enable", false);

  /* dump input file and exit */
  if( cl.search(2, "--dump-param-file", "-d") ) {
    std::cout << configMap << std::endl;
    exit(0);
  }
  
  /*
   * Using the z-slab method ???
   */
  bool zSlabEnabled = configMap.getBool("implementation","zSlabEnabled",false);
  // make sure we have a 3D problem
  int nz = configMap.getInteger("mesh","nz", 1);
  if (zSlabEnabled and nz<=1) {
    // disable z-slab method
    zSlabEnabled = false;
    if (MpiComm::world().getRank() == 0) {
      std::cout << "WARNING : you are trying to use z-slab method on a 2D problem...\n";
      std::cout << "This is not possible ! Only available on 3D problems.\n";
      std::cout << "Z-slab implementation is disabled.\n";
    }
  }
  if (zSlabEnabled) {
    int zSlabNb = configMap.getInteger("implementation","zSlabNb",0);
    if (MpiComm::world().getRank() == 0)
      std::cout << "Using z-slab method with " << zSlabNb << " z-slab pieces." << std::endl;
  }

  /*
   * print date to be the first item printed in log
   */
  if (MpiComm::world().getRank() == 0) {
    std::cout << "#####################################" << std::endl;
    if (mhdEnabled)
      std::cout << "###### MHD   simulations (MPI) ######" << std::endl;
    else
      std::cout << "###### Hydro simulations (MPI) ######" << std::endl;
    std::cout << "#####################################" << std::endl;
    //std::cout << "SVN revision : " << svn_version()  << std::endl;
    //std::cout << "Build   date : " << build_date()   << std::endl;
    std::cout << "Current date : " << current_date() << std::endl;
    
  }
  
  /* 
   * process parameter file when constructing HydroParameter object;
   * initialize hydro grid.
   *
   * Please notice that various MPI stuff-related sanity checks are done 
   * in HydroMpiParameters' constructor.
   */
  if (!mhdEnabled) { // HYDRO simulations

      if (zSlabEnabled) {
	
	HydroRunGodunovZslabMpi * hydroRun = new HydroRunGodunovZslabMpi(configMap);
	
	// start simulation
	hydroRun->start();

	delete hydroRun;

      } else {

	std::cerr << "Z-slab not enabled ! Check your parameter file." << std::endl;

      }

  } else { // MHD simulations

    if (zSlabEnabled) {

      MHDRunGodunovZslabMpi * mhdRun = new MHDRunGodunovZslabMpi(configMap);
      
      mhdRun->start();

      delete mhdRun;

    } else {
      
      std::cerr << "Z-slab not enabled ! Check your parameter file." << std::endl; 
      
    }

  } // end MHD simulation run

  return EXIT_SUCCESS;

} // main

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void print_help(int argc, char* argv[]) {
  
  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << argv[0] << " is a C++ version of hydro simulations : " << endl;
  cout << "solve 3D Euler (Hydro or MHD) equations on a cartesian grid with MPI+CUDA and zslab method." << endl;
  cout << endl; 
  cout << "USAGE:" << endl;
  cout << "--help, -h, --sos" << endl;
  cout << "        get some help about this program." << endl << endl;
  cout << "--param [string]" << endl;
  cout << "        specify configuration parameter file (INI format)" << endl;
  cout << endl << endl;       
  cout << "--dump-param-file, -d" << endl;
  cout << "        show contents of database that was created by file parser.";
  cout << endl;

} // print_help
