/**
 * \file euler_mpi_main.cpp
 * \brief
 * Solve 2D/3D Euler equation on a cartesian grid using the Godunov
 * method (with Riemann solvers) using MPI+CUDA.
 *
 * \date 19 Oct 2010
 * \author P. Kestener
 *
 * $Id: euler_mpi_main.cpp 3452 2014-06-17 10:09:48Z pkestene $ 
 */

#include <cstdlib>
#include <iostream>

// HYDRO solver
#include "HydroRunGodunovMpi.h"

// MHD solver
#include "MHDRunGodunovMpi.h"

#include <GlobalMpiSession.h>

#include "utils/monitoring/date.h"

//#include "svn_version.h"
//#include "build_date.h"

/** Parse configuration parameter file */
#include "GetPot.h"
#include <ConfigMap.h>

// OpenMP
#if _OPENMP
#include <omp.h>
#endif

void print_help(int argc, char* argv[]);

using hydroSimu::HydroRunGodunovMpi;
using hydroSimu::MHDRunGodunovMpi;
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
  
  // hydro only parameters:
  // set traceVersion and doTrace :
  // doTrace parameter (trigger trace characteristic computations)
  int traceVersion  = configMap.getInteger("hydro", "traceVersion", 1);
  bool traceEnabled = (traceVersion == 0) ? false : true;
  bool unsplitEnabled = configMap.getBool("hydro", "unsplit", true);
  int unsplitVersion = -1;
  if (unsplitEnabled) {
    unsplitVersion = configMap.getInteger("hydro", "unsplitVersion", 0);
  }

  // OpenMP enabled ?
  if (MpiComm::world().getRank() == 0) {
#if _OPENMP
#pragma omp parallel
    {
      const int threadId = omp_get_thread_num();
#pragma omp barrier
      if ( threadId == 0 ) {
	const int numberOfThreads = omp_get_num_threads();
	std::cout << "OpenMP enabled with " << numberOfThreads << " threads" << std::endl;
      }
    }
#endif // _OPENMP
  } // end OpenMP enabled ?

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

    // other usefull information
    if (!mhdEnabled) {
      std::cout << "unsplit scheme enabled ?  " << unsplitEnabled << std::endl;
      std::cout << "unsplit scheme version :  " << unsplitVersion << std::endl;
    }
  }
  
  /* 
   * process parameter file when constructing HydroParameter object;
   * initialize hydro grid.
   *
   * Please notice that various MPI stuff-related sanity checks are done 
   * in HydroMpiParameters' constructor.
   */
  if (!mhdEnabled) { // HYDRO simulations

    HydroRunGodunovMpi * hydroRun = new HydroRunGodunovMpi(configMap);
  
    // start simulation
    hydroRun->setTraceEnabled(traceEnabled);
    hydroRun->setTraceVersion(traceVersion);
    hydroRun->start();

    delete hydroRun;

  } else { // MHD simulations

    MHDRunGodunovMpi * mhdRun = new MHDRunGodunovMpi(configMap);

    mhdRun->start();

    delete mhdRun;

  }

  return EXIT_SUCCESS;

} // main

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void print_help(int argc, char* argv[]) {
  
  (void) argc;

  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << argv[0] << " is a C++ version of hydro simulations : " << endl;
  cout << "solve 2D/3D Euler (Hydro or MHD) equations on a cartesian grid with MPI+CUDA." << endl;
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
