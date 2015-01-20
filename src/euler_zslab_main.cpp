/**
 * \file euler_zslab_main.cpp
 * \brief
 *
 * Solve 3D Euler equations on a cartesian grid using 
 * the Godunov scheme (with Riemann solvers) and z-slab update method.
 *
 * \date Sept 12, 2012
 * \author P. Kestener
 *
 * $Id: euler_zslab_main.cpp 3452 2014-06-17 10:09:48Z pkestene $
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE

// for catching floating point errors
#include <fenv.h>
#include <signal.h>

// common header
#include <cstdlib>
#include <iostream>
#include <ctime>

// HYDRO solvers
#include <HydroRunGodunovZslab.h>

// MHD solver
#include <MHDRunGodunovZslab.h>

#include "utils/monitoring/date.h"

//#include "svn_version.h"
//#include "build_date.h"

/** Parse command line / configuration parameter file */
#include "GetPot.h"
#include <ConfigMap.h>

/* Graphics Magick C++ API to dump PNG image files */
#ifdef USE_GM
#include <Magick++.h>
#endif // USE_GM

// OpenMP
#if _OPENMP
#include <omp.h>
#endif

// signal handler for catching floating point errors
void fpehandler(int sig_num)
{
  signal(SIGFPE, fpehandler);
  printf("SIGFPE: floating point exception occured of type %d, exiting.\n",sig_num);
  abort();
}

void print_help(int argc, char* argv[]);

using hydroSimu::HydroRunGodunovZslab;
using hydroSimu::MHDRunGodunovZslab;

/************************************
 ************************************/
int main (int argc, char * argv[]) {

  /* install signal handler for floating point errors (only usefull when debugging, doing a backtrace in gdb) */
  //feenableexcept(FE_DIVBYZERO | FE_INVALID);
  //signal(SIGFPE, fpehandler);

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

  /* search for configuration parameter file */
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
    std::cout << "WARNING : you are trying to use z-slab method on a 2D problem...\n";
    std::cout << "This is not possible ! Only available on 3D problems.\n";
    std::cout << "Z-slab implementation is disabled.\n";
  }
  if (zSlabEnabled) {
    int zSlabNb = configMap.getInteger("implementation","zSlabNb",0);
    std::cout << "Using z-slab method with " << zSlabNb << " z-slab pieces." << std::endl;
  }

  /*
   * print date to be the first item printed in log
   */
  std::cout << "###############################"   << std::endl;
  if (mhdEnabled)
    std::cout << "######  MHD simulations  ######" << std::endl;
  else
    std::cout << "###### Hydro simulations ######" << std::endl;
  std::cout << "###############################"   << std::endl;
  //std::cout << "SVN revision : " << svn_version()  << std::endl;
  //std::cout << "Build   date : " << build_date()   << std::endl;
  std::cout << "Current date : " << current_date() << std::endl;

  // when using PNG output, we must initialize GraphicsMagick library
#ifdef USE_GM
  Magick::InitializeMagick("");
#endif

  /* 
   * process parameter file when constructing HydroParameter object;
   * initialize hydro grid.
   */
  if (!mhdEnabled) { // HYDRO simulations

      if (zSlabEnabled) {
	HydroRunGodunovZslab * hydroRun = new HydroRunGodunovZslab(configMap);
	hydroRun->start();
	delete hydroRun;
      } else {
	std::cerr << "Z-slab not enabled ! Check your parameter file." << std::endl;
      }

  } else { // MHD simulations

    if (zSlabEnabled) {
      MHDRunGodunovZslab * mhdRun = new MHDRunGodunovZslab(configMap);
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

  (void) argc;
  
  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << argv[0] << " is a C++ version of hydro simulations : " << endl;
  cout << "solve 3D Euler equations on a cartesian grid." << endl;
  cout << "available method : Godunov with z-slab" << endl;
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
