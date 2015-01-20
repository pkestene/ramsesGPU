/**
 * \file euler_main.cpp
 * \brief
 *
 * Solve 2D/3D Euler equations on a cartesian grid using either :
 * - the Godunov scheme (with Riemann solvers)
 * - the Kurganov-Tadmor scheme 
 * - the Relaxing TVD scheme
 *
 * Also solve 2D/3D MHD equations Godunov+CTU scheme.
 *
 * \date 16/01/2010
 * \author P. Kestener
 *
 * $Id: euler_main.cpp 3452 2014-06-17 10:09:48Z pkestene $
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
#include <HydroRunGodunov.h>
#include <HydroRunRelaxingTVD.h>
#include <HydroRunKT.h>

// MHD solver
#include <MHDRunGodunov.h>

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

using hydroSimu::HydroRunGodunov;
using hydroSimu::HydroRunKT;
using hydroSimu::HydroRunRelaxingTVD;

using hydroSimu::MHDRunGodunov;

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

  const std::string numScheme  = cl.follow("godunov", "--scheme");
  bool useGodunov     = !numScheme.compare("godunov");
  bool useKurganov    = !numScheme.compare("kurganov");
  bool useRelaxingTVD = !numScheme.compare("relaxingTVD");
  std::cout << "method : " << numScheme << std::endl;

  /* parse parameters from input file */
  ConfigMap configMap(input_file); 

  /* MHD enabled ? */
  bool mhdEnabled = configMap.getBool("MHD","enable", false);

  // hydro-only relaxing TVD scheme
  if (useRelaxingTVD) // make sure we use 3 ghost cells on borders
    configMap.setInteger("mesh","ghostWidth", 3);

  /* dump input file and exit */
  if( cl.search(2, "--dump-param-file", "-d") ) {
    std::cout << configMap << std::endl;
    exit(0);
  }

  // hydro-only :
  // set traceVersion and doTrace :
  // doTrace parameter (trigger trace characteristic computations)
  int traceVersion  = configMap.getInteger("hydro", "traceVersion", 1);
  bool traceEnabled = (traceVersion == 0) ? false : true;
  bool unsplitEnabled = configMap.getBool("hydro", "unsplit", true);
  std::cout << "unsplit scheme enabled ?  " << unsplitEnabled << std::endl;
  if (unsplitEnabled) {
    int unsplitVersion = configMap.getInteger("hydro", "unsplitVersion", 0);
    std::cout << "unsplit scheme version :  " << unsplitVersion << std::endl;
  }
  if (!mhdEnabled) {
    if (traceEnabled)
      std::cout << "Use trace computations with version " << traceVersion << std::endl;
    else
      std::cout << "Do not use trace computations" << std::endl;
  }

  // OpenMP enabled ?
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
    if (useGodunov) {  
      HydroRunGodunov * hydroRun = new HydroRunGodunov(configMap);
      hydroRun->setTraceEnabled(traceEnabled);
      hydroRun->setTraceVersion(traceVersion);
      hydroRun->start();
      delete hydroRun;
    } else if (useKurganov) {
      HydroRunKT      * hydroRun = new HydroRunKT(configMap);
      hydroRun->start();
      delete hydroRun;
    } else if (useRelaxingTVD) {
      HydroRunRelaxingTVD * hydroRun = new HydroRunRelaxingTVD(configMap);
      hydroRun->start();
      delete hydroRun;
    }
  } else { // MHD simulations
    MHDRunGodunov * mhdRun = new MHDRunGodunov(configMap);
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
  cout << "solve 2D/3D Euler equations on a cartesian grid." << endl;
  cout << "available method : Godunov, Kurganov, RelaxingTVD." << endl;
  cout << endl; 
  cout << "USAGE:" << endl;
  cout << "--help, -h, --sos" << endl;
  cout << "        get some help about this program." << endl << endl;
  cout << "--param [string]" << endl;
  cout << "        specify configuration parameter file (INI format)" << endl;
  cout << "--scheme [string]" << endl;
  cout << "        string parameter must be godunov or kurganov (numerical scheme)" << endl;
  cout << endl << endl;       
  cout << "--dump-param-file, -d" << endl;
  cout << "        show contents of database that was created by file parser.";
  cout << endl;

} // print_help
