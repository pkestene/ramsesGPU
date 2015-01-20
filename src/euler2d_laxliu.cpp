/**
 * \file euler2d_laxliu.cpp
 * \brief
 * Solve 2D Euler equation on a cartesian grid using positive scheme (Lax-Liu).
 *
 * \date 21/01/2010
 * \author P. Kestener
 *
 * $Id: euler2d_laxliu.cpp 3452 2014-06-17 10:09:48Z pkestene $
 */

#include <cstdlib>
#include <iostream>

#include "HydroRunLaxLiu.h"

#include "utils/monitoring/date.h"

/** Parse configuration parameter file */
#include "GetPot.h"

/* Graphics Magick C++ API to dump PNG image files */
#ifdef USE_GM
#include <Magick++.h>
#endif // USE_GM

void print_help(int argc, char* argv[]);

using hydroSimu::HydroRunLaxLiu;

/***********************************
 ***********************************/
int main (int argc, char * argv[]) {

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

  /* dump input file and exit */
  if( cl.search(2, "--dump-param-file", "-d") ) {
    std::cout << configMap << std::endl;
    exit(0);
  }  

  /* 
   * process parameter file when constructing HydroParameter object;
   * initialize hydro grid.
   */
  HydroRunLaxLiu * hydroRun = new HydroRunLaxLiu(configMap);

  /*
   * print date to be the first item printed in log
   */
  std::cout << "#######################################################" << std::endl;
  std::cout << "##### Hydro simulations (Lax-Liu positive scheme) #####" << std::endl;
  std::cout << "#######################################################" << std::endl;
  std::cout << "time : " << current_date() << std::endl;

  // when using PNG output, we must initialize GraphicsMagick library
#ifdef USE_GM
  Magick::InitializeMagick("");
#endif

  /*
   * Main loop.
   */
  hydroRun->start();

  /*
   * memory free
   */
  delete hydroRun;
  //deletea par;
  return EXIT_SUCCESS;

}

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void print_help(int argc, char* argv[]) {
  
  (void) argc;
  
  using std::cerr;
  using std::cout;
  using std::endl;

  cerr << endl;
  cout << argv[0] << ": " << endl;
  cout << "solve 2D Euler equations on a cartesian grid using Lax-Liu positive scheme." << endl;
  cerr << endl; 
  cerr << "USAGE:" << endl;
  cerr << "--help, -h, --sos" << endl;
  cerr << "        get some help about this program." << endl << endl;
  cerr << "--param [string]" << endl;
  cerr << "        specify configuration parameter file (INI format)" << endl;
  cerr << endl << endl;       
  cerr << "--dump-param-file, -d" << endl;
  cerr << "        show contents of database that was created by file parser.";
  cerr << endl;
}
