/**
 * \file testHydroMpiParameters.cpp
 * \brief test class HydroMpiParameters
  *
 * \date 7 Oct 2010
 * \author P. Kestener
 */

#include <cstdlib>
#include <iostream>

#include <GlobalMpiSession.h>
#include <HydroMpiParameters.h>

#include <GetPot.h>
#include <ConfigMap.h>

void make_test_paramFile(std::string fileName) {

  std::fstream outFile;
  outFile.open (fileName.c_str(), std::ios_base::out);

  outFile << "# define a 2D MPI cartesian mesh with mx x my MPI processes\n";
  outFile << "[mpi]\n";
  outFile << "mx=2\n";
  outFile << "my=2\n";
  outFile << "mz=1\n";
  outFile << "\n\n";

  outFile << "# define the 2D problem for each MPI process\n";
  outFile << "[mesh]\n";
  outFile << "nx=100\n";
  outFile << "ny=400\n";
  outFile << "nz=1\n";

  outFile.close();
}


int main(int argc, char * argv[]) {

  int myRank, numTasks, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];
  
  /* Initialize MPI session */
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
  hydroSimu::MpiComm worldComm = hydroSimu::MpiComm::world();
  myRank = worldComm.getRank();
  numTasks = worldComm.getNProc();
  MPI::Get_processor_name(processor_name,namelength);

  try {
    
    
    /* 
     * read parameters from parameter file or command line arguments
     */
    
    /* parse command line arguments */
    GetPot cl(argc, argv);

    // default parameter file name
    const std::string default_input_file = std::string(argv[0])+ ".ini";

    /* search for multiple options with the same meaning HELP */
    if( !cl.search(1, "--param") ) {
      std::cout << "[Warning] You did not provide a .pot parameter file !\n";
      std::cout << "[Warning] Using the built-in one\n";
      std::cout << "[Warning] Make test parameter file : " << default_input_file << std::endl;
      make_test_paramFile(default_input_file);
    } 
    
    /* search for configuration parameter file */
    const std::string input_file = cl.follow(default_input_file.c_str(),    "--param");
    
    /* parse parameters from input file */
    ConfigMap configMap(input_file);
    
    /* create HydroMpiParameter object */
    hydroSimu::HydroMpiParameters param(configMap);

    std::cout << "[" << processor_name << "] : MPI proc " << myRank << " out of " << numTasks << std::endl;
    std::cout << "param nx : " << param.nx << std::endl;

  } catch (...) {
    
    std::cerr << "Exception caught, something really bad happened...\n\n\n";
    return EXIT_FAILURE;
  
  }

  return EXIT_SUCCESS;

}
