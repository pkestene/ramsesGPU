/**
 * \file testMpiOutputVtk.cpp
 * \brief test routine outputVtk in class HydroRunBaseMpi.
  *
 * \date 16 Oct 2010
 * \author Pierre Kestener
 */

#include <cstdlib>
#include <iostream>

#include "utils/mpiUtils/GlobalMpiSession.h"
#include "hydro/HydroRunBaseMpi.h"

#include "hydro/GetPot.h"
#include "utils/config/ConfigMap.h"

/*
 * test class
 */
class Test : public hydroSimu::HydroRunBaseMpi
{
public:
  Test(ConfigMap &_configMap);
  void runTest();

  void oneStepIntegration(int& nStep, real_t& t, real_t& dt) {};
  void init_test_data();

};

Test::Test(ConfigMap &_configMap) : hydroSimu::HydroRunBaseMpi(_configMap)
{
}

void Test::init_test_data() {

  if (dimType == TWO_D) {
    
    for (int j=0; j<jsize; j++)
      for (int i=0; i<isize; i++) {
	// compute global indexes
	int ii = i + nx*myMpiPos[0];
	int jj = j + ny*myMpiPos[1];

	h_U(i,j,ID)=1.0f*myRank;
	h_U(i,j,IP)=1.0f*(ii+jj);
	h_U(i,j,IU)=1.0f*ii;
	h_U(i,j,IV)=1.0f*jj;
      }
    
  } else {
    
    for (int k=0; k<ksize; k++)
      for (int j=0; j<jsize; j++)
	for (int i=0; i<isize; i++) {
	  // compute global indexes
	  int ii = i + nx*myMpiPos[0];
	  int jj = j + ny*myMpiPos[1];
	  int kk = k + nz*myMpiPos[2];	    
	  
	  h_U(i,j,k,ID)=1.0f*myRank;
	  h_U(i,j,k,IP)=1.0f*(ii+jj+kk);
	  h_U(i,j,k,IU)=1.0f*ii;
	  h_U(i,j,k,IV)=1.0f*jj;
	  h_U(i,j,k,IW)=1.0f*kk;
	}   
    
  }
  
}

void Test::runTest() 
{
  // initialize host data: h_U
  init_test_data();
  
  // dump h_U data into multiple files (one per MPI process using the 
  // VTK parallel vti file format
  outputVtk(h_U,0);

}

/*
 * Make a dummy parameter file test.pot
 */
void make_test_paramFile(std::string fileName) {

  std::fstream outFile;
  outFile.open (fileName.c_str(), std::ios_base::out);

  outFile << "# define a 2D MPI cartesian mesh with mx x my MPI processes\n";
  outFile << "[mpi]\n";
  outFile << "mx=2\n";
  outFile << "my=3\n";
  outFile << "mz=1\n";
  outFile << "\n\n";

  outFile << "# define the 2D problem for each MPI process\n";
  outFile << "[mesh]\n";
  outFile << "nx=50\n";
  outFile << "ny=100\n";
  outFile << "nz=1\n";

  outFile << "# example of VTK output configuration (hand written in raw binary)\n";
  outFile << "[output]\n";
  outFile << "outputVtkAscii=no\n";
  outFile << "outputVtkBase64=no\n";
  outFile << "outputVtkHandWritten=yes\n";
  outFile << "outputVtkCompression=no\n";

  outFile.close();
}

// ================================
// ================================
// ================================
int main(int argc, char * argv[]) {

  /* Initialize MPI session */
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);

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
      std::cout << "[Warning] You did not provide a .ini parameter file !\n";
      std::cout << "[Warning] Using the built-in one\n";
      std::cout << "[Warning] Make test parameter file : " << default_input_file << std::endl;
      make_test_paramFile(default_input_file);
    }
    
    /* search for configuration parameter file */
    const std::string input_file = cl.follow(default_input_file.c_str(),    "--param");
    
    /* parse parameters from input file */
    ConfigMap configMap(input_file);
    
    /* create a Test object */
    Test test(configMap);
    test.runTest();

  } catch (...) {
    
    std::cerr << "Exception caught, something really bad happened...\n\n\n";
    return EXIT_FAILURE;
  
  }

  return EXIT_SUCCESS;

}
