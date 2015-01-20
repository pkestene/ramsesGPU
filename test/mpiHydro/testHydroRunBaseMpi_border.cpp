/**
 * \file testHydroRunBaseMpi-border.cpp
 * \brief test routine make_boundaries in class HydroRunBaseMpi.
  *
 * \date 18 Oct 2010
 * \author Pierre Kestener
 */

#include <cstdlib>
#include <iostream>

#include <GlobalMpiSession.h>
#include <HydroRunBaseMpi.h>

#include <GetPot.h>
#include <ConfigMap.h>

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
	h_U(i,j,ID)=1.0f*myRank;
	h_U(i,j,IP)=0.0f;
	h_U(i,j,IU)=1.0f;
	h_U(i,j,IV)=2.0f;
      }
    
  } else {
    
    for (int k=0; k<ksize; k++)
      for (int j=0; j<jsize; j++)
	for (int i=0; i<isize; i++) {
	  // fill density, U, V, W and pressure sub-arrays
	  h_U(i,j,k,ID)=1.0f*myRank;
	  h_U(i,j,k,IP)=0.0f;
	  h_U(i,j,k,IU)=1.0f;
	  h_U(i,j,k,IV)=2.0f;
	  h_U(i,j,k,IW)=3.0f;
	}   
    
  }
  
}

void Test::runTest() 
{
  // initialize host data: h_U
  init_test_data();

  // call make boundaries
  make_boundaries(h_U,XDIR);
  make_boundaries(h_U,YDIR);
  
  // dump h_U data into multiple files : one per MPI process using the 
  // XSM file format, and including ghost borders
  outputXsm(h_U,0,ID,true);
  outputXsm(h_U,0,IU,true);

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
  // periodic boundary conditions
  outFile << "boundary_xmin=1\n";
  outFile << "boundary_xmax=1\n";
  outFile << "boundary_ymin=3\n";
  outFile << "boundary_ymax=3\n";
  outFile << "boundary_zmin=3\n";
  outFile << "boundary_zmax=3\n";
  outFile << "\n\n";
  
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
    const std::string default_input_file = std::string(argv[0])+ ".pot";

    /* search for multiple options with the same meaning HELP */
    if( !cl.search(1, "--param") ) {
      std::cout << "[Warning] You did not provide a .pot parameter file !\n";
      std::cout << "[Warning] Using the built-in one\n";
      
      // make sure we have a test.pot file in current directory
      std::cout << "Make test parameter file : " << default_input_file << std::endl;
      make_test_paramFile(default_input_file);
    }
    
    /* search for configuration parameter file */
    const std::string input_file = cl.follow(default_input_file.c_str(),    "--param");
    
    /* parse parameters from input file */
    ConfigMap configMap(input_file.c_str());
    
    /* create a Test object */
    Test test(configMap);
    test.runTest();

  } catch (...) {
    
    std::cerr << "Exception caught, something really bad happened...\n\n\n";
    return EXIT_FAILURE;
  
  }

  return EXIT_SUCCESS;

}
