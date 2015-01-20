/**
 * \file projectedDensityMpi.cpp
 *
 * This file contains a simple programs to compute 2D projected density of a large
 * 3D data array stored in a netcdf file (use Parallel-NetCDF to load data).
 * Projection along Z into (X,Y) plane.
 * PNetCDF reading routine removes ghost zones; so we read only the inner sub-domain.
 *
 * \warning here MPI domain decomposition only along Z axis; each MPI task read
 * a z-slab sub-domain.
 *
 * \author P. Kestener
 * \date 26/08/2013
 *
 * $Id: projectedDensityMpi.cpp 2957 2013-08-26 15:08:40Z pkestene $
 */

#include <math.h>
#include <iostream>
#include <fstream>

#include <mpi.h>

#include <GetPot.h>
#include <ConfigMap.h>
#include <cnpy.h>

#include <Arrays.h>
using hydroSimu::HostArray;

#include "constants.h"

#include "pnetcdf_io.h"

void print_help(int argc, char* argv[]);

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
int main(int argc, char **argv){

#ifndef USE_PNETCDF
  std::cout << "Parallel-NetCDF is not available; please enable to build this application\n";
  return 0;
#endif // USE_PNETCDF


#if defined(USE_PNETCDF)
 
  /* parse command line arguments */
  GetPot cl(argc, argv);
  
  /* search for multiple options with the same meaning HELP */
  if( cl.search(3, "--help", "-h", "--sos") ) {
    print_help(argc,argv);
    exit(0);
  }
  
  /* set default configuration parameter fileName */
  const std::string default_param_file = "projDens.ini";
  const std::string param_file = cl.follow(default_param_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(param_file);
  
  const std::string input_file  = cl.follow("test.nc", 2, "-i", "--in");
  const std::string output_file = cl.follow("projD",   2, "-o", "--out");

  /* ******************************************* */
  int myRank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // check input file 
  if (input_file.size() == 0) {
    std::cout << "Wrong input.\n";
  } else {
    if (myRank==0) std::cout << "input file used : " << input_file << std::endl;
  }


  /* 
   * Sanity check
   */
  /* read mpi geometry */
  // mpi geometry
  int mx,my,mz;
  mx=configMap.getInteger("mpi","mx",1);
  my=configMap.getInteger("mpi","my",1);
  mz=configMap.getInteger("mpi","mz",1);

  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);
  if (mx*my*mz != nbMpiProc || mx!=1 || my!=1) {
    std::cout << "Invalid configuration : check parameter file\n";
    return -1;
  }

  
  /*
   * Read data
   */
  // read local domain size
  int nx=configMap.getInteger("mesh","nx",32);
  int ny=configMap.getInteger("mesh","ny",32);
  int nz=configMap.getInteger("mesh","nz",32);

  int NX=nx*mx, NY=ny*my, NZ=nz*mz;

  HostArray<double> data_read;
  data_read.allocate(make_uint4(nx, ny, nz, 1));

  // ////////////////////
  // projected density
  // ////////////////////
  {
  
    // read data (density - ID)
    read_pnetcdf(input_file,ID,configMap,data_read);
    HostArray<double> &rho = data_read;
    
    // local projected density
    HostArray<double> projDensLoc;
    projDensLoc.allocate(make_uint4(nx,ny,1,1));
    projDensLoc.reset();

    // global projected density
    HostArray<double> projDensGlob;
    projDensGlob.allocate(make_uint4(nx,ny,1,1));
    projDensGlob.reset();

    // compute local projetion
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++) {
	  projDensLoc(i,j,0,0) += rho(i,j,k,0);
	}
    
    // gather all result by performing reduce
    MPI_Reduce(projDensLoc.data(), projDensGlob.data(), nx*ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // normalize global projected density
    if (myRank == 0) {
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++) {
	  projDensGlob(i,j,0,0) /= (mz*nz);      
	}
    }
    
    // save global data to file
    if (myRank==0) {
      // save array projDensGlob to file (default behaviour is to use row-major in numpy)
      const unsigned int shape[] = {(unsigned int) ny, (unsigned int) nx};
      cnpy::npy_save(output_file.c_str(),projDensGlob.data(),shape,2,"w");
      
      std::cout << "projected density computed and saved...\n";
      
    }
  
  } // end compute projected density

  MPI_Finalize();

  if (myRank==0) printf("MPI finalized...\n");

  return 0;

#endif // defined(USE_PNETCDF)

} // main

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void print_help(int argc, char* argv[]) {
  
  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << "[" << argv[0] << "] read a NetCDF file, compute 2D projected density and dump in npy format  : " << endl;
  cout << endl; 
  cout << "USAGE:" << endl;
  cout << "--help, -h, --sos" << endl;
  cout << "        get some help about this program." << endl << endl;
  cout << "Examples of use:\n";
  cout << "  " << argv[0] << " -i data.nc -o projDens.npy --param param.ini\n";
  cout << endl;

} // print_help

