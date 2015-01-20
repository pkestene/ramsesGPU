/**
 * \file test_write_pnetcdf.cpp
 *
 * \author P. Kestener
 * \date April, 14 2014
 *
 * $Id: test_pnetcdf_write.cpp 3404 2014-05-22 10:43:59Z pkestene $
 */

#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>     // numeric limits

#include <GetPot.h>
#include <ConfigMap.h>

#include <Arrays.h>
using hydroSimu::HostArray;

#include "constants.h"

#include "pnetcdf_io.h"

void init_array(HostArray<double> &data, double val);

/** ------------------------- randnum --------------------------------
 ** returns a random number in [0;1]
 ** ------------------------------------------------------------------ */
inline double randnum(void)
{
  return static_cast<double>(rand())/static_cast<double>(RAND_MAX);
}

/** ------------------------- randnum_int -----------------------------
 ** returns a random number in [0, N [
 ** assumes N < RAND_MAX (= 2147483647 = 2^31-1)
 ** ------------------------------------------------------------------ */
inline int randnum_int(int N)
{
  return rand() % N;
}

/* ####################################### */
/* ####################################### */
/* ####################################### */
int main(int argc, char **argv){

#ifndef USE_PNETCDF
  std::cout << "Parallel-NetCDF is not available; please enable to build this application\n";
  return 0;

#else
 
  /* ******************************************* */
  int myRank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  int nbMpiProc;
  MPI_Comm_size(MPI_COMM_WORLD, &nbMpiProc);

  /* parse command line arguments */
  GetPot cl(argc, argv);
  
  /* set default configuration parameter fileName */
  const std::string default_param_file = "test_pnetcdf.ini";
  const std::string param_file = cl.follow(default_param_file.c_str(),    "--param");

  /* parse parameters from input file */
  ConfigMap configMap(param_file);
  
  /* 
   * Sanity check
   */
  // read mpi geometry
  int mx,my,mz;
  mx=configMap.getInteger("mpi","mx",1);
  my=configMap.getInteger("mpi","my",1);
  mz=configMap.getInteger("mpi","mz",1);
  
  if (mx*my*mz != nbMpiProc) {
    if (myRank==0) std::cerr << "Invalid configuration : check parameter file\n";
    if (myRank==0) std::cerr << "mx*my*mz must be equal to the number of MPI proc (3D domain decomp).\n";
    return -1;
  }
  if (myRank==0) std::cout << "Use a 3D domain decomposition with mx=" << mx 
			   << ", my=" << my 
			   << ", mz=" << mz << "\n";
  
  
  /*
   * Read parameter file
   */
  // read local domain sizes
  int nx=configMap.getInteger("mesh","nx",32);
  int ny=configMap.getInteger("mesh","ny",32);
  int nz=configMap.getInteger("mesh","nz",32);

  // global sizes
  int NX=nx*mx, NY=ny*my, NZ=nz*mz;
  
  int ghostWidth = configMap.getInteger("mesh","ghostWidth",3);
  
  // MPI cartesian coordinates
  // myRank = mpiCoord[0] + mx*mpiCoord[1] + mx*my*mpiCoord[2]
  int mpiCoord[3];
  {
    mpiCoord[2] =  myRank/(mx*my);
    mpiCoord[1] = (myRank - mx*my*mpiCoord[2])/mx;
    mpiCoord[0] =  myRank - mx*my*mpiCoord[2] -mx*mpiCoord[1];
  }

  /* ******************************************* */
  /*               Initialization                */
  /* ******************************************* */

  bool ghostIncluded = configMap.getBool("output", "ghostIncluded",false);

  // main data 
  HostArray<double> data_local; // 8 variables: rho, rho*vx, rho*vy, rho*vz

  int local_sizes[3];
  local_sizes[0] = nx;
  local_sizes[1] = ny;
  local_sizes[2] = nz;

  if ( ghostIncluded ) {
    local_sizes[0] += 2*ghostWidth;
    local_sizes[1] += 2*ghostWidth;
    local_sizes[2] += 2*ghostWidth;
  }

  data_local.allocate(make_uint4(local_sizes[0], 
				 local_sizes[1], 
				 local_sizes[2], 
				 8));
  
  // global size
  int gsizes[3];
  gsizes[IZ] = NX;
  gsizes[IY] = NY;
  gsizes[IX] = NZ;
  
  if ( ghostIncluded ) {
    gsizes[IZ] += 2*ghostWidth;
    gsizes[IY] += 2*ghostWidth;
    gsizes[IX] += 2*ghostWidth;
  }

  // writing parameter (offset and size)
  MPI_Offset         starts[3] = {0};
  MPI_Offset         counts[3] = {nz, ny, nx};
  
  // take care that row-major / column major format
  starts[IZ] = mpiCoord[IX]*nx;
  starts[IY] = mpiCoord[IY]*ny;
  starts[IX] = mpiCoord[IZ]*nz;
 
  if ( ghostIncluded ) {

    if ( mpiCoord[IX] == 0 )
      counts[IZ] += ghostWidth;
    if ( mpiCoord[IY] == 0 )
      counts[IY] += ghostWidth;
    if ( mpiCoord[IZ] == 0 )
      counts[IX] += ghostWidth;

    if ( mpiCoord[IX] == mx-1 )
      counts[IZ] += ghostWidth;
    if ( mpiCoord[IY] == my-1 )
      counts[IY] += ghostWidth;
    if ( mpiCoord[IZ] == mz-1 )
      counts[IX] += ghostWidth;

    starts[IZ] += ghostWidth;
    starts[IY] += ghostWidth;
    starts[IX] += ghostWidth;

    if ( mpiCoord[IX] == 0 )
      starts[IZ] -= ghostWidth;
    if ( mpiCoord[IY] == 0 )
      starts[IY] -= ghostWidth;
    if ( mpiCoord[IZ] == 0 )
      starts[IX] -= ghostWidth;
  
  }
  
  // for (int iRank=0; iRank<nbMpiProc; iRank++) {
  //   if ( iRank == myRank)
  //     printf("[MPI %d] starts : %2d %2d %2d counts : %2d %2d %2d\n",myRank,starts[IZ],starts[IY],starts[IX], counts[IZ], counts[IY], counts[IX]);
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // initialize local data
  init_array(data_local, myRank);

  // write data
  std::string output_file = configMap.getString("output", "outputFile", "./data.nc");
  write_pnetcdf(output_file,starts,counts,gsizes,data_local,configMap);
      
  if (myRank==0) printf("MPI finalize...\n");
  
  MPI_Finalize();
  
  return 0;

#endif // defined(USE_PNETCDF)

} // end main

/*
 *
 */
void init_array(HostArray<double> &data, double val)
{

  int nbVar = data.nvar();

  for (int iVar=0; iVar<nbVar; iVar++) {

    for (unsigned int i=0; i<data.section(); i++) {
      
      data(i+iVar*data.section()) = val*val*(iVar+1)*(iVar+2);
      
    }

  }

} // end init_array
