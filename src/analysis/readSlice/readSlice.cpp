/**
 * \file readSlice.cpp
 * \brief
 * MPI app (Parallel-NetCDF) to read a slice in a large 3D netcdf file.
 * 
 * example of use:
 * mpirun -np 1 readSlice_double -i test.nc -o test.npy -x --slice 12
 *
 * \date 2 July 2013
 * \author P. Kestener
 *
 * $Id: readSlice.cpp 2957 2013-08-26 15:08:40Z pkestene $ 
 */

#include <cstdlib>
#include <iostream>

#include "utils/cnpy/cnpy.h"

#include <hydro/Arrays.h>
#include <hydro/real_type.h>
#include <hydro/constants.h>

#include <utils/mpiUtils/MpiComm.h>
#include <utils/mpiUtils/GlobalMpiSession.h>

#include <utils/monitoring/date.h>

/** Parse configuration parameter file */
#include "hydro/GetPot.h"
#include "utils/config/ConfigMap.h"


void print_help(int argc, char* argv[]);

using hydroSimu::HostArray;
using hydroSimu::GlobalMpiSession;
using hydroSimu::MpiComm;


void read_PnetCDF(const std::string  filename, 
		  int                sliceNum,
		  int                sliceDir,
		  int                varNum,
		  HostArray<real_t> &U);

// for Parallel-netCDF support
#ifdef USE_PNETCDF
#include <pnetcdf.h>

#define PNETCDF_HANDLE_ERROR {                        \
    if (err != NC_NOERR)                              \
        printf("Error at line %d (%s)\n", __LINE__,   \
               ncmpi_strerror(err));                  \
}

#endif // USE_PNETCDF

// slice direction
enum sliceDir {
  X_SLICE=0,
  Y_SLICE=1,
  Z_SLICE=2
};


/************************************
 ************************************/
int main (int argc, char * argv[]) {

  int myRank, numTasks, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];

  /* Initialize MPI session */
  GlobalMpiSession mpiSession(&argc,&argv);
  hydroSimu::MpiComm worldComm = hydroSimu::MpiComm::world();
  myRank = worldComm.getRank();
  numTasks = worldComm.getNProc();
  MPI_Get_processor_name(processor_name,&namelength);

  if ( numTasks != 1 ) {
    std::cerr << "App must use only one processor...\n";
    return EXIT_FAILURE;
  }

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

  // check for input / output filenames
  const std::string  inFile        = cl.follow("", 2, "-i","--input"); 
  const std::string  outFile       = cl.follow("", 2, "-o","--output"); 
  
  // check for slice parameters
  const int          sliceNum      = cl.follow(0,  2, "-n","--slice");
  const bool         sliceXEnabled = cl.search(1, "-x");
  const bool         sliceYEnabled = cl.search(1, "-y");
  const bool         sliceZEnabled = cl.search(1, "-z");
  const int          varNum        = cl.follow(-1,  2, "-v","--var");

  // if varNum = -1, means we read them all, else we read only one identified
  // by varNum

  // sanity checks
  if (inFile.size() == 0 or outFile.size() == 0) {
    
    if (MpiComm::world().getRank() == 0)
      std::cerr << "Please specify input and output filename !!!\n";

    return EXIT_FAILURE;

  }
  
  /*
   * print date to be the first item printed in log
   */
  if (myRank == 0) {
    std::cout << "#####################################" << std::endl;
    std::cout << "########### Convert File ############" << std::endl;
    std::cout << "#####################################" << std::endl;
    std::cout << "Current date : " << current_date() << std::endl;
    std::cout << "Using " << numTasks << " MPI tasks"    << std::endl;
  }

  // sliceDir
  int sliceDir=-1;
  if (sliceXEnabled) sliceDir = X_SLICE;
  if (sliceYEnabled) sliceDir = Y_SLICE;
  if (sliceZEnabled) sliceDir = Z_SLICE;

  // data
  HostArray<real_t> U;

  // read NetCDF file
  read_PnetCDF(inFile, sliceNum, sliceDir, varNum, U);

  // save slice in npy format (default behaviour is to use row-major in numpy)
  const unsigned int shape[4] = { U.nvar(), U.dimz(), U.dimy(), U.dimx() };
  cnpy::npy_save(outFile.c_str(), U.data(), shape, 4);

  return 0;

} // end main


// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void print_help(int argc, char* argv[]) {
  
  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << "[" << argv[0] << "] read a NetCDF file, extract slice and dump in npy format  : " << endl;
  cout << endl; 
  cout << "USAGE:" << endl;
  cout << "--help, -h, --sos" << endl;
  cout << "        get some help about this program." << endl << endl;
  cout << "Examples of use:\n";
  cout << "Read all variables:\n";
  cout << "  readSlice_double -i test.nc -o test.npy -x --slice 12\n";
  cout << "If you only want to read density (var=0):\n";
  cout << "  readSlice_double -i test.nc -o test.npy -x --slice 12 -v 0\n";
  cout << endl;

} // print_help

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void read_PnetCDF(const std::string  filename, 
		  int                sliceNum,
		  int                sliceDir,
		  int                varNum,
		  HostArray<real_t> &U)
{

#ifdef USE_PNETCDF
  MpiComm communicator = MpiComm::world();
  int myRank   = communicator.getRank();
  int numTasks = communicator.getNProc();

  // netcdf file id
  int ncFileId;
  int err;
  
  // file creation mode
  int ncOpenMode = NC_NOWRITE;
  
  // verbose log ?
  bool pnetcdf_verbose = true;
  
  int varIds[8];
  MPI_Offset starts[3], counts[3], read_size, sum_read_size;
  //MPI_Info mpi_info_used;
  //char str[512];
  
  // time measurement variables
  double read_timing, max_read_timing, read_bw;
  
  // sizes to read
  int nx, ny, nz;  // logical sizes / per sub-domain
  
  // measure time ??
  if (pnetcdf_verbose) {
    MPI_Barrier( communicator.getComm() );
    read_timing = MPI_Wtime();
  }
  
  /* 
   * Open NetCDF file
   */
  err = ncmpi_open( communicator.getComm(), filename.c_str(), 
		    ncOpenMode,
		    MPI_INFO_NULL, &ncFileId);
  if (err != NC_NOERR) {
    printf("Error: ncmpi_open() file %s (%s)\n",filename.c_str(),ncmpi_strerror(err));
    MPI_Barrier( communicator.getComm() );
    MPI_Abort( communicator.getComm() , -1 );
  }
  
  /*
   * Query NetCDF mode
   */
  if (pnetcdf_verbose) {
    int NC_mode;
    err = ncmpi_inq_version(ncFileId, &NC_mode);
    if (myRank==0) {
      if (NC_mode == NC_64BIT_DATA)
	std::cout << "Pnetcdf Input mode : NC_64BIT_DATA (CDF-5)\n";
      else if (NC_mode == NC_64BIT_OFFSET)
	std::cout << "Pnetcdf Input mode : NC_64BIT_OFFSET (CDF-2)\n";
      else
	std::cout << "Pnetcdf Input mode : unknown\n";
    }
  }

  /* Query dimId : remember that here x is the slowest dimension */
  int dimId;
  MPI_Offset dimX, dimY, dimZ; // global sizes
  err = ncmpi_inq_dimid(ncFileId, "x", &dimId);
  err = ncmpi_inq_dimlen(ncFileId, dimId, &dimX);

  err = ncmpi_inq_dimid(ncFileId, "y", &dimId);
  err = ncmpi_inq_dimlen(ncFileId, dimId, &dimY);

  err = ncmpi_inq_dimid(ncFileId, "z", &dimId);
  err = ncmpi_inq_dimlen(ncFileId, dimId, &dimZ);

  if (myRank==0) {
    std::cout << "Input file dimensions: " << dimX
	      << " " << dimY
	      << " " << dimZ << std::endl;

    if ( (sliceDir == X_SLICE) and (sliceNum>=dimX or sliceNum<0))
      std::cerr << "PnetCDF read slice error; sliceNum not in X range\n";

    if ( (sliceDir == Y_SLICE) and (sliceNum>=dimY or sliceNum<0))
      std::cerr << "PnetCDF read slice error; sliceNum not in X range\n";

    if ( (sliceDir == Z_SLICE) and (sliceNum>=dimZ or sliceNum<0))
      std::cerr << "PnetCDF read slice error; sliceNum not in X range\n";

  }


  /* Set geometry */
  // local sizes
  nx = dimZ;
  ny = dimY;
  nz = dimX;
  
  // default values
  counts[IX] = nz;
  counts[IY] = ny;
  counts[IZ] = nx;
  starts[IX] = 0;
  starts[IY] = 0;
  starts[IZ] = 0;

  if (sliceDir == X_SLICE) {
    counts[IX]=1;
    starts[IX]=sliceNum;
  } else if (sliceDir == Y_SLICE) {
    counts[IY]=1;
    starts[IY]=sliceNum;
  } else if (sliceDir == Z_SLICE) {
    counts[IZ]=1;
    starts[IZ]=sliceNum;
  } else {
    std::cerr << "[readSlice] Error, we shouldn't be here !\n";
  }

  int nItems = counts[IX]*counts[IY]*counts[IZ];

  //std::cout << "myRank : " << myRank << " " << nz << " " << nzi << "\n";

  /* Query number of variables */
  int nbVar = 0;
  err = ncmpi_inq_nvars(ncFileId, &nbVar);
  if (myRank==0) {
    std::cout << "Number of variables: " << nbVar << std::endl;
  }

  /* Query variables Ids */
  err = ncmpi_inq_varid(ncFileId, "rho",    &varIds[ID]);
  err = ncmpi_inq_varid(ncFileId, "E",      &varIds[IP]);
  err = ncmpi_inq_varid(ncFileId, "rho_vx", &varIds[IU]);
  err = ncmpi_inq_varid(ncFileId, "rho_vy", &varIds[IV]);
  err = ncmpi_inq_varid(ncFileId, "rho_vz", &varIds[IW]);
  err = ncmpi_inq_varid(ncFileId, "Bx",     &varIds[IA]);
  err = ncmpi_inq_varid(ncFileId, "By",     &varIds[IB]);
  err = ncmpi_inq_varid(ncFileId, "Bz",     &varIds[IC]);

  /* Query var type */
  nc_type varType;
  MPI_Datatype mpiDataType;
  err = ncmpi_inq_vartype(ncFileId, varIds[ID], &varType);
  if (varType == NC_FLOAT) {
    mpiDataType = MPI_FLOAT;
    if (sizeof(real_t) != sizeof(float)) {
      if (myRank==0)
	std::cerr << "Wrong data type : sizeof(real_t) != sizeof(float) !!!" << std::endl;
      MPI_Barrier( communicator.getComm() );
      MPI_Abort( communicator.getComm() , -1 );
    }
  }
  if (varType == NC_DOUBLE) {
    mpiDataType = MPI_DOUBLE;
    if (sizeof(real_t) != sizeof(double)) {
      if (myRank==0)
	std::cerr << "Wrong data type : sizeof(real_t) != sizeof(double) !!!" << std::endl;
      MPI_Barrier( communicator.getComm() );
      MPI_Abort( communicator.getComm() , -1 );
    }
  }

  if (pnetcdf_verbose) {
    if (myRank == 0) {
      if (varType == NC_FLOAT)
	std::cout << "using NC_FLOAT"  << std::endl; 
      if (varType == NC_DOUBLE)
	std::cout << "using NC_DOUBLE" << std::endl; 
    }
  }

  /* memory allocation */
  if (varNum > -1) // only read one var
    U.allocate( make_uint4(counts[IZ],counts[IY],counts[IX],1) );
  else
    U.allocate( make_uint4(counts[IZ],counts[IY],counts[IX],nbVar) );

  /* read variables */
  if (varNum > -1) {
      real_t* data;
      data = &(U(0,0,0,0));
      err = ncmpi_get_vara_all(ncFileId, varIds[varNum], 
			       starts, counts, data, nItems, mpiDataType);
      PNETCDF_HANDLE_ERROR;
  } else {
    for (int iVar=0; iVar<nbVar; iVar++) {
      real_t* data;
      data = &(U(0,0,0,iVar));
      err = ncmpi_get_vara_all(ncFileId, varIds[iVar], 
			       starts, counts, data, nItems, mpiDataType);
      PNETCDF_HANDLE_ERROR;
      if (myRank == 0) std::cout << "varId " << iVar << "read - OK\n";
    } // end for loop reading heavy data
  }

  /* 
   * close the file 
   */
  err = ncmpi_close(ncFileId);
  PNETCDF_HANDLE_ERROR;
  
  /*
   * verbose log about memory bandwidth
   */
  if (pnetcdf_verbose) {
    
    read_timing = MPI_Wtime() - read_timing;
    
    //read_size = nbVar * U.section() * sizeof(real_t);
    read_size = U.sizeBytes();
    sum_read_size = read_size *  numTasks;
    
    MPI_Reduce(&read_timing, &max_read_timing, 1, MPI_DOUBLE, MPI_MAX, 0, communicator.getComm());
    
    if (myRank==0) {
      printf("########################################################\n");
      printf("############## Parallel-netCDF bandwidth ###############\n");
      printf("########################################################\n");
      printf("Local  array size %d x %d x %d reals(%lu bytes), read size = %.2f MB\n",
	     nx,
	     ny,
	     nz,
	     sizeof(real_t),
	     1.0*read_size/1048576.0);
      sum_read_size /= 1048576.0;
      printf("Global array size %lld x %lld x %lld reals(%lu bytes), read size = %.2f GB\n",
	     dimX,
	     dimY,
	     dimZ,
	     sizeof(real_t),
	     1.0*sum_read_size/1024);
      
      read_bw = sum_read_size/max_read_timing;
      printf(" procs    Global array size  exec(sec)  read(MB/s)\n");
      printf("-------  ------------------  ---------  -----------\n");
      printf(" %4d    %4lld x %4lld x %4lld %8.2f  %10.2f\n", numTasks,
	     dimX,
	     dimY,
	     dimZ,
	     max_read_timing, read_bw);
      printf("########################################################\n");
      
    } // end (myRank==0)
    
  } // pnetcdf_verbose
  
  if (pnetcdf_verbose and myRank==0) std::cout << "Input file read OK\n";
  
#else
  
  if (MpiComm::world().getRank() == 0) {
    
    std::cerr << "Please enable PnetCDF ..." << std::endl;
    
  }
  
#endif

} // read_PnetCDF
