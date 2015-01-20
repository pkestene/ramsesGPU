/**
 * \file ioConvert.cpp
 * \brief
 * MPI app to convert large 3D netcdf file into HDF5 file using MPI collective IO.
 * 
 * example of use:
 * mpirun -np 4 ioConvert -i test.nc -o test.h5
 *
 * \date 28 Feb 2013
 * \author P. Kestener
 *
 * $Id: ioConvert.cpp 3529 2014-08-29 16:34:41Z pkestene $ 
 */

#include <cstdlib>
#include <iostream>

#include <hydro/Arrays.h>
#include <hydro/real_type.h>
#include <hydro/constants.h>

#include <utils/mpiUtils/MpiComm.h>
#include <utils/mpiUtils/GlobalMpiSession.h>

#include <utils/monitoring/date.h>
#include <utils/cnpy/cnpy.h>

/** Parse configuration parameter file */
#include "hydro/GetPot.h"
#include "utils/config/ConfigMap.h"


void print_help(int argc, char* argv[]);

using hydroSimu::HostArray;
using hydroSimu::GlobalMpiSession;
using hydroSimu::MpiComm;


void read_PnetCDF(const std::string filename, 
		  HostArray<real_t> &U,
		  int &dimZGlob,
		  int &timeStep,
		  double &totalTime);
void write_HDF5  (const std::string filename, 
		  HostArray<real_t> &U,
		  int dimZGlob,
		  int timeStep,
		  double totalTime);

void write_Xdmf (const std::string xdmfFilename, 
		 const std::string hdf5Filename, 
		 int dimX,
		 int dimY,
		 int dimZ,
		 int timeStep,
		 double totalTime,
		 bool mhdEnabled);

void reduce (const HostArray<real_t> &U, 
	     HostArray<real_t> &Ureduce,
	     int reduceParam);

void projected_density(const HostArray<real_t> &U, 
		       HostArray<double> &Uproj);

// for HDF5 file format output
#ifdef USE_HDF5
#include <hdf5.h>
#endif // USE_HDF5

// for Parallel-netCDF support
#ifdef USE_PNETCDF
#include <pnetcdf.h>

#define PNETCDF_HANDLE_ERROR {                        \
    if (err != NC_NOERR)                              \
        printf("Error at line %d (%s)\n", __LINE__,   \
               ncmpi_strerror(err));                  \
}

#endif // USE_PNETCDF

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
  const std::string  inFile  = cl.follow("", 2, "-i","--input"); 
  const std::string  outFile = cl.follow("", 2, "-o","--output"); 

  if (inFile.size() == 0 or outFile.size() == 0) {
    
    if (MpiComm::world().getRank() == 0)
      std::cerr << "Please specify input and output filename !!!\n";

    return EXIT_FAILURE;

  }

  // do we want to reduce output size by 2 or 4 ?
  bool reduceSize = false;
  if ( cl.search(1, "--reduce") )
    reduceSize = true;
  int reduceParam = cl.follow(2, 1,"--reduce"); 

  if (reduceParam !=2 and reduceParam !=4) {
    if (myRank==0) {
      std::cerr << "reduceParam must be set to 2 or 4 !\n";
      std::cerr << "defaulting to 2\n";
      reduceParam=2;
    }
  }

  bool projDensity = false;
  if ( cl.search(1, "--proj") )
    projDensity = true;

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

  // data
  HostArray<real_t> U;
  int dimZGlob;
  int timeStep;
  double totalTime;

  /* 
   * Read input file.
   */
  read_PnetCDF(inFile, U, dimZGlob, timeStep, totalTime);

  if (projDensity) {

    int nx    = U.dimx();
    int ny    = U.dimy();

    HostArray<double> Uproj;
    Uproj.allocate( make_uint4(nx,ny,1,1) );

    projected_density(U, Uproj);

    HostArray<double> UprojTotal;
    UprojTotal.allocate( make_uint4(nx,ny,1,1) );
    
    // mpi reduce
    MPI_Reduce(Uproj.data(), UprojTotal.data(), nx*ny, MPI_DOUBLE, MPI_SUM, 0, worldComm.getComm() );

    if (myRank==0) {
      // replace .h5 by .npy
      std::string outFileNpy = outFile;
      outFileNpy.replace(outFileNpy.end()-3,outFileNpy.end(),".npy");
      
      {
    	//mpirun --mca btl sm,self --mca mtl ^psm -np 4 ../utils/IO/ioConvert -i turbulence_mhd_cpu_0000100.nc -o turbulence_mhd_cpu_0000100_projDensity.h5  --proj
    	const unsigned int shape[] = {ny,nx};
    	cnpy::npy_save(outFileNpy.c_str(),UprojTotal.data(),shape,2,"w");
      }
    }


  } else {

    /*
     * Write output file.
     */
    if (reduceSize) {
      int nx = U.dimx();
      int ny = U.dimy();
      int nz = U.dimz();
      int nbVar = U.nvar();
    
      HostArray<real_t> Ureduce;
      Ureduce.allocate( make_uint4(nx/reduceParam,ny/reduceParam,nz/reduceParam,nbVar) );

      reduce(U,Ureduce,reduceParam);
      write_HDF5(outFile, Ureduce, dimZGlob/reduceParam, timeStep, totalTime);

    } else {

      write_HDF5(outFile, U      , dimZGlob  , timeStep, totalTime);

    }

    if (myRank==0) {
      int nx = U.dimx();
      int ny = U.dimy();
      int nz = dimZGlob;
    
      if (reduceSize) {

	nx /= reduceParam;
	ny /= reduceParam;
	nz /= reduceParam;
      }
      // replace .h5 by .xdmf
      std::string outFileXdmf = outFile;
      outFileXdmf.replace(outFileXdmf.end()-3,outFileXdmf.end(),".xdmf");
      write_Xdmf (outFileXdmf,
		  outFile,
		  nx,
		  ny,
		  nz,
		  timeStep,
		  totalTime,
		  true);
    }

  }

  return EXIT_SUCCESS;

} // main

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void print_help(int argc, char* argv[]) {
  
  using std::cerr;
  using std::cout;
  using std::endl;

  cout << endl;
  cout << argv[0] << " convert NetCDF file into HDF using collective IO : " << endl;
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

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void read_PnetCDF(const std::string filename, 
		  HostArray<real_t> &U,
		  int &dimZGlob,
		  int &timeStep,
		  double &totalTime)
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
  int nx, ny, nz, nzi;  // logical sizes / per sub-domain
  
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
  }

  /* Set geometry */
  // local sizes
  nx = dimZ;
  ny = dimY;
  nzi = dimX/numTasks;
  
  dimZGlob=dimX;

  /* take care that dimZ/numTasks is not integer (so round to match dimZ) */
  if ( myRank == (numTasks-1) )
    nz = dimX-(numTasks-1)*nzi;
  else
    nz = nzi;

  counts[IX] = nz;
  counts[IY] = ny;
  counts[IZ] = nx;
  
  int nItems = counts[IX]*counts[IY]*counts[IZ];

  starts[IX] = myRank*nzi;
  starts[IY] = 0;
  starts[IZ] = 0;

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
  U.allocate( make_uint4(nx,ny,nz,nbVar) );

  /* read variables */
  for (int iVar=0; iVar<nbVar; iVar++) {
    real_t* data;
    data = &(U(0,0,0,iVar));
    err = ncmpi_get_vara_all(ncFileId, varIds[iVar], 
			     starts, counts, data, nItems, mpiDataType);
    PNETCDF_HANDLE_ERROR;
  } // end for loop reading heavy data

  /* 
   * read attributes 
   */
  // Query timeStep (global attribute)
  {
    nc_type timeStep_type;
    MPI_Offset timeStep_len;
    err = ncmpi_inq_att (ncFileId, NC_GLOBAL, "time step", 
			 &timeStep_type, 
			 &timeStep_len);
    PNETCDF_HANDLE_ERROR;
    
    /* read timeStep */
    err = ncmpi_get_att_int(ncFileId, NC_GLOBAL, "time step", &timeStep);
    PNETCDF_HANDLE_ERROR;
    
    if (pnetcdf_verbose and myRank==0)
      std::cout << "input PnetCDF time step: " << timeStep << std::endl;
    
  }

  // Query total time (global attribute)
  {
    nc_type    totalTime_type;
    MPI_Offset totalTime_len;
    err = ncmpi_inq_att (ncFileId, NC_GLOBAL, "total time", 
			 &totalTime_type, 
			 &totalTime_len);
    PNETCDF_HANDLE_ERROR;
    
    /* read total time */
    double timeValue;
    err = ncmpi_get_att_double(ncFileId, NC_GLOBAL, "total time", &timeValue);
    PNETCDF_HANDLE_ERROR;
    
    totalTime = (real_t) timeValue;

    if (pnetcdf_verbose and myRank==0)
      std::cout << "input PnetCDF totalTime: " << totalTime << std::endl;
    
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


// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void write_HDF5(const std::string filename, 
		HostArray<real_t> &U,
		int dimZGlob,
		int timeStep,
		double totalTime)
{

  MpiComm communicator = MpiComm::world();
  int myRank   = communicator.getRank();
  int numTasks = communicator.getNProc();

  //DimensionType dimType = THREE_D;

#ifdef USE_HDF5_PARALLEL

  // verbose log ?
  bool hdf5_verbose = true;
  
  // time measurement variables
  double write_timing, max_write_timing, write_bw;
  MPI_Offset write_size, sum_write_size;
  
  // hdf5 error
  herr_t status;

  /*
   * creation date
   */
  std::string stringDate;
  int stringDateSize;
  if (myRank==0) {
    stringDate = current_date();
    stringDateSize = stringDate.size();
  }
  // broadcast stringDate size to all other MPI tasks
  communicator.bcast(&stringDateSize, 1, MpiComm::INT, 0);

  // broadcast stringDate to all other MPI task
  if (myRank != 0) stringDate.reserve(stringDateSize);
  char* cstr = const_cast<char*>(stringDate.c_str());
  communicator.bcast(cstr, stringDateSize, MpiComm::CHAR, 0);

  // measure time ??
  if (hdf5_verbose) {
    MPI_Barrier( communicator.getComm() );
    write_timing = MPI_Wtime();
  }

  /*
   * write HDF5 file
   */
  // Create a new file using property list with parallel I/O access.
  MPI_Info mpi_info     = MPI_INFO_NULL;
  hid_t    propList_create_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(propList_create_id, communicator.getComm(), mpi_info);
  hid_t    file_id  = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, propList_create_id);
  H5Pclose(propList_create_id);
  
  // Create the data space for the dataset in memory and in file.
  hsize_t  dims_file[3];
  hsize_t  dims_memory[3];
  hsize_t  dims_chunk[3];
  hid_t dataspace_memory;
  //hid_t dataspace_chunk;
  hid_t dataspace_file;
  
  // geometry
  int nx = U.dimx();
  int ny = U.dimy();
  int nz = U.dimz();

  int nzi = dimZGlob/numTasks;

  // if (hdf5_verbose) 
  //   std::cout << "Hdf5 output rank " << myRank << " dim U   " << nx << " " << ny << " " << nz << " " << dimZGlob << "\n"; 

  dims_file[0] = dimZGlob;
  dims_file[1] = ny;
  dims_file[2] = nx;
  dims_memory[0] = nz; 
  dims_memory[1] = ny;
  dims_memory[2] = nx;
  dims_chunk[0] = nz;
  dims_chunk[1] = ny;
  dims_chunk[2] = nx;
  dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
  dataspace_file   = H5Screate_simple(3, dims_file  , NULL);

  // Create the chunked datasets.
  hid_t dataType;
  if (sizeof(real_t) == sizeof(float))
    dataType = H5T_NATIVE_FLOAT;
  else
    dataType = H5T_NATIVE_DOUBLE;
  
  
  /*
   * Memory space hyperslab :
   * select data with or without ghost zones
   */

  hsize_t  start[3] = { myRank*nzi, 0, 0 };
  hsize_t stride[3] = { 1,  1,  1 };
  hsize_t  count[3] = { 1,  1,  1 };
  hsize_t  block[3] = { dims_chunk[0], dims_chunk[1], dims_chunk[2] }; // row-major instead of column-major here
  status = H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start, stride, count, block);
  
  /*
   *
   * write heavy data to HDF5 file
   *
   */
  real_t* data;
  propList_create_id = H5Pcreate(H5P_DATASET_CREATE);
  // if (dimType == TWO_D)
  //   H5Pset_chunk(propList_create_id, 2, dims_chunk);
  // else
  //   H5Pset_chunk(propList_create_id, 3, dims_chunk);
  
  // please note that HDF5 parallel I/O does not support yet filters
  // so we can't use here H5P_deflate to perform compression !!!
  // Weak solution : call h5repack after the file is created
  // (performance of that has not been tested)
  
  // take care not to use parallel specific features if the HDF5
  // library available does not support them !!
  hid_t propList_xfer_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(propList_xfer_id, H5FD_MPIO_COLLECTIVE);
  
  hid_t dataset_id;
  
  /*
   * write density    
   */
  dataset_id = H5Dcreate2(file_id, "/density", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  data = &(U(0,0,0,ID));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
  H5Dclose(dataset_id);

  /*
   * write energy
   */
  dataset_id = H5Dcreate2(file_id, "/energy", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  data = &(U(0,0,0,IP));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
  H5Dclose(dataset_id);

  /*
   * write momentum X
   */
  dataset_id = H5Dcreate2(file_id, "/momentum_x", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  data = &(U(0,0,0,IU));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
  H5Dclose(dataset_id);
  
  /*
   * write momentum Y
   */
  dataset_id = H5Dcreate2(file_id, "/momentum_y", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  data = &(U(0,0,0,IV));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
  H5Dclose(dataset_id);
  
  /*
   * write momentum Z 
   */
  dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  data = &(U(0,0,0,IW));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
  H5Dclose(dataset_id);
  
  
  if (U.nvar() > 5) { // MHD enabled

    // write magnetic field components
    dataset_id = H5Dcreate2(file_id, "/magnetic_field_x", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    data = &(U(0,0,0,IA));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    dataset_id = H5Dcreate2(file_id, "/magnetic_field_y", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    data = &(U(0,0,0,IB));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
    dataset_id = H5Dcreate2(file_id, "/magnetic_field_z", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    data = &(U(0,0,0,IC));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, propList_xfer_id, data);
    H5Dclose(dataset_id);
    
  }

  // write time step number
  hid_t ds_id   = H5Screate(H5S_SCALAR);
  hid_t attr_id;
  {
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "time step", H5T_NATIVE_INT, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &timeStep);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }
  
  // write total time 
  {
    double timeValue = (double) totalTime;
    
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "total time", H5T_NATIVE_DOUBLE, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &timeValue);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }
  
  // close/release resources.
  H5Pclose(propList_create_id);
  H5Pclose(propList_xfer_id);
  H5Sclose(dataspace_memory);
  H5Sclose(dataspace_file);
  H5Fclose(file_id);

  // verbose log about memory bandwidth
  if (hdf5_verbose) {
    
    write_timing = MPI_Wtime() - write_timing;
    
    //write_size = nbVar * U.section() * sizeof(real_t);
    write_size = U.sizeBytes();
    sum_write_size = write_size *  numTasks;
    
    MPI_Reduce(&write_timing, &max_write_timing, 1, MPI_DOUBLE, MPI_MAX, 0, communicator.getComm());
    
    if (myRank==0) {
      printf("########################################################\n");
      printf("################### HDF5 bandwidth #####################\n");
      printf("########################################################\n");
      printf("Local  array size %d x %d x %d reals(%lu bytes), write size = %.2f MB\n",
	     nx,
	     ny,
	     nz,
	     sizeof(real_t),
	     1.0*write_size/1048576.0);
      sum_write_size /= 1048576.0;
      printf("Global array size %d x %d x %d reals(%lu bytes), write size = %.2f GB\n",
	     nx,
	     ny,
	     dimZGlob,
	     sizeof(real_t),
	     1.0*sum_write_size/1024);
      
      write_bw = sum_write_size/max_write_timing;
      printf(" procs    Global array size  exec(sec)  write(MB/s)\n");
      printf("-------  ------------------  ---------  -----------\n");
      printf(" %4d    %4d x %4d x %4d %8.2f  %10.2f\n", numTasks,
	     nx,
	     ny,
	     dimZGlob,
	     max_write_timing, write_bw);
      printf("########################################################\n");
    } // end (myRank == 0)
    
  } // hdf5_verbose
  
#else

  if (myRank == 0) {
    std::cerr << "Parallel HDF5 library is not available ! You can't load a data file for restarting the simulation run !!!" << std::endl;
    std::cerr << "Please install Parallel HDF5 library !!!" << std::endl;
  }

#endif // USE_HDF5_PARALLEL

} // write_HDF5

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void write_Xdmf(const std::string xdmfFilename, 
		const std::string hdf5Filename, 
		int dimX,
		int dimY,
		int dimZ,
		int timeStep,
		double totalTime,
		bool mhdEnabled)
{

  // get data type as a string for Xdmf
  std::string dataTypeName;
  if (sizeof(real_t) == sizeof(float))
    dataTypeName = "Float";
  else
    dataTypeName = "Double";
  
  /*
   * 1. open XDMF and write header lines
   */
  std::fstream xdmfFile;
  xdmfFile.open(xdmfFilename.c_str(), std::ios_base::out);
  
  xdmfFile << "<?xml version=\"1.0\" ?>"                       << std::endl;
  xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>"         << std::endl;
  xdmfFile << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
  xdmfFile << "  <Domain>"                                     << std::endl;
  xdmfFile << "    <Grid Name=\"SomeData\" GridType=\"Uniform\" >" << std::endl;

  // topology CoRectMesh
  xdmfFile << "      <Topology TopologyType=\"3DCoRectMesh\" NumberOfElements=\"" << dimZ << " " << dimY << " " << dimX << "\"/>" << std::endl;

  // geometry
  xdmfFile << "    <Geometry Type=\"ORIGIN_DXDYDZ\">"      << std::endl;
  xdmfFile << "    <DataStructure"                         << std::endl;
  xdmfFile << "       Name=\"Origin\""                     << std::endl;
  xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
  xdmfFile << "       Dimensions=\"3\""                    << std::endl;
  xdmfFile << "       Format=\"XML\">"                     << std::endl;
  xdmfFile << "       0 0 0"                               << std::endl;
  xdmfFile << "    </DataStructure>"                       << std::endl;
  xdmfFile << "    <DataStructure"                         << std::endl;
  xdmfFile << "       Name=\"Spacing\""                    << std::endl;
  xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
  xdmfFile << "       Dimensions=\"3\""                    << std::endl;
  xdmfFile << "       Format=\"XML\">"                     << std::endl;
  xdmfFile << "       1 1 1"                               << std::endl;
  xdmfFile << "    </DataStructure>"                       << std::endl;
  xdmfFile << "    </Geometry>"                            << std::endl;
  
  // density
  xdmfFile << "      <Attribute Center=\"Node\" Name=\"density\">" << std::endl;
  xdmfFile << "        <DataStructure"                             << std::endl;
  xdmfFile << "           DataType=\"" << dataTypeName <<  "\""    << std::endl;
  xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
  xdmfFile << "           Format=\"HDF\">"                         << std::endl;
  xdmfFile << "           "<<hdf5Filename<<":/density"             << std::endl;
  xdmfFile << "        </DataStructure>"                           << std::endl;
  xdmfFile << "      </Attribute>"                                 << std::endl;
  
  // energy
  xdmfFile << "      <Attribute Center=\"Node\" Name=\"energy\">" << std::endl;
  xdmfFile << "        <DataStructure"                              << std::endl;
  xdmfFile << "           DataType=\"" << dataTypeName <<  "\""     << std::endl;
  xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
  xdmfFile << "           Format=\"HDF\">"                          << std::endl;
  xdmfFile << "           "<<hdf5Filename<<":/energy"             << std::endl;
  xdmfFile << "        </DataStructure>"                            << std::endl;
  xdmfFile << "      </Attribute>"                                  << std::endl;
  
  // momentum X
  xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_x\">" << std::endl;
  xdmfFile << "        <DataStructure"                                << std::endl;
  xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
  xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
  xdmfFile << "           Format=\"HDF\">"                            << std::endl;
  xdmfFile << "           "<<hdf5Filename<<":/momentum_x"             << std::endl;
  xdmfFile << "        </DataStructure>"                              << std::endl;
  xdmfFile << "      </Attribute>"                                    << std::endl;
  
  // momentum Y
  xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_y\">" << std::endl;
  xdmfFile << "        <DataStructure" << std::endl;
  xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
  xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
  xdmfFile << "           Format=\"HDF\">"                            << std::endl;
  xdmfFile << "           "<<hdf5Filename<<":/momentum_y"             << std::endl;
  xdmfFile << "        </DataStructure>"                              << std::endl;
  xdmfFile << "      </Attribute>"                                    << std::endl;
  
  if (mhdEnabled) {
    // momentum Z
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_z\">" << std::endl;
    xdmfFile << "        <DataStructure" << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/momentum_z"             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
    
    // magnetic field X
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_x\">" << std::endl;
    xdmfFile << "        <DataStructure" << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_x"             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
    
    // magnetic field Y
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_y\">" << std::endl;
    xdmfFile << "        <DataStructure" << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_y"             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
    
    // magnetic field Z
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_z\">" << std::endl;
    xdmfFile << "        <DataStructure" << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    xdmfFile << "           Dimensions=\"" << dimZ << " " << dimY << " " << dimX << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_z"             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
    
  } // end mhdEnabled
  
  // finalize grid file for the current time step
  xdmfFile << "   </Grid>" << std::endl;
  
  
  /*
   * 3. footer
   */
  xdmfFile << " </Domain>" << std::endl;
  xdmfFile << "</Xdmf>"    << std::endl;

  xdmfFile.close();

} // write_Xdmf

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void reduce (const HostArray<real_t> &U, 
	     HostArray<real_t> &Ureduce,
	     int reduceParam)
{

  int nx = Ureduce.dimx();
  int ny = Ureduce.dimy();
  int nz = Ureduce.dimz();

  if (reduceParam==2) {
    
    // hydro variables (average over a 2x2x2 cube)
    for (int iVar=0; iVar<5; iVar++) {
      
      for (int k=0; k<nz; k++) {
	for (int j=0; j<ny; j++) {
	  for (int i=0; i<nx; i++) {
	    
	    Ureduce(i,j,k,iVar) = 0;
	    
	    Ureduce(i,j,k,iVar) += U(2*i  ,2*j  ,2*k  ,iVar);
	    Ureduce(i,j,k,iVar) += U(2*i+1,2*j  ,2*k  ,iVar);
	    
	    Ureduce(i,j,k,iVar) += U(2*i  ,2*j+1,2*k  ,iVar);
	    Ureduce(i,j,k,iVar) += U(2*i+1,2*j+1,2*k  ,iVar);
	    
	    Ureduce(i,j,k,iVar) += U(2*i  ,2*j  ,2*k+1,iVar);
	    Ureduce(i,j,k,iVar) += U(2*i+1,2*j  ,2*k+1,iVar);
	    
	    Ureduce(i,j,k,iVar) += U(2*i  ,2*j+1,2*k+1,iVar);
	    Ureduce(i,j,k,iVar) += U(2*i+1,2*j+1,2*k+1,iVar);
	    
	    
	  }
	}
      }
      
    }
    
    
    // magnetic field, only average over face
    for (int k=0; k<nz; k++) {
      for (int j=0; j<ny; j++) {
	for (int i=0; i<nx; i++) {
	  
	  // BX
	  Ureduce(i,j,k,IBX) = 0;
	  
	  Ureduce(i,j,k,IBX) += U(2*i  ,2*j  ,2*k  ,IBX);
	  Ureduce(i,j,k,IBX) += U(2*i  ,2*j+1,2*k  ,IBX);
	  Ureduce(i,j,k,IBX) += U(2*i  ,2*j  ,2*k+1,IBX);
	  Ureduce(i,j,k,IBX) += U(2*i  ,2*j+1,2*k+1,IBX);
	  
	  // BY
	  Ureduce(i,j,k,IBY) = 0;
	  
	  Ureduce(i,j,k,IBY) += U(2*i  ,2*j  ,2*k  ,IBY);
	  Ureduce(i,j,k,IBY) += U(2*i+1,2*j  ,2*k  ,IBY);
	  Ureduce(i,j,k,IBY) += U(2*i  ,2*j  ,2*k+1,IBY);
	  Ureduce(i,j,k,IBY) += U(2*i+1,2*j  ,2*k+1,IBY);
	  
	  
	  // BZ
	  Ureduce(i,j,k,IBZ) = 0;
	  
	  Ureduce(i,j,k,IBZ) += U(2*i  ,2*j  ,2*k  ,IBZ);
	  Ureduce(i,j,k,IBZ) += U(2*i+1,2*j  ,2*k  ,IBZ);	
	  Ureduce(i,j,k,IBZ) += U(2*i  ,2*j+1,2*k  ,IBZ);
	  Ureduce(i,j,k,IBZ) += U(2*i+1,2*j+1,2*k  ,IBZ);
	  
	  
	}
      }
    }

  } else if (reduceParam == 4) {

    // hydro variables (average over a 4x4x4 cube)
    for (int iVar=0; iVar<5; iVar++) {
      
      for (int k=0; k<nz; k++) {
	for (int j=0; j<ny; j++) {
	  for (int i=0; i<nx; i++) {
	    
	    Ureduce(i,j,k,iVar) = 0;
	    
	    for (int dk=0; dk<4; dk++) {
	      for (int dj=0; dj<4; dj++) {
		for (int di=0; di<4; di++) {
		  Ureduce(i,j,k,iVar) += U(4*i+di,4*j+dj,4*k+dk,iVar);
		}
	      }
	    }
	    
	    
	  }
	}
      }
      
    }
    
    
    // magnetic field, only average over face
    for (int k=0; k<nz; k++) {
      for (int j=0; j<ny; j++) {
	for (int i=0; i<nx; i++) {
	  
	  // BX
	  Ureduce(i,j,k,IBX) = 0;
	  
	  for (int dk=0; dk<4; dk++) {
	    for (int dj=0; dj<4; dj++) {
	      Ureduce(i,j,k,IBX) += U(4*i   ,4*j+dj,4*k+dk,IBX);
	    }
	  }
	  

	  // BY
	  Ureduce(i,j,k,IBY) = 0;
	  
	  for (int dk=0; dk<4; dk++) {
	    for (int di=0; di<4; di++) {
	      Ureduce(i,j,k,IBY) += U(4*i+di,4*j   ,4*k+dk,IBY);
	    }
	  }
	  	  
	  // BZ
	  Ureduce(i,j,k,IBZ) = 0;
	  
	  for (int dj=0; dj<4; dj++) {
	    for (int di=0; di<4; di++) {
	      Ureduce(i,j,k,IBZ) += U(4*i+di,4*j+dj,4*k   ,IBZ);
	    }
	  }
	  
	}
      }
    }

  }

} // reduce

// -------------------------------------------------------
// -------------------------------------------------------
// -------------------------------------------------------
void projected_density(const HostArray<real_t> &U, 
		       HostArray<double> &Uproj)
{

  // compute projected density on local sub-domain

  int nx = U.dimx();
  int ny = U.dimy();
  int nz = U.dimz();

  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      Uproj(i,j,0,ID) = 0;
    }
  }
  
  for (int k=0; k<nz; k++) {
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
	
	Uproj(i,j,0,ID) += U(i,j,k,ID);
	
      }
    }
  }

} // projected_density
