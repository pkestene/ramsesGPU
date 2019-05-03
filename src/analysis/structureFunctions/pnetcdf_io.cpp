/**
 * \file pnetcdf_io.cpp
 *
 * Parallel NetCDF IO.
 *
 * \author P. Kestener
 * \date 20/06/2013
 *
 * $Id: pnetcdf_io.cpp 3404 2014-05-22 10:43:59Z pkestene $
 */

#include <stdlib.h>
#include <mpi.h>

#include "pnetcdf_io.h"

#include "hydro/constants.h"
#include "common_sf.h" // for enum DataIndex

// for Parallel-netCDF support
#ifdef USE_PNETCDF
#include <pnetcdf.h>

#define PNETCDF_HANDLE_ERROR {                        \
    if (err != NC_NOERR)                              \
        printf("Error at line %d (%s)\n", __LINE__,   \
               ncmpi_strerror(err));                  \
}

#endif // USE_PNETCDF

/*
 * adapted from HydroRunBaseMpi::inputPnetcdf
 */
void read_pnetcdf(const std::string &filename,
		  MPI_Offset         starts[3],
		  MPI_Offset         counts[3],
		  HostArray<double> &localData)
{
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // netcdf file id
  int ncFileId;
  int err;
  
  // file creation mode
  int ncOpenMode = NC_NOWRITE;
  
  int varIds[8];
  //MPI_Offset starts[3], counts[3];
  MPI_Info mpi_info_used;
  
  /* 
   * Open NetCDF file
   */
  err = ncmpi_open(MPI_COMM_WORLD, filename.c_str(), 
		   ncOpenMode,
		   MPI_INFO_NULL, &ncFileId);
  if (err != NC_NOERR) {
    printf("Error: ncmpi_open() file %s (%s)\n",filename.c_str(),ncmpi_strerror(err));
    MPI_Abort(MPI_COMM_WORLD, -1);
    exit(1);
  }

  /*
   * Query NetCDF mode
   */
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

  /*
   * Query information about variables
   */
  {
    int ndims, nvars, ngatts, unlimited;
    err = ncmpi_inq(ncFileId, &ndims, &nvars, &ngatts, &unlimited);
    PNETCDF_HANDLE_ERROR;

    err = ncmpi_inq_varid(ncFileId, "rho", &varIds[ID]);
    PNETCDF_HANDLE_ERROR;
    err = ncmpi_inq_varid(ncFileId, "E", &varIds[IP]);
    PNETCDF_HANDLE_ERROR;
    err = ncmpi_inq_varid(ncFileId, "rho_vx", &varIds[IU]);
    PNETCDF_HANDLE_ERROR;
    err = ncmpi_inq_varid(ncFileId, "rho_vy", &varIds[IV]);
    PNETCDF_HANDLE_ERROR;
    err = ncmpi_inq_varid(ncFileId, "rho_vz", &varIds[IW]);
    PNETCDF_HANDLE_ERROR;    
    err = ncmpi_inq_varid(ncFileId, "Bx", &varIds[IA]);
    PNETCDF_HANDLE_ERROR;
    err = ncmpi_inq_varid(ncFileId, "By", &varIds[IB]);
    PNETCDF_HANDLE_ERROR;
    err = ncmpi_inq_varid(ncFileId, "Bz", &varIds[IC]);
    PNETCDF_HANDLE_ERROR;	
  } // end query information

  /* 
   * Define expected data types (no conversion done here)
   */
  //nc_type ncDataType;
  MPI_Datatype mpiDataType;
  
  //ncDataType  = NC_DOUBLE;
  mpiDataType = MPI_DOUBLE;

  /* 
   * Get all the MPI_IO hints used (just in case, we want to print it after 
   * reading data...
   */
  err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
  PNETCDF_HANDLE_ERROR;

  /*
   * Read heavy data (take care of row-major / column major format !)
   */
  /*counts[IZ] = nx;
  counts[IY] = ny;
  counts[IX] = nz;
  
  starts[IZ] = ghostWidth;
  starts[IY] = ghostWidth;
  starts[IX] = ghostWidth+myRank*nz;*/

  int nItems = counts[IX]*counts[IY]*counts[IZ];

  /*
   * Actual reading (assume double; should probe for float / double)
   */
  {
    double* data;

    // read rho
    data = &(localData(0,0,0,IID));
    err = ncmpi_get_vara_all(ncFileId, varIds[ID], 
			     starts, counts, data, nItems, mpiDataType);
    PNETCDF_HANDLE_ERROR;

    // read rho_vx
    data = &(localData(0,0,0,IIU));
    err = ncmpi_get_vara_all(ncFileId, varIds[IU], 
			     starts, counts, data, nItems, mpiDataType);
    PNETCDF_HANDLE_ERROR;

    // read rho_vy
    data = &(localData(0,0,0,IIV));
    err = ncmpi_get_vara_all(ncFileId, varIds[IV], 
			     starts, counts, data, nItems, mpiDataType);
    PNETCDF_HANDLE_ERROR;

    // read rho_vz
    data = &(localData(0,0,0,IIW));
    err = ncmpi_get_vara_all(ncFileId, varIds[IW], 
			     starts, counts, data, nItems, mpiDataType);
    PNETCDF_HANDLE_ERROR;

  } // end for loop reading heavy data

  /* 
   * close the file 
   */
  err = ncmpi_close(ncFileId);
  PNETCDF_HANDLE_ERROR;

} // read_pnetcdf

/*
 * adapted from HydroRunBaseMpi::outputPnetcdf
 *
 * assumes here that localData have size nx,ny,nz (no ghostWidth)
 *
 * see : test_pnetcdf_write.cpp
 *
 * Note that if ghostIncluded is false local_data must be sized upon nx,ny,nz
 * if not size must be nx+2*ghostWidth,ny+2*ghostWidth,nz+2*ghostWidth 
 *
 */
void write_pnetcdf(const std::string &filename,
		   MPI_Offset         starts[3],
		   MPI_Offset         counts[3],
		   int                gsizes[3],
		   HostArray<double> &localData,
		   ConfigMap         &configMap)
{
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // netcdf file id
  int ncFileId;
  int err;

  // file creation mode
  int ncCreationMode = NC_CLOBBER;
  bool useCDF5 = configMap.getBool("output","pnetcdf_cdf5",false);
  if (useCDF5)
    ncCreationMode = NC_CLOBBER|NC_64BIT_DATA;
  else // use CDF-2 file format
    ncCreationMode = NC_CLOBBER|NC_64BIT_OFFSET;

  // verbose log ?
  bool pnetcdf_verbose = configMap.getBool("output","pnetcdf_verbose",false);
  
  int nbVar=8;
  int dimIds[3], varIds[nbVar];
  MPI_Offset write_size, sum_write_size;
  MPI_Info mpi_info_used;
  char str[512];
  
  // time measurement variables
  double write_timing, max_write_timing, write_bw;

  /* 
   * Create NetCDF file
   */
  err = ncmpi_create(MPI_COMM_WORLD, filename.c_str(), 
		     ncCreationMode,
		     MPI_INFO_NULL, &ncFileId);
  if (err != NC_NOERR) {
    printf("Error: ncmpi_create() file %s (%s)\n",filename.c_str(),ncmpi_strerror(err));
    MPI_Abort(MPI_COMM_WORLD, -1);
    exit(1);
  }

  /*
   * Define dimensions
   */
  err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
  PNETCDF_HANDLE_ERROR;
  
  err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
  PNETCDF_HANDLE_ERROR;
  
  err = ncmpi_def_dim(ncFileId, "z", gsizes[2], &dimIds[2]);
  PNETCDF_HANDLE_ERROR;

  /* 
   * Define variables
   */
  nc_type       ncDataType =  NC_DOUBLE;
  MPI_Datatype mpiDataType = MPI_DOUBLE;

  err = ncmpi_def_var(ncFileId, "rho", ncDataType, 3, dimIds, &varIds[ID]);
  PNETCDF_HANDLE_ERROR;
  err = ncmpi_def_var(ncFileId, "E", ncDataType, 3, dimIds, &varIds[IP]);
  PNETCDF_HANDLE_ERROR;
  err = ncmpi_def_var(ncFileId, "rho_vx", ncDataType, 3, dimIds, &varIds[IU]);
  PNETCDF_HANDLE_ERROR;
  err = ncmpi_def_var(ncFileId, "rho_vy", ncDataType, 3, dimIds, &varIds[IV]);
  PNETCDF_HANDLE_ERROR;
  err = ncmpi_def_var(ncFileId, "rho_vz", ncDataType, 3, dimIds, &varIds[IW]);
  PNETCDF_HANDLE_ERROR;
  
  err = ncmpi_def_var(ncFileId, "Bx", ncDataType, 3, dimIds, &varIds[IA]);
  PNETCDF_HANDLE_ERROR;
  err = ncmpi_def_var(ncFileId, "By", ncDataType, 3, dimIds, &varIds[IB]);
  PNETCDF_HANDLE_ERROR;
  err = ncmpi_def_var(ncFileId, "Bz", ncDataType, 3, dimIds, &varIds[IC]);
  PNETCDF_HANDLE_ERROR;

  /*
   * global attributes
   */
  // did we use CDF-2 or CDF-5
  {
    int useCDF5_int = useCDF5 ? 1 : 0;
    err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &useCDF5_int);
    PNETCDF_HANDLE_ERROR;
  }
  
  /* 
   * exit the define mode 
   */
  err = ncmpi_enddef(ncFileId);
  PNETCDF_HANDLE_ERROR;
  
  /* 
   * Get all the MPI_IO hints used
   */
  err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
  PNETCDF_HANDLE_ERROR;
  
  int nItems = counts[IX]*counts[IY]*counts[IZ];
  
  for (int iVar=0; iVar<nbVar; iVar++) {
    double *data = &(localData(0,0,0,iVar));
    err = ncmpi_put_vara_all(ncFileId, varIds[iVar], starts, counts, data, nItems, mpiDataType);
    PNETCDF_HANDLE_ERROR;
  }

  /* 
   * close the file 
   */
  err = ncmpi_close(ncFileId);
  PNETCDF_HANDLE_ERROR;
  
} // write_pnetcdf
