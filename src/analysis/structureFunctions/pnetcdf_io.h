/**
 * \file pnetcdf_io.h
 *
 * Parallel NetCDF IO.
 *
 * \author P. Kestener
 * \date 20/06/2013
 *
 * $Id: pnetcdf_io.h 3404 2014-05-22 10:43:59Z pkestene $
 */

#ifndef PNETCDF_IO_H_
#define PNETCDF_IO_H_

#include <mpi.h>

#include "utils/config/ConfigMap.h"
#include "hydro/Arrays.h"
using hydroSimu::HostArray;

/**
 * Read netcdf file using Parallel-NetCDF, removes ghosts.
 * Memory allocation for HostArray is done outside.
 *
 * Read variable rho, rho_vx, rho_vy, rho_vz from a sub-domain inside NetCDF file.
 *
 * \param[in]    filename (input data)
 * \param[in]    starts offset to lower left corner coordinates
 * \param[in]    counts sizes of sub-domain
 * \param[inout] localData A reference to a HostArray already allocated
 *
 */
void read_pnetcdf(const std::string &filename,
		  MPI_Offset         starts[3],
		  MPI_Offset         counts[3],
		  HostArray<double> &localData);

/**
 * Write netcdf file using Parallel-NetCDF, with/without ghosts.
 *
 * NetCDF file creation supports:
 * - CDF-2 (using creation mode NC_64BIT_OFFSET)
 * - CDF-5 (using creation mode NC_64BIT_DATA)
 *
 * Write variable rho, rho_vx, rho_vy, rho_vz from a sub-domain inside NetCDF file.
 *
 * \param[in]    filename (input data)
 * \param[in]    starts offset to lower left corner coordinates
 * \param[in]    counts sizes of sub-domain
 * \param[inout] localData A reference to a HostArray (data to write)
 *
 */
void write_pnetcdf(const std::string &filename,
		   MPI_Offset         starts[3],
		   MPI_Offset         counts[3],
		   int                gsizes[3],
		   HostArray<double> &localData,
		   ConfigMap         &configMap);

#endif /* PNETCDF_IO_H_ */
